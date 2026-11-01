#include "cuda_utils.h"
#include "projection_stages.cuh"
#include "task_topk.cuh"
#include "trie_builder.h"

#include <cub/device/device_topk.cuh>
#include <cuda/stream>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace zerogr {
namespace {

struct Options {
  int samples = 10000;
  int vocab = 256;
  int length = 5;
  int hidden = 256;
  int beam = 128;
  int seed = 1234;
  int warmup = 2;
  int trials = 5;
};

void Usage() {
  std::cout << "ZeroGR: synthetic constrained decoding\n"
    "  --num-samples N  sampled SID rows (default 10000)\n"
    "  --vocab-size N   1..65536 (default 256)\n"
    "  --sid-length N   tokens per SID (default 5)\n"
    "  --hidden-dim N   FP32 projection dimension (default 256)\n"
    "  --beam-width N   64..512; block TopK uses 128 (default 128)\n"
    "  --seed N         nonnegative seed (default 1234)\n"
    "  --warmup N       warmups per path (default 2)\n"
    "  --trials N       calibration and decode trials (default 5)\n";
}

Options ParseArgs(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--help" || key == "-h") { Usage(); std::exit(0); }
    int *value = nullptr;
    if (key == "--num-samples") value = &o.samples;
    else if (key == "--vocab-size") value = &o.vocab;
    else if (key == "--sid-length") value = &o.length;
    else if (key == "--hidden-dim") value = &o.hidden;
    else if (key == "--beam-width") value = &o.beam;
    else if (key == "--seed") value = &o.seed;
    else if (key == "--warmup") value = &o.warmup;
    else if (key == "--trials") value = &o.trials;
    else throw std::invalid_argument("unknown option: " + key);
    if (++i == argc) throw std::invalid_argument("missing value for " + key);
    const std::string text = argv[i];
    size_t consumed = 0;
    const long long n = std::stoll(text, &consumed);
    if (consumed != text.size() || n < 0 || n > std::numeric_limits<int>::max())
      throw std::invalid_argument("invalid value for " + key);
    *value = static_cast<int>(n);
  }
  if (o.samples == 0 || o.vocab == 0 || o.vocab > 65536 || o.length == 0 ||
      o.hidden == 0 || o.beam < 64 || o.beam > 512 || o.trials == 0)
    throw std::invalid_argument("invalid shape; see --help");
  return o;
}

HostTrie<uint16_t> SyntheticTrie(const Options &o, size_t *unique_rows) {
  std::mt19937_64 rng(o.seed);
  std::uniform_int_distribution<int> token(0, o.vocab - 1);
  std::vector<std::vector<uint16_t>> rows(o.samples, std::vector<uint16_t>(o.length));
  for (auto &row : rows)
    for (auto &t : row) t = static_cast<uint16_t>(token(rng));
  std::sort(rows.begin(), rows.end());
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
  *unique_rows = rows.size();
  std::vector<uint16_t> flat;
  flat.reserve(Product(rows.size(), size_t(o.length)));
  for (const auto &row : rows) flat.insert(flat.end(), row.begin(), row.end());
  return BuildTrieFromSortedUnique(flat.data(), rows.size(), o.length);
}

std::vector<int> BranchLimits(const HostTrie<uint16_t> &trie, int length) {
  std::vector<int> limits(length, 0);
  uint32_t begin = 0, end = 1;
  for (int d = 0; d < length; ++d) {
    for (uint32_t s = begin; s < end; ++s)
      limits[d] = std::max(limits[d], int(trie.row_ptr[s + 1] - trie.row_ptr[s]));
    const uint32_t next_begin = trie.row_ptr[begin] + 1;
    end = trie.row_ptr[end] + 1;
    begin = next_begin;
  }
  return limits;
}

float Median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  const size_t middle = values.size() / 2;
  return values.size() % 2 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
}

__global__ void InitIndices(uint32_t *indices, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) indices[i] = static_cast<uint32_t>(i);
}

// DeviceTopK returns unsorted pairs; the next step consumes the same beam order.
__global__ void GatherTopK(const float *candidate_scores, const uint16_t *candidate_tokens,
    const uint32_t *candidate_states, const uint32_t *indices, int selected, int beam,
    float *scores, uint16_t *tokens, uint32_t *states) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= beam) return;
  const uint32_t index = i < selected ? indices[i] : 0;
  const bool valid = i < selected && isfinite(candidate_scores[index]);
  scores[i] = valid ? candidate_scores[index] : -__int_as_float(0x7f800000);
  tokens[i] = valid ? candidate_tokens[index] : 0;
  states[i] = valid ? candidate_states[index] : 0;
}

cudaError_t DeviceTopK(void *storage, size_t &bytes, const float *scores,
    float *selected_scores, const uint32_t *indices, uint32_t *selected_indices,
    int count, int k) {
  cuda::stream_ref stream{cudaStream_t{}};
  auto requirements = cuda::execution::require(
      cuda::execution::determinism::not_guaranteed,
      cuda::execution::output_ordering::unsorted);
  auto env = cuda::std::execution::env{stream, requirements};
  return cub::DeviceTopK::MaxPairs(storage, bytes, scores, selected_scores,
      indices, selected_indices, count, k, env);
}

enum class Projection { kPlan, kDense, kSelective };

struct DepthPlan {
  bool dense = false;
  bool small_topk = false;
  uint64_t candidates = 0;
  std::vector<float> dense_us, selective_us;
};

struct StageEvents {
  Event project_begin, project_end, score_begin, score_end;
};

class Decoder {
 public:
  Decoder(const Options &o, const HostTrie<uint16_t> &trie, const cudaDeviceProp &gpu)
      : o_(o), limits_(BranchLimits(trie, o.length)), plan_(o.length),
        stored_states_(trie.stored_states), blocks_(std::max(1, gpu.multiProcessorCount * 2)),
        row_ptr_(trie.row_ptr.size()), col_(trie.col.size()),
        hidden_(Product(Product(size_t(o.length), size_t(o.beam)), size_t(o.hidden))),
        weights_(Product(size_t(o.vocab), size_t(o.hidden))),
        logits_(Product(size_t(o.beam), size_t(o.vocab))),
        task_count_(1), counts_(o.length), errors_(o.length),
        selected_indices_(o.beam), selected_scores_(o.beam) {
    if (Product(size_t(o.hidden), sizeof(float)) > gpu.sharedMemPerBlock)
      throw std::invalid_argument("hidden vector exceeds per-block shared memory");
    row_ptr_.Upload(trie.row_ptr);
    col_.Upload(trie.col);
    for (int d = 0; d < o.length; ++d) {
      const int active = d == 0 ? 1 : o.beam;
      capacity_ = std::max(capacity_, active * limits_[d]);
      task_capacity_ = std::max(task_capacity_, active * int(detail::TaskCountForDegree(limits_[d])));
    }
    candidate_scores_.Allocate(capacity_);
    candidate_tokens_.Allocate(capacity_);
    candidate_states_.Allocate(capacity_);
    candidate_valid_.Allocate(capacity_);
    indices_.Allocate(capacity_);
    tasks_.Allocate(task_capacity_);
    for (int i = 0; i < 2; ++i) {
      states_[i].Allocate(o.beam);
      scores_[i].Allocate(o.beam);
      tokens_[i].Allocate(o.beam);
    }
    InitIndices<<<(capacity_ + 255) / 256, 256>>>(indices_.get(), capacity_);
    Check(cudaGetLastError());
    // Query every depth shape; scratch requirements may depend on both count and k.
    size_t max_bytes = 1;
    for (int d = 0; d < o.length; ++d) {
      const int count = (d == 0 ? 1 : o.beam) * limits_[d];
      size_t bytes = 0;
      Check(DeviceTopK(nullptr, bytes, candidate_scores_.get(), selected_scores_.get(),
          indices_.get(), selected_indices_.get(), count, std::min(count, o.beam)));
      max_bytes = std::max(max_bytes, bytes);
    }
    scratch_.Allocate(max_bytes);
    scratch_bytes_ = max_bytes;
    std::mt19937_64 rng(uint64_t(o.seed) + 1);
    std::uniform_real_distribution<float> value(-0.1f, 0.1f);
    std::vector<float> hidden(Product(Product(size_t(o.length), size_t(o.beam)), size_t(o.hidden)));
    std::vector<float> weights(Product(size_t(o.vocab), size_t(o.hidden)));
    // Beam order differs between TopK backends. Keep the synthetic hidden
    // vector identical across beams at each depth.
    for (int d = 0; d < o.length; ++d) {
      const size_t begin = size_t(d) * o.beam * o.hidden;
      for (int h = 0; h < o.hidden; ++h) hidden[begin + h] = value(rng);
      for (int b = 1; b < o.beam; ++b)
        std::copy_n(hidden.data() + begin, o.hidden, hidden.data() + begin + size_t(b) * o.hidden);
    }
    for (auto &x : weights) x = value(rng);
    hidden_.Upload(hidden);
    weights_.Upload(weights);
    Check(cudaDeviceSynchronize());
  }

  void Run(Projection projection, bool use_small_topk,
           std::vector<StageEvents> *events = nullptr, bool collect_counts = false) {
    states_[0].Clear();
    scores_[0].Clear();
    errors_.Clear();
    if (collect_counts) counts_.Clear();
    for (int d = 0; d < o_.length; ++d) {
      const int in = d % 2, out = 1 - in;
      const int active = d == 0 ? 1 : o_.beam;
      const int count = active * limits_[d];
      const bool dense = projection == Projection::kDense ||
          (projection == Projection::kPlan && plan_[d].dense);
      const bool small = use_small_topk && plan_[d].small_topk;
      const float *hidden = hidden_.get() + size_t(d) * o_.beam * o_.hidden;
      StageEvents *trace = events ? &(*events)[d] : nullptr;
      if (trace) trace->project_begin.Record();
      if (dense) {
        const float alpha = 1, beta = 0;
        // Row-major H[beam, dim] * W[token, dim]^T via column-major cuBLAS.
        Check(cublasSgemm(blas_.get(), CUBLAS_OP_T, CUBLAS_OP_N,
            o_.vocab, active, o_.hidden, &alpha, weights_.get(), o_.hidden,
            hidden, o_.hidden, &beta, logits_.get(), o_.vocab));
      }
      if (trace) trace->project_end.Record();
      if (dense && !small && !collect_counts) {
        detail::LookupCandidatesKernel<<<(count + 255) / 256, 256>>>(
            states_[in].get(), scores_[in].get(), row_ptr_.get(), col_.get(),
            active, limits_[d], candidate_tokens_.get(), candidate_states_.get(),
            candidate_valid_.get(), nullptr);
      } else {
        task_count_.Clear();
        detail::PaddedLookupAndBuildTasksKernel<<<(count + 255) / 256, 256>>>(
            states_[in].get(), scores_[in].get(), row_ptr_.get(), stored_states_, col_.get(),
            active, limits_[d], o_.vocab, count, candidate_tokens_.get(),
            candidate_states_.get(), candidate_valid_.get(), candidate_scores_.get(),
            tasks_.get(), task_count_.get(), task_capacity_,
            collect_counts ? counts_.get() + d : nullptr, errors_.get() + d);
      }
      Check(cudaGetLastError());
      if (trace) trace->score_begin.Record();
      if (dense) {
        detail::GatherDenseScoreKernel<<<(count + 255) / 256, 256>>>(
            logits_.get(), scores_[in].get(), candidate_tokens_.get(),
            candidate_valid_.get(), limits_[d], count, o_.vocab, candidate_scores_.get());
      } else {
        detail::PaddedGroupedSelectiveProjectScoreKernel<<<blocks_, 256, size_t(o_.hidden) * sizeof(float)>>>(
            hidden, weights_.get(), scores_[in].get(), candidate_tokens_.get(),
            tasks_.get(), task_count_.get(), task_capacity_, o_.hidden, o_.vocab,
            active, count, candidate_scores_.get(), errors_.get() + d);
      }
      Check(cudaGetLastError());
      if (trace) trace->score_end.Record();
      if (small) {
        detail::TaskAwareSmallTopKAndPostKernel<<<1, detail::kSmallTopKThreads>>>(
            candidate_scores_.get(), candidate_tokens_.get(), candidate_states_.get(),
            tasks_.get(), task_count_.get(), task_capacity_, count, active,
            scores_[out].get(), tokens_[out].get(), states_[out].get(), errors_.get() + d);
      } else {
        const int selected = std::min(count, o_.beam);
        size_t bytes = scratch_bytes_;
        Check(DeviceTopK(scratch_.get(), bytes, candidate_scores_.get(), selected_scores_.get(),
            indices_.get(), selected_indices_.get(), count, selected));
        GatherTopK<<<(o_.beam + 255) / 256, 256>>>(candidate_scores_.get(),
            candidate_tokens_.get(), candidate_states_.get(), selected_indices_.get(),
            selected, o_.beam, scores_[out].get(), tokens_[out].get(), states_[out].get());
      }
      Check(cudaGetLastError());
    }
  }

  void CheckErrors() const {
    const auto errors = errors_.Download();
    for (size_t d = 0; d < errors.size(); ++d)
      if (errors[d]) throw std::runtime_error("device error at depth " +
          std::to_string(d) + ": " + std::to_string(errors[d]));
  }

  void Calibrate() {
    for (int i = 0; i < o_.warmup; ++i) {
      Run(Projection::kDense, false);
      CheckErrors();
      Run(Projection::kSelective, false);
      CheckErrors();
    }
    std::vector<StageEvents> events(o_.length);
    auto sample = [&](Projection projection) {
      Run(projection, false, &events);
      Check(cudaDeviceSynchronize());
      CheckErrors();
      for (int d = 0; d < o_.length; ++d) {
        const auto &e = events[d];
        float us = e.score_end.MicrosecondsSince(e.score_begin);
        if (projection == Projection::kDense) {
          us += e.project_end.MicrosecondsSince(e.project_begin);
          plan_[d].dense_us.push_back(us);
        } else plan_[d].selective_us.push_back(us);
      }
    };
    for (int i = 0; i < o_.trials; ++i) {
      sample(i % 2 ? Projection::kSelective : Projection::kDense);
      sample(i % 2 ? Projection::kDense : Projection::kSelective);
    }
    for (auto &p : plan_) p.dense = Median(p.dense_us) < Median(p.selective_us);
    // Freeze TopK independently, using a decode with the selected projection plan.
    Run(Projection::kPlan, false, nullptr, true);
    CheckErrors();
    const auto counts = counts_.Download();
    for (int d = 0; d < o_.length; ++d) {
      auto &p = plan_[d];
      p.candidates = counts[d];
      p.small_topk = o_.beam == detail::kSmallTopKBeamWidth &&
          counts[d] <= detail::kSmallTopKMaxCandidates;
      std::cout << "depth=" << d << " candidates=" << p.candidates
          << " projection=" << (p.dense ? "dense" : "selective")
          << " dense_us=" << Median(p.dense_us)
          << " selective_us=" << Median(p.selective_us)
          << " topk=" << (p.small_topk ? "block" : "device") << '\n';
    }
  }

  void PrintFinal() const {
    const auto scores = scores_[o_.length % 2].Download();
    const auto tokens = tokens_[o_.length % 2].Download();
    const auto best = std::max_element(scores.begin(), scores.end());
    const auto finite = std::count_if(scores.begin(), scores.end(), [](float x) { return std::isfinite(x); });
    std::cout << "finite_beams=" << finite << " best_score=" << *best
        << " last_token=" << tokens[size_t(best - scores.begin())] << '\n';
  }

 private:
  Options o_;
  std::vector<int> limits_;
  std::vector<DepthPlan> plan_;
  uint32_t stored_states_;
  int blocks_, capacity_ = 0, task_capacity_ = 0;
  size_t scratch_bytes_ = 0;
  Blas blas_;
  DeviceBuffer<uint32_t> row_ptr_;
  DeviceBuffer<uint16_t> col_;
  DeviceBuffer<float> hidden_, weights_, logits_;
  DeviceBuffer<float> scores_[2], candidate_scores_;
  DeviceBuffer<uint16_t> tokens_[2], candidate_tokens_;
  DeviceBuffer<uint32_t> states_[2], candidate_states_;
  DeviceBuffer<uint8_t> candidate_valid_;
  DeviceBuffer<detail::PaddedProjectionTask> tasks_;
  DeviceBuffer<uint32_t> task_count_;
  DeviceBuffer<uint64_t> counts_;
  DeviceBuffer<uint32_t> errors_, selected_indices_;
  DeviceBuffer<float> selected_scores_;
  DeviceBuffer<uint32_t> indices_;
  DeviceBuffer<uint8_t> scratch_;
};

int Main(int argc, char **argv) {
  const Options o = ParseArgs(argc, argv);
  int device = 0;
  Check(cudaGetDevice(&device));
  cudaDeviceProp gpu{};
  Check(cudaGetDeviceProperties(&gpu, device));
  size_t unique_rows = 0;
  const auto trie = SyntheticTrie(o, &unique_rows);
  std::cout << "ZeroGR: GPU=" << gpu.name << " sampled=" << o.samples
      << " unique=" << unique_rows << " stored_states=" << trie.stored_states
      << " edges=" << trie.edges << '\n' << std::fixed << std::setprecision(3);
  Decoder decoder(o, trie, gpu);
  decoder.Calibrate();
  for (int i = 0; i < o.warmup; ++i) {
    decoder.Run(Projection::kPlan, true);
    decoder.CheckErrors();
  }
  std::vector<float> samples;
  Event begin, end;
  for (int i = 0; i < o.trials; ++i) {
    begin.Record();
    decoder.Run(Projection::kPlan, true);
    end.Record();
    end.Wait();
    samples.push_back(end.MicrosecondsSince(begin));
    decoder.CheckErrors();
  }
  std::cout << "ZeroGR median_decode_us=" << Median(samples) << '\n';
  decoder.PrintFinal();
  return 0;
}

}  // namespace
}  // namespace zerogr

int main(int argc, char **argv) {
  try { return zerogr::Main(argc, argv); }
  catch (const std::exception &error) {
    std::cerr << "ZeroGR: " << error.what() << '\n';
    return 1;
  }
}
