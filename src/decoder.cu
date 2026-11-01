#include "zerogr/decoder.h"

#include "cuda/cuda_utils.h"
#include "cuda/device_topk.cuh"
#include "cuda/projection_stages.cuh"
#include "cuda/task_topk.cuh"

#include <algorithm>
#include <stdexcept>

namespace zerogr {
namespace {

std::vector<int> BranchLimits(const HostTrie<uint16_t> &trie, int length) {
  if (trie.row_ptr.size() != size_t(trie.stored_states) + 1 ||
      trie.col.size() != trie.edges || trie.edges == 0)
    throw std::invalid_argument("decoder requires a nonempty CSR trie");
  std::vector<int> limits(length, 0);
  uint32_t begin = 0, end = 1;
  for (int d = 0; d < length; ++d) {
    if (begin >= end || end > trie.stored_states)
      throw std::invalid_argument("trie depth does not match sid_length");
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

DecodeConfig CheckedConfig(const DecodeConfig &config) {
  config.Validate();
  return config;
}

cudaDeviceProp CurrentDeviceProperties() {
  int device = 0;
  Check(cudaGetDevice(&device));
  cudaDeviceProp gpu{};
  Check(cudaGetDeviceProperties(&gpu, device));
  return gpu;
}

enum class Projection { kPlan, kDense, kSelective };

struct StageEvents {
  Event project_begin, project_end, score_begin, score_end;
};

}  // namespace

struct Decoder::Impl {
  Impl(const DecodeConfig &o, const HostTrie<uint16_t> &trie,
       const std::vector<float> &hidden, const std::vector<float> &weights)
      : config_(CheckedConfig(o)), limits_(BranchLimits(trie, o.sid_length)), plan_(o.sid_length),
        stored_states_(trie.stored_states), gpu_(CurrentDeviceProperties()),
        blocks_(std::max(1, gpu_.multiProcessorCount * 2)),
        row_ptr_(trie.row_ptr.size()), col_(trie.col.size()),
        hidden_(Product(Product(size_t(o.sid_length), size_t(o.beam_width)), size_t(o.hidden_dim))),
        weights_(Product(size_t(o.vocab_size), size_t(o.hidden_dim))),
        logits_(Product(size_t(o.beam_width), size_t(o.vocab_size))),
        task_count_(1), counts_(o.sid_length), errors_(o.sid_length),
        selected_indices_(o.beam_width), selected_scores_(o.beam_width) {
    if (Product(size_t(o.hidden_dim), sizeof(float)) > gpu_.sharedMemPerBlock)
      throw std::invalid_argument("hidden vector exceeds per-block shared memory");
    row_ptr_.Upload(trie.row_ptr);
    col_.Upload(trie.col);
    for (int d = 0; d < o.sid_length; ++d) {
      const int active = d == 0 ? 1 : o.beam_width;
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
      states_[i].Allocate(o.beam_width);
      scores_[i].Allocate(o.beam_width);
      tokens_[i].Allocate(o.beam_width);
    }
    detail::InitIndices<<<(capacity_ + 255) / 256, 256>>>(indices_.get(), capacity_);
    Check(cudaGetLastError());
    // Query every depth shape; scratch requirements may depend on both count and k.
    size_t max_bytes = 1;
    for (int d = 0; d < o.sid_length; ++d) {
      const int count = (d == 0 ? 1 : o.beam_width) * limits_[d];
      size_t bytes = 0;
      Check(detail::DeviceTopK(nullptr, bytes, candidate_scores_.get(), selected_scores_.get(),
          indices_.get(), selected_indices_.get(), count, std::min(count, o.beam_width)));
      max_bytes = std::max(max_bytes, bytes);
    }
    scratch_.Allocate(max_bytes);
    scratch_bytes_ = max_bytes;
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
    for (int d = 0; d < config_.sid_length; ++d) {
      const int in = d % 2, out = 1 - in;
      const int active = d == 0 ? 1 : config_.beam_width;
      const int count = active * limits_[d];
      const bool dense = projection == Projection::kDense ||
          (projection == Projection::kPlan && plan_[d].projection == ProjectionBackend::kDense);
      const bool small = use_small_topk && plan_[d].topk == TopKBackend::kBlock;
      const float *hidden = hidden_.get() + size_t(d) * config_.beam_width * config_.hidden_dim;
      StageEvents *trace = events ? &(*events)[d] : nullptr;
      if (trace) trace->project_begin.Record();
      if (dense) {
        const float alpha = 1, beta = 0;
        // Row-major H[beam, dim] * W[token, dim]^T via column-major cuBLAS.
        Check(cublasSgemm(blas_.get(), CUBLAS_OP_T, CUBLAS_OP_N,
            config_.vocab_size, active, config_.hidden_dim, &alpha, weights_.get(), config_.hidden_dim,
            hidden, config_.hidden_dim, &beta, logits_.get(), config_.vocab_size));
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
            active, limits_[d], config_.vocab_size, count, candidate_tokens_.get(),
            candidate_states_.get(), candidate_valid_.get(), candidate_scores_.get(),
            tasks_.get(), task_count_.get(), task_capacity_,
            collect_counts ? counts_.get() + d : nullptr, errors_.get() + d);
      }
      Check(cudaGetLastError());
      if (trace) trace->score_begin.Record();
      if (dense) {
        detail::GatherDenseScoreKernel<<<(count + 255) / 256, 256>>>(
            logits_.get(), scores_[in].get(), candidate_tokens_.get(),
            candidate_valid_.get(), limits_[d], count, config_.vocab_size, candidate_scores_.get());
      } else {
        detail::PaddedGroupedSelectiveProjectScoreKernel<<<blocks_, 256, size_t(config_.hidden_dim) * sizeof(float)>>>(
            hidden, weights_.get(), scores_[in].get(), candidate_tokens_.get(),
            tasks_.get(), task_count_.get(), task_capacity_, config_.hidden_dim, config_.vocab_size,
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
        const int selected = std::min(count, config_.beam_width);
        size_t bytes = scratch_bytes_;
        Check(detail::DeviceTopK(scratch_.get(), bytes, candidate_scores_.get(), selected_scores_.get(),
            indices_.get(), selected_indices_.get(), count, selected));
        detail::GatherTopK<<<(config_.beam_width + 255) / 256, 256>>>(candidate_scores_.get(),
            candidate_tokens_.get(), candidate_states_.get(), selected_indices_.get(),
            selected, config_.beam_width, scores_[out].get(), tokens_[out].get(), states_[out].get());
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

  void Calibrate(const CalibrationConfig &calibration) {
    calibration.Validate();
    calibrated = false;
    has_result = false;
    plan_.assign(config_.sid_length, DepthPlan{});
    for (int i = 0; i < calibration.warmup; ++i) {
      Run(Projection::kDense, false);
      CheckErrors();
      Run(Projection::kSelective, false);
      CheckErrors();
    }
    std::vector<StageEvents> events(config_.sid_length);
    std::vector<std::vector<float>> dense_samples(config_.sid_length);
    std::vector<std::vector<float>> selective_samples(config_.sid_length);
    auto sample = [&](Projection projection) {
      Run(projection, false, &events);
      Check(cudaDeviceSynchronize());
      CheckErrors();
      for (int d = 0; d < config_.sid_length; ++d) {
        const auto &e = events[d];
        float us = e.score_end.MicrosecondsSince(e.score_begin);
        if (projection == Projection::kDense) {
          us += e.project_end.MicrosecondsSince(e.project_begin);
          dense_samples[d].push_back(us);
        } else selective_samples[d].push_back(us);
      }
    };
    for (int i = 0; i < calibration.trials; ++i) {
      sample(i % 2 ? Projection::kSelective : Projection::kDense);
      sample(i % 2 ? Projection::kDense : Projection::kSelective);
    }
    for (int d = 0; d < config_.sid_length; ++d) {
      auto &p = plan_[d];
      p.dense_us = Median(dense_samples[d]);
      p.selective_us = Median(selective_samples[d]);
      p.projection = p.dense_us < p.selective_us
          ? ProjectionBackend::kDense : ProjectionBackend::kSelective;
    }
    // Freeze TopK independently, using the selected projection plan.
    Run(Projection::kPlan, false, nullptr, true);
    CheckErrors();
    const auto counts = counts_.Download();
    for (int d = 0; d < config_.sid_length; ++d) {
      auto &p = plan_[d];
      p.candidates = counts[d];
      p.topk = config_.beam_width == detail::kSmallTopKBeamWidth &&
          counts[d] <= detail::kSmallTopKMaxCandidates
          ? TopKBackend::kBlock : TopKBackend::kDevice;
    }
    calibrated = true;
  }

  bool calibrated = false;
  bool has_result = false;
  DecodeConfig config_;
  std::vector<int> limits_;
  std::vector<DepthPlan> plan_;
  uint32_t stored_states_;
  cudaDeviceProp gpu_;
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

Decoder::Decoder(const DecodeConfig &config, const HostTrie<uint16_t> &trie,
                 const std::vector<float> &hidden, const std::vector<float> &weights)
    : impl_(std::make_unique<Impl>(config, trie, hidden, weights)) {}

Decoder::~Decoder() = default;

void Decoder::Calibrate(const CalibrationConfig &config) {
  impl_->Calibrate(config);
}

const std::vector<DepthPlan> &Decoder::plan() const {
  if (!impl_->calibrated) throw std::logic_error("calibrate before reading the plan");
  return impl_->plan_;
}

float Decoder::Decode() {
  if (!impl_->calibrated) throw std::logic_error("calibrate before decoding");
  impl_->has_result = false;
  Event begin, end;
  begin.Record();
  impl_->Run(Projection::kPlan, true);
  end.Record();
  end.Wait();
  const float us = end.MicrosecondsSince(begin);
  impl_->CheckErrors();
  impl_->has_result = true;
  return us;
}

DecodeResult Decoder::Download() const {
  if (!impl_->has_result) throw std::logic_error("decode before downloading results");
  const int output = impl_->config_.sid_length % 2;
  return {impl_->scores_[output].Download(), impl_->tokens_[output].Download(),
          impl_->states_[output].Download()};
}

}  // namespace zerogr
