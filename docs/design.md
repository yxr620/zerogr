# Design

ZeroGR separates its decoder library from the synthetic workload. Public headers
in `include/zerogr/` use standard C++ types; CUDA storage and kernel launches
remain inside `src/`.

## Modules

- `config.h`: decode dimensions and calibration settings.
- `trie.h`: CPU construction of a leaf-trimmed, value-free CSR trie.
- `decoder.h`: device input ownership, per-depth planning and beam decoding.
- `src/cuda/`: lookup, task queues, projection, TopK and CUDA resource helpers.
- `examples/`: command-line options, synthetic inputs and result reporting.

## Execution

1. Generate fixed-length synthetic semantic IDs, sort them and remove duplicates.
2. Build the CSR trie. Root state is zero; an edge's child state is its index
   plus one. Complete semantic IDs have virtual leaf states without stored rows.
3. Construct a decoder, uploading the trie, hidden vectors and vocabulary weights.
4. Calibrate projection costs and candidate counts, then freeze a plan per depth.
5. Decode using that plan and download the final beams.

Dense scoring projects the full vocabulary with cuBLAS, then gathers legal
candidate logits. Selective scoring builds tasks over legal candidates and
computes grouped hidden/weight dot products. A candidate's score is its parent
score plus its dot product.

Calibration compares the median dense GEMM + gather cost with the median
selective score cost. Ties select the selective path. Beam width 128 with at
most 512 actual candidates uses block TopK fused with output gathering; other
depths use DeviceTopK and a separate gather.

## API and input shapes

`Decoder` owns device copies of its inputs and workspace. The input vectors are
FP32, with hidden shape `[sid_length, beam_width, hidden_dim]` and weight shape
`[vocab_size, hidden_dim]`. Synthetic hidden vectors are identical across beams
within each depth, keeping the workload independent of TopK output order.

Call `Calibrate()` before `Decode()`. `plan()` returns projection and TopK choices
for each depth. `Decode()` waits for completion and returns CUDA event time in
microseconds; allocation, calibration, error checks and result downloads are
outside that interval. `Download()` returns the final scores, last tokens and
trie states. Padding entries have negative-infinity scores. The plan applies
to the fixed inputs owned by that decoder.
