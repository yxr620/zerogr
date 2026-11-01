# ZeroGR

ZeroGR is a CUDA implementation of trie-constrained decoding for fixed-length
semantic IDs. It combines leaf-trimmed, value-free CSR, grouped selective
projection, per-depth projection selection and hybrid TopK.

The synthetic example generates inputs in memory and uses one request, FP32
hidden vectors/weights/scores and uint16 tokens. It prints the calibrated plan,
median decode time and final beam summary.

## Repository layout

| Directory | Contents |
| --- | --- |
| `include/zerogr/` | Configuration, trie and decoder interfaces |
| `src/` | Core library, trie builder and internal CUDA kernels |
| `examples/` | Synthetic data generator and command-line driver |
| `cmake/` | CUDA and CCCL dependency configuration |
| `docs/` | [Design and API overview](docs/design.md) |

## Build and run

On a Linux machine with an NVIDIA GPU, CUDA Toolkit, CMake and an external
[NVIDIA CCCL 3.3.4 checkout](https://github.com/NVIDIA/cccl/tree/v3.3.4):

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=90 -DCCCL_ROOT=/path/to/cccl-3.3.4
cmake --build build -j
./build/zerogr --num-samples 10000 --vocab-size 256 --sid-length 5 \
  --hidden-dim 256 --beam-width 128
```

Set the CUDA architecture for your GPU. `--help` lists the available options.
The core library is the CMake target `ZeroGR::zerogr`; `zerogr_synthetic` builds
the executable shown above.
