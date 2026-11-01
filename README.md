# ZeroGR

Minimal CUDA implementation of ZeroGR for synthetic fixed-length semantic IDs:
leaf-trimmed, value-free CSR; grouped selective projection; per-depth selection
between full-vocabulary GEMM + gather and selective scoring; hybrid TopK.

This small example uses one request, FP32 hidden vectors/weights/scores and
uint16 tokens. Inputs are generated in memory; hidden vectors are shared across
beams within each depth. Each candidate score is its parent score plus the
hidden/weight dot product. Projection calibration and
candidate counts produce a fixed plan for these inputs; calibration is excluded
from the reported decode time.

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

Set the CUDA architecture for your GPU. `--help` lists the few available options.
The program prints its plan, median decode time and final beam summary to stdout.
