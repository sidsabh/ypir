# PIR GEMM Benchmark

Measures GPU throughput for SimplePIR-style matrix multiplication:
- **DB**: `2^15 × 2^15` matrix of `uint32_t` (~4 GB)
- **Queries**: `2^15 × batch_size` matrix of `uint32_t` or `uint64_t`
- **Output**: `2^15 × batch_size` matrix matching query type

This uses NVIDIA CUTLASS for the GEMM kernel, which supports
mixed uint32 × uint64 multiplication that cuBLAS cannot do.

## Prerequisites

- CUDA Toolkit >= 11.4
- An NVIDIA GPU with SM >= 70 (V100, A100, H100, etc.)
- CUTLASS headers (header-only, no build needed)

## Setup

```bash
# Clone CUTLASS (header-only, just need the includes)
git clone https://github.com/NVIDIA/cutlass.git ~/cutlass

# Build the benchmark
cd pir_bench
cmake -S . -B build -DCUTLASS_DIR=$HOME/cutlass
cmake --build build -j$(nproc)
```

## For different GPUs

```bash
# 1080/1080Ti (default)
cmake -S . -B build -DCUTLASS_DIR=$HOME/cutlass -DCMAKE_CUDA_ARCHITECTURES=61

# V100
cmake -S . -B build -DCUTLASS_DIR=$HOME/cutlass -DCMAKE_CUDA_ARCHITECTURES=70

# A100
cmake -S . -B build -DCUTLASS_DIR=$HOME/cutlass -DCMAKE_CUDA_ARCHITECTURES=80

# H100
cmake -S . -B build -DCUTLASS_DIR=$HOME/cutlass -DCMAKE_CUDA_ARCHITECTURES=90
```

**Important**: Use CUTLASS 2.x branch for SM61 (1080). CUTLASS 3.x requires SM70+.
```bash
# For 1080:
git clone --branch v2.11.0 https://github.com/NVIDIA/cutlass.git ~/cutlass
# For V100/A100/H100, main branch is fine:
git clone https://github.com/NVIDIA/cutlass.git ~/cutlass
```

## Run

```bash
# Default: 32-bit queries, batch sizes 1-64
./build/pir_bench

# 64-bit queries (SimplePIR with q = 2^64)
./build/pir_bench --bits 64

# Custom batch sizes
./build/pir_bench --bits 32 --batches 1,2,4,8,16,32,64,128,256

# Smaller matrix for quick test (2^12 × 2^12 = ~64 MB)
./build/pir_bench --log_dim 12

# More iterations for stable measurements
./build/pir_bench --warmup 5 --iters 20
```

## Output Columns

- **Batch**: Number of concurrent client queries
- **Time (ms)**: Average GEMM execution time
- **QPS**: Queries per second (batch / time)
- **Eff Tput (GB/s)**: `(DB_size × batch) / time` — the amortized metric used in PIR papers. Can exceed HBM bandwidth because the DB is read once for all queries.
- **HW BW (GB/s)**: `(DB + queries + output) / time` — approximate actual memory traffic. Should approach but not exceed the GPU's theoretical HBM bandwidth.

## Expected Results (A100 80GB)

For a 4 GB database with 32-bit queries:

| Batch | Eff Tput (GB/s) | HW BW (GB/s) |
|-------|----------------|---------------|
| 1     | ~800-1200      | ~800-1200     |
| 8     | ~4000-8000     | ~1000-1500    |
| 64    | ~20000+        | ~1500-2000    |

The effective throughput scales roughly linearly with batch size
because the DB read is amortized. The HW bandwidth saturates
at the A100's theoretical ~2 TB/s.