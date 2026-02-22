/**
 * PIR GEMM Benchmark
 *
 * Measures throughput of DB × queries for SimplePIR-style computation.
 *
 * Modes:
 *   db32_q32  — uint32 DB × uint32 queries  (baseline)
 *   db32_q64  — uint32 DB × uint64 queries
 *   db16_q32  — uint16 DB × uint32 queries  (half DB size)
 *   db16_q64  — uint16 DB × uint64 queries
 *   db16_crt     — CRT: two (uint16 DB × uint32 queries), combined time
 *   db8_q32      — uint8 DB × uint32 queries → uint32 accum (DoublePIR Step 1, SIMT)
 *   db32_crt_i64 — CRT: two (int32 DB × int32 queries → int64 accum), matches actual CUTLASS kernel
 *   db8_q32_tc   — Tensor core: 4x int8 GEMM with alpha/beta folding (DoublePIR Step 1, cuBLAS)
 *
 * Usage:
 *   ./pir_bench [--mode db32_q32|db32_q64|db16_q32|db16_q64|db16_crt|db8_q32|db32_crt_i64|db8_q32_tc|all]
 *               [--batches 1,2,4,...] [--log_dim N] [--warmup N] [--iters N]
 *
 * Requires CUTLASS headers (clone https://github.com/NVIDIA/cutlass)
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <sstream>

#include <cublas_v2.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"

// ---------- Error checking ----------

#define CUDA_CHECK(status)                                           \
    {                                                                \
        cudaError_t error = status;                                  \
        if (error != cudaSuccess) {                                  \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) \
                      << " at " << __FILE__ << ":" << __LINE__       \
                      << std::endl;                                  \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    }

#define CUTLASS_CHECK(status)                                        \
    {                                                                \
        cutlass::Status error = status;                              \
        if (error != cutlass::Status::kSuccess) {                    \
            std::cerr << "CUTLASS error: "                           \
                      << cutlassGetStatusString(error)               \
                      << " at " << __FILE__ << ":" << __LINE__       \
                      << std::endl;                                  \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    }

#define CUBLAS_CHECK(status)                                          \
    {                                                                \
        cublasStatus_t error = status;                               \
        if (error != CUBLAS_STATUS_SUCCESS) {                        \
            std::cerr << "cuBLAS error: " << (int)error              \
                      << " at " << __FILE__ << ":" << __LINE__       \
                      << std::endl;                                  \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    }

// ---------- Kernel: decompose int32 query into 4 int8 byte slices ----------

__global__ void decompose_query_bytes_kernel(
    int8_t* __restrict__ q0,
    int8_t* __restrict__ q1,
    int8_t* __restrict__ q2,
    int8_t* __restrict__ q3,
    const int32_t* __restrict__ query,
    size_t count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < count) {
        int32_t v = query[idx];
        q0[idx] = (int8_t)(v & 0xFF);
        q1[idx] = (int8_t)((v >> 8) & 0xFF);
        q2[idx] = (int8_t)((v >> 16) & 0xFF);
        q3[idx] = (int8_t)((v >> 24) & 0xFF);
    }
}

// ---------- Benchmark kernel ----------

template <typename DbT, typename QueryT>
struct PirBench {
    using Rows = cutlass::layout::RowMajor;
    using Cols = cutlass::layout::ColumnMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        DbT,    Rows,   // A (database)
        QueryT, Cols,   // B (queries)
        QueryT, Rows    // C/D (output)
    >;

    static void print_header(uint64_t M, uint64_t K, int warmup, int iters, const std::string& label) {
        const double GiB = 1024.0 * 1024.0 * 1024.0;
        std::cout << "\n=== " << label << " ===" << std::endl;
        std::cout << "DB: " << M << " x " << K << " uint" << (sizeof(DbT) * 8)
                  << " (" << std::fixed << std::setprecision(1)
                  << (double(M) * K * sizeof(DbT)) / GiB << " GiB)" << std::endl;
        std::cout << "Query/Output type: uint" << (sizeof(QueryT) * 8) << std::endl;

        size_t free_mem, total_mem;
        CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
        std::cout << "GPU memory: " << std::fixed << std::setprecision(1)
                  << total_mem / GiB << " GiB total, "
                  << free_mem / GiB << " GiB free" << std::endl;

        cudaDeviceProp props;
        CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
        std::cout << "GPU: " << props.name << std::endl;
        std::cout << "HBM BW (theoretical): ~"
                  << (2.0 * props.memoryClockRate * (props.memoryBusWidth / 8.0) / 1.0e6)
                  << " GB/s" << std::endl;
        std::cout << std::string(72, '-') << std::endl;

        std::cout << std::setw(8) << "Batch"
                  << std::setw(14) << "Time (ms)"
                  << std::setw(14) << "QPS"
                  << std::setw(18) << "Eff Tput (GB/s)"
                  << std::setw(18) << "HW BW (GB/s)"
                  << std::endl;
        std::cout << std::string(72, '-') << std::endl;
    }

    static void run(uint64_t M, uint64_t K, const std::vector<int>& batch_sizes,
                    int warmup, int iters, const std::string& label) {
        print_header(M, K, warmup, iters, label);

        // Allocate DB once
        cutlass::HostTensor<DbT, Rows> tensor_A({int(M), int(K)});
        cutlass::reference::host::TensorFill(tensor_A.host_view());
        tensor_A.sync_device();

        for (int N : batch_sizes) {
            cutlass::gemm::GemmCoord problem_size(M, N, K);

            cutlass::HostTensor<QueryT, Cols> tensor_B(problem_size.kn());
            cutlass::HostTensor<QueryT, Rows> tensor_C(problem_size.mn());
            cutlass::HostTensor<QueryT, Rows> tensor_D(problem_size.mn());

            cutlass::reference::host::TensorFill(tensor_B.host_view());
            cutlass::reference::host::TensorFill(tensor_C.host_view());
            cutlass::reference::host::TensorFill(tensor_D.host_view());

            tensor_B.sync_device();
            tensor_C.sync_device();
            tensor_D.sync_device();

            QueryT alpha = QueryT(1);
            QueryT beta  = QueryT(0);

            typename Gemm::Arguments arguments{
                problem_size,
                tensor_A.device_ref(),
                tensor_B.device_ref(),
                tensor_C.device_ref(),
                tensor_D.device_ref(),
                {alpha, beta},
                1
            };

            Gemm gemm_op;
            cutlass::Status status = gemm_op.can_implement(arguments);
            if (status != cutlass::Status::kSuccess) {
                std::cerr << "Cannot implement GEMM for batch=" << N
                          << ": " << cutlassGetStatusString(status) << std::endl;
                continue;
            }

            size_t workspace_size = Gemm::get_workspace_size(arguments);
            cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
            status = gemm_op.initialize(arguments, workspace.get());
            CUTLASS_CHECK(status);

            for (int i = 0; i < warmup; i++) {
                status = gemm_op();
                CUTLASS_CHECK(status);
            }
            CUDA_CHECK(cudaDeviceSynchronize());

            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            CUDA_CHECK(cudaEventRecord(start));
            for (int i = 0; i < iters; i++) {
                status = gemm_op();
                CUTLASS_CHECK(status);
            }
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));

            float elapsed_ms = 0;
            CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
            double avg_ms = elapsed_ms / iters;

            double db_bytes = double(M) * K * sizeof(DbT);
            double queries_per_sec = N / (avg_ms / 1000.0);
            double eff_tput_gbs = (db_bytes * N) / (avg_ms / 1000.0) / 1.0e9;

            double query_bytes = double(K) * N * sizeof(QueryT);
            double output_bytes = double(M) * N * sizeof(QueryT);
            double hw_bw_gbs = (db_bytes + query_bytes + output_bytes) / (avg_ms / 1000.0) / 1.0e9;

            std::cout << std::setw(8) << N
                      << std::setw(14) << std::fixed << std::setprecision(2) << avg_ms
                      << std::setw(14) << std::fixed << std::setprecision(0) << queries_per_sec
                      << std::setw(18) << std::fixed << std::setprecision(1) << eff_tput_gbs
                      << std::setw(18) << std::fixed << std::setprecision(1) << hw_bw_gbs
                      << std::endl;

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));
        }

        std::cout << std::string(72, '-') << std::endl;
        std::cout << "Eff Tput = (DB_size * batch) / time  [amortized]" << std::endl;
        std::cout << "HW BW   = (DB + queries + output) / time" << std::endl;
    }
};

// ---------- CRT benchmark: two uint16 x uint32 GEMMs ----------

void run_crt(uint64_t M, uint64_t K, const std::vector<int>& batch_sizes,
             int warmup, int iters) {
    using Rows = cutlass::layout::RowMajor;
    using Cols = cutlass::layout::ColumnMajor;
    using Gemm = cutlass::gemm::device::Gemm<uint16_t, Rows, uint32_t, Cols, uint32_t, Rows>;

    const double GiB = 1024.0 * 1024.0 * 1024.0;
    std::cout << "\n=== CRT: 2x (uint16 DB x uint32 query) ===" << std::endl;
    std::cout << "DB: " << M << " x " << K << " uint16 ("
              << std::fixed << std::setprecision(1)
              << (double(M) * K * sizeof(uint16_t)) / GiB << " GiB)" << std::endl;
    std::cout << "Each GEMM: uint16 x uint32 -> uint32, combined time for both" << std::endl;

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU memory: " << std::fixed << std::setprecision(1)
              << total_mem / GiB << " GiB total, "
              << free_mem / GiB << " GiB free" << std::endl;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << std::string(72, '-') << std::endl;

    std::cout << std::setw(8) << "Batch"
              << std::setw(14) << "Time (ms)"
              << std::setw(14) << "QPS"
              << std::setw(18) << "Eff Tput (GB/s)"
              << std::setw(18) << "HW BW (GB/s)"
              << std::endl;
    std::cout << std::string(72, '-') << std::endl;

    // Allocate DB once
    cutlass::HostTensor<uint16_t, Rows> tensor_A({int(M), int(K)});
    cutlass::reference::host::TensorFill(tensor_A.host_view());
    tensor_A.sync_device();

    for (int N : batch_sizes) {
        cutlass::gemm::GemmCoord problem_size(M, N, K);

        // Two sets of queries (CRT residues)
        cutlass::HostTensor<uint32_t, Cols> tensor_B1(problem_size.kn());
        cutlass::HostTensor<uint32_t, Rows> tensor_C1(problem_size.mn());
        cutlass::HostTensor<uint32_t, Rows> tensor_D1(problem_size.mn());
        cutlass::HostTensor<uint32_t, Cols> tensor_B2(problem_size.kn());
        cutlass::HostTensor<uint32_t, Rows> tensor_C2(problem_size.mn());
        cutlass::HostTensor<uint32_t, Rows> tensor_D2(problem_size.mn());

        cutlass::reference::host::TensorFill(tensor_B1.host_view());
        cutlass::reference::host::TensorFill(tensor_C1.host_view());
        cutlass::reference::host::TensorFill(tensor_D1.host_view());
        cutlass::reference::host::TensorFill(tensor_B2.host_view());
        cutlass::reference::host::TensorFill(tensor_C2.host_view());
        cutlass::reference::host::TensorFill(tensor_D2.host_view());

        tensor_B1.sync_device(); tensor_C1.sync_device(); tensor_D1.sync_device();
        tensor_B2.sync_device(); tensor_C2.sync_device(); tensor_D2.sync_device();

        uint32_t alpha = 1, beta = 0;

        typename Gemm::Arguments args1{
            problem_size, tensor_A.device_ref(), tensor_B1.device_ref(),
            tensor_C1.device_ref(), tensor_D1.device_ref(), {alpha, beta}, 1
        };
        typename Gemm::Arguments args2{
            problem_size, tensor_A.device_ref(), tensor_B2.device_ref(),
            tensor_C2.device_ref(), tensor_D2.device_ref(), {alpha, beta}, 1
        };

        Gemm gemm_op1, gemm_op2;
        cutlass::Status s1 = gemm_op1.can_implement(args1);
        cutlass::Status s2 = gemm_op2.can_implement(args2);
        if (s1 != cutlass::Status::kSuccess || s2 != cutlass::Status::kSuccess) {
            std::cerr << "Cannot implement CRT GEMM for batch=" << N << std::endl;
            continue;
        }

        size_t ws1 = Gemm::get_workspace_size(args1);
        size_t ws2 = Gemm::get_workspace_size(args2);
        cutlass::device_memory::allocation<uint8_t> workspace1(ws1), workspace2(ws2);
        CUTLASS_CHECK(gemm_op1.initialize(args1, workspace1.get()));
        CUTLASS_CHECK(gemm_op2.initialize(args2, workspace2.get()));

        for (int i = 0; i < warmup; i++) {
            CUTLASS_CHECK(gemm_op1());
            CUTLASS_CHECK(gemm_op2());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            CUTLASS_CHECK(gemm_op1());
            CUTLASS_CHECK(gemm_op2());
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double avg_ms = elapsed_ms / iters;

        // Effective throughput uses DB size (uint16)
        double db_bytes = double(M) * K * sizeof(uint16_t);
        double queries_per_sec = N / (avg_ms / 1000.0);
        double eff_tput_gbs = (db_bytes * N) / (avg_ms / 1000.0) / 1.0e9;

        // HW BW: read DB twice + read 2x queries + write 2x outputs
        double query_bytes = 2.0 * double(K) * N * sizeof(uint32_t);
        double output_bytes = 2.0 * double(M) * N * sizeof(uint32_t);
        double hw_bw_gbs = (2.0 * db_bytes + query_bytes + output_bytes) / (avg_ms / 1000.0) / 1.0e9;

        std::cout << std::setw(8) << N
                  << std::setw(14) << std::fixed << std::setprecision(2) << avg_ms
                  << std::setw(14) << std::fixed << std::setprecision(0) << queries_per_sec
                  << std::setw(18) << std::fixed << std::setprecision(1) << eff_tput_gbs
                  << std::setw(18) << std::fixed << std::setprecision(1) << hw_bw_gbs
                  << std::endl;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    std::cout << std::string(72, '-') << std::endl;
    std::cout << "Eff Tput = (DB_size * batch) / time  [amortized]" << std::endl;
    std::cout << "HW BW   = (2*DB + 2*queries + 2*output) / time  [both GEMMs]" << std::endl;
}

// ---------- Mixed-precision: uint8 DB x uint32 query -> uint32 accum ----------
// Matches DoublePIR Step 1: DB elements are uint8 (unpacked), queries are uint32

void run_db8_q32(uint64_t M, uint64_t K, const std::vector<int>& batch_sizes,
                 int warmup, int iters) {
    using Rows = cutlass::layout::RowMajor;
    using Cols = cutlass::layout::ColumnMajor;
    using Gemm = cutlass::gemm::device::Gemm<
        uint8_t,                                // ElementA (DB)
        Rows,                                   // LayoutA
        uint32_t,                               // ElementB (Query)
        Cols,                                   // LayoutB
        uint32_t,                               // ElementC (Output)
        Cols,                                   // LayoutC
        uint32_t,                               // ElementAccumulator
        cutlass::arch::OpClassSimt,             // OperatorClass
        cutlass::arch::Sm50,                    // ArchTag
        cutlass::gemm::GemmShape<64, 64, 8>,    // ThreadblockShape
        cutlass::gemm::GemmShape<32, 32, 8>,    // WarpShape
        cutlass::gemm::GemmShape<1, 1, 1>,      // InstructionShape
        cutlass::epilogue::thread::LinearCombination<uint32_t, 1, uint32_t, uint32_t>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
        2                                        // Stages
    >;

    const double GiB = 1024.0 * 1024.0 * 1024.0;
    std::cout << "\n=== uint8 DB x uint32 query -> uint32 (DoublePIR Step 1) ===" << std::endl;
    std::cout << "DB: " << M << " x " << K << " uint8 ("
              << std::fixed << std::setprecision(1)
              << (double(M) * K * sizeof(uint8_t)) / GiB << " GiB)" << std::endl;

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU memory: " << std::fixed << std::setprecision(1)
              << total_mem / GiB << " GiB total, "
              << free_mem / GiB << " GiB free" << std::endl;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "HBM BW (theoretical): ~"
              << (2.0 * props.memoryClockRate * (props.memoryBusWidth / 8.0) / 1.0e6)
              << " GB/s" << std::endl;
    std::cout << std::string(72, '-') << std::endl;

    std::cout << std::setw(8) << "Batch"
              << std::setw(14) << "Time (ms)"
              << std::setw(14) << "QPS"
              << std::setw(18) << "Eff Tput (GB/s)"
              << std::setw(18) << "HW BW (GB/s)"
              << std::endl;
    std::cout << std::string(72, '-') << std::endl;

    cutlass::HostTensor<uint8_t, Rows> tensor_A({int(M), int(K)});
    cutlass::reference::host::TensorFill(tensor_A.host_view());
    tensor_A.sync_device();

    for (int N : batch_sizes) {
        cutlass::gemm::GemmCoord problem_size(M, N, K);

        cutlass::HostTensor<uint32_t, Cols> tensor_B(problem_size.kn());
        cutlass::HostTensor<uint32_t, Cols> tensor_C(problem_size.mn());
        cutlass::HostTensor<uint32_t, Cols> tensor_D(problem_size.mn());

        cutlass::reference::host::TensorFill(tensor_B.host_view());
        cutlass::reference::host::TensorFill(tensor_C.host_view());
        cutlass::reference::host::TensorFill(tensor_D.host_view());

        tensor_B.sync_device(); tensor_C.sync_device(); tensor_D.sync_device();

        uint32_t alpha = 1, beta = 0;

        typename Gemm::Arguments arguments{
            problem_size,
            tensor_A.device_ref(),
            tensor_B.device_ref(),
            tensor_C.device_ref(),
            tensor_D.device_ref(),
            {alpha, beta},
            1
        };

        Gemm gemm_op;
        cutlass::Status status = gemm_op.can_implement(arguments);
        if (status != cutlass::Status::kSuccess) {
            std::cerr << "Cannot implement db8_q32 GEMM for batch=" << N
                      << ": " << cutlassGetStatusString(status) << std::endl;
            continue;
        }

        size_t workspace_size = Gemm::get_workspace_size(arguments);
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
        status = gemm_op.initialize(arguments, workspace.get());
        CUTLASS_CHECK(status);

        for (int i = 0; i < warmup; i++) {
            status = gemm_op();
            CUTLASS_CHECK(status);
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            status = gemm_op();
            CUTLASS_CHECK(status);
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double avg_ms = elapsed_ms / iters;

        double db_bytes = double(M) * K * sizeof(uint8_t);
        double queries_per_sec = N / (avg_ms / 1000.0);
        double eff_tput_gbs = (db_bytes * N) / (avg_ms / 1000.0) / 1.0e9;

        double query_bytes = double(K) * N * sizeof(uint32_t);
        double output_bytes = double(M) * N * sizeof(uint32_t);
        double hw_bw_gbs = (db_bytes + query_bytes + output_bytes) / (avg_ms / 1000.0) / 1.0e9;

        std::cout << std::setw(8) << N
                  << std::setw(14) << std::fixed << std::setprecision(2) << avg_ms
                  << std::setw(14) << std::fixed << std::setprecision(0) << queries_per_sec
                  << std::setw(18) << std::fixed << std::setprecision(1) << eff_tput_gbs
                  << std::setw(18) << std::fixed << std::setprecision(1) << hw_bw_gbs
                  << std::endl;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    std::cout << std::string(72, '-') << std::endl;
    std::cout << "Eff Tput = (DB_uint8_size * batch) / time  [amortized]" << std::endl;
    std::cout << "HW BW   = (DB + queries + output) / time" << std::endl;
}

// ---------- CRT benchmark: 2x (int32 DB x int32 query -> int64 accum) ----------
// This matches the actual CUTLASS kernel used in online_kernel_sp_cutlass.cu

void run_crt_i64(uint64_t M, uint64_t K, const std::vector<int>& batch_sizes,
                 int warmup, int iters) {
    using Rows = cutlass::layout::RowMajor;
    using Cols = cutlass::layout::ColumnMajor;
    using Gemm = cutlass::gemm::device::Gemm<
        int32_t, Rows,
        int32_t, Cols,
        int64_t, Cols,
        int64_t,
        cutlass::arch::OpClassSimt,
        cutlass::arch::Sm50,
        cutlass::gemm::GemmShape<64, 64, 8>,
        cutlass::gemm::GemmShape<32, 32, 8>,
        cutlass::gemm::GemmShape<1, 1, 1>,
        cutlass::epilogue::thread::LinearCombination<int64_t, 1, int64_t, int64_t>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>,
        2
    >;

    const double GiB = 1024.0 * 1024.0 * 1024.0;
    std::cout << "\n=== CRT-i64: 2x (int32 DB x int32 query -> int64 accum) ===" << std::endl;
    std::cout << "DB: " << M << " x " << K << " int32 (widened from uint16, "
              << std::fixed << std::setprecision(1)
              << (double(M) * K * sizeof(int32_t)) / GiB << " GiB)" << std::endl;
    std::cout << "Each GEMM: int32 x int32 -> int64, combined time for both" << std::endl;

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU memory: " << std::fixed << std::setprecision(1)
              << total_mem / GiB << " GiB total, "
              << free_mem / GiB << " GiB free" << std::endl;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "HBM BW (theoretical): ~"
              << (2.0 * props.memoryClockRate * (props.memoryBusWidth / 8.0) / 1.0e6)
              << " GB/s" << std::endl;
    std::cout << std::string(72, '-') << std::endl;

    std::cout << std::setw(8) << "Batch"
              << std::setw(14) << "Time (ms)"
              << std::setw(14) << "QPS"
              << std::setw(18) << "Eff Tput (GB/s)"
              << std::setw(18) << "HW BW (GB/s)"
              << std::endl;
    std::cout << std::string(72, '-') << std::endl;

    // Allocate widened DB (int32, simulating uint16 -> int32 widen)
    cutlass::HostTensor<int32_t, Rows> tensor_A({int(M), int(K)});
    cutlass::reference::host::TensorFill(tensor_A.host_view());
    tensor_A.sync_device();

    for (int N : batch_sizes) {
        cutlass::gemm::GemmCoord problem_size(M, N, K);

        cutlass::HostTensor<int32_t, Cols> tensor_B1(problem_size.kn());
        cutlass::HostTensor<int64_t, Cols> tensor_C1(problem_size.mn());
        cutlass::HostTensor<int64_t, Cols> tensor_D1(problem_size.mn());
        cutlass::HostTensor<int32_t, Cols> tensor_B2(problem_size.kn());
        cutlass::HostTensor<int64_t, Cols> tensor_C2(problem_size.mn());
        cutlass::HostTensor<int64_t, Cols> tensor_D2(problem_size.mn());

        cutlass::reference::host::TensorFill(tensor_B1.host_view());
        cutlass::reference::host::TensorFill(tensor_C1.host_view());
        cutlass::reference::host::TensorFill(tensor_D1.host_view());
        cutlass::reference::host::TensorFill(tensor_B2.host_view());
        cutlass::reference::host::TensorFill(tensor_C2.host_view());
        cutlass::reference::host::TensorFill(tensor_D2.host_view());

        tensor_B1.sync_device(); tensor_C1.sync_device(); tensor_D1.sync_device();
        tensor_B2.sync_device(); tensor_C2.sync_device(); tensor_D2.sync_device();

        int64_t alpha = 1, beta = 0;

        typename Gemm::Arguments args1{
            problem_size, tensor_A.device_ref(), tensor_B1.device_ref(),
            tensor_C1.device_ref(), tensor_D1.device_ref(), {alpha, beta}, 1
        };
        typename Gemm::Arguments args2{
            problem_size, tensor_A.device_ref(), tensor_B2.device_ref(),
            tensor_C2.device_ref(), tensor_D2.device_ref(), {alpha, beta}, 1
        };

        Gemm gemm_op1, gemm_op2;
        cutlass::Status s1 = gemm_op1.can_implement(args1);
        cutlass::Status s2 = gemm_op2.can_implement(args2);
        if (s1 != cutlass::Status::kSuccess || s2 != cutlass::Status::kSuccess) {
            std::cerr << "Cannot implement CRT-i64 GEMM for batch=" << N
                      << ": " << cutlassGetStatusString(s1) << " / "
                      << cutlassGetStatusString(s2) << std::endl;
            continue;
        }

        size_t ws1 = Gemm::get_workspace_size(args1);
        size_t ws2 = Gemm::get_workspace_size(args2);
        cutlass::device_memory::allocation<uint8_t> workspace1(ws1), workspace2(ws2);
        CUTLASS_CHECK(gemm_op1.initialize(args1, workspace1.get()));
        CUTLASS_CHECK(gemm_op2.initialize(args2, workspace2.get()));

        for (int i = 0; i < warmup; i++) {
            CUTLASS_CHECK(gemm_op1());
            CUTLASS_CHECK(gemm_op2());
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) {
            CUTLASS_CHECK(gemm_op1());
            CUTLASS_CHECK(gemm_op2());
        }
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double avg_ms = elapsed_ms / iters;

        // Effective throughput uses original DB size (uint16, since that's the real data)
        double db_bytes_real = double(M) * K * sizeof(uint16_t);
        double queries_per_sec = N / (avg_ms / 1000.0);
        double eff_tput_gbs = (db_bytes_real * N) / (avg_ms / 1000.0) / 1.0e9;

        // HW BW: read int32 DB twice + read 2x int32 queries + write 2x int64 outputs
        double db_bytes_gpu = double(M) * K * sizeof(int32_t);
        double query_bytes = 2.0 * double(K) * N * sizeof(int32_t);
        double output_bytes = 2.0 * double(M) * N * sizeof(int64_t);
        double hw_bw_gbs = (2.0 * db_bytes_gpu + query_bytes + output_bytes) / (avg_ms / 1000.0) / 1.0e9;

        std::cout << std::setw(8) << N
                  << std::setw(14) << std::fixed << std::setprecision(2) << avg_ms
                  << std::setw(14) << std::fixed << std::setprecision(0) << queries_per_sec
                  << std::setw(18) << std::fixed << std::setprecision(1) << eff_tput_gbs
                  << std::setw(18) << std::fixed << std::setprecision(1) << hw_bw_gbs
                  << std::endl;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    }

    std::cout << std::string(72, '-') << std::endl;
    std::cout << "Eff Tput = (DB_uint16_size * batch) / time  [amortized]" << std::endl;
    std::cout << "HW BW   = (2*DB_i32 + 2*queries_i32 + 2*output_i64) / time" << std::endl;
}

// ---------- Tensor Core: 4x int8 GEMM via cuBLAS (DoublePIR Step 1) ----------
//
// Uses cublasGemmEx with CUDA_R_8I for reliable tensor core int8 dispatch.
// Decomposes uint32 query into 4 byte slices, runs 4 int8×int8→int32 GEMMs
// with alpha/beta folding:
//   GEMM 0: D =        1 * DB·q0 + 0·D
//   GEMM 1: D =      256 * DB·q1 + 1·D
//   GEMM 2: D =    65536 * DB·q2 + 1·D
//   GEMM 3: D = 16777216 * DB·q3 + 1·D

void run_db8_q32_tensor(uint64_t M, uint64_t K, const std::vector<int>& batch_sizes,
                        int warmup, int iters) {
    const double GiB = 1024.0 * 1024.0 * 1024.0;
    std::cout << "\n=== cuBLAS Tensor Core: 4x int8 GEMM (DoublePIR Step 1) ===" << std::endl;
    std::cout << "DB: " << M << " x " << K << " int8 ("
              << std::fixed << std::setprecision(1)
              << (double(M) * K) / GiB << " GiB)" << std::endl;
    std::cout << "4 GEMMs with alpha/beta folding: int8 x int8 -> int32 (tensor cores via cuBLAS)" << std::endl;

    size_t free_mem, total_mem;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total_mem));
    std::cout << "GPU memory: " << std::fixed << std::setprecision(1)
              << total_mem / GiB << " GiB total, "
              << free_mem / GiB << " GiB free" << std::endl;

    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    std::cout << "GPU: " << props.name << std::endl;
    std::cout << "HBM BW (theoretical): ~"
              << (2.0 * props.memoryClockRate * (props.memoryBusWidth / 8.0) / 1.0e6)
              << " GB/s" << std::endl;

    if (props.major < 7 || (props.major == 7 && props.minor < 5)) {
        std::cout << "SKIPPED: Tensor cores require SM75+ (Turing or later)" << std::endl;
        return;
    }

    std::cout << std::string(72, '-') << std::endl;
    std::cout << std::setw(8) << "Batch"
              << std::setw(14) << "Time (ms)"
              << std::setw(14) << "QPS"
              << std::setw(18) << "Eff Tput (GB/s)"
              << std::setw(18) << "HW BW (GB/s)"
              << std::endl;
    std::cout << std::string(72, '-') << std::endl;

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));

    // Allocate DB (int8, row-major M×K = transposed col-major K×M for cuBLAS)
    int8_t* d_db;
    CUDA_CHECK(cudaMalloc(&d_db, M * K));
    {
        std::vector<int8_t> h_db(M * K);
        srand(42);
        for (size_t i = 0; i < h_db.size(); i++)
            h_db[i] = (int8_t)(rand() & 0xFF);
        CUDA_CHECK(cudaMemcpy(d_db, h_db.data(), M * K, cudaMemcpyHostToDevice));
    }

    for (int N : batch_sizes) {
        size_t query_elems = K * N;
        size_t output_elems = M * N;

        // Allocate and fill query (int32), then decompose
        int32_t* d_query;
        CUDA_CHECK(cudaMalloc(&d_query, query_elems * sizeof(int32_t)));
        {
            std::vector<int32_t> h_query(query_elems);
            for (size_t i = 0; i < h_query.size(); i++)
                h_query[i] = (int32_t)rand();
            CUDA_CHECK(cudaMemcpy(d_query, h_query.data(),
                                  query_elems * sizeof(int32_t),
                                  cudaMemcpyHostToDevice));
        }

        // Decompose into 4 byte slices (not timed — negligible)
        int8_t* d_q[4];
        for (int i = 0; i < 4; i++)
            CUDA_CHECK(cudaMalloc(&d_q[i], query_elems));
        {
            int threads = 256;
            int blocks = ((int)query_elems + threads - 1) / threads;
            decompose_query_bytes_kernel<<<blocks, threads>>>(
                d_q[0], d_q[1], d_q[2], d_q[3], d_query, query_elems);
            CUDA_CHECK(cudaDeviceSynchronize());
        }

        // Output buffer (int32, col-major M×N)
        int32_t* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, output_elems * sizeof(int32_t)));

        int32_t alphas[4] = {1, 256, 65536, 16777216};
        int32_t betas[4]  = {0, 1, 1, 1};

        // Warmup
        for (int w = 0; w < warmup; w++) {
            for (int g = 0; g < 4; g++) {
                CUBLAS_CHECK(cublasGemmEx(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)M, N, (int)K,
                    &alphas[g],
                    d_db,     CUDA_R_8I, (int)K,   // A: row-major → OP_T, lda=K
                    d_q[g],   CUDA_R_8I, (int)K,   // B: col-major, ldb=K
                    &betas[g],
                    d_output, CUDA_R_32I, (int)M,   // C: col-major, ldc=M
                    CUBLAS_COMPUTE_32I,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
        }
        CUDA_CHECK(cudaDeviceSynchronize());

        // Timed
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++)
            for (int g = 0; g < 4; g++)
                CUBLAS_CHECK(cublasGemmEx(handle,
                    CUBLAS_OP_T, CUBLAS_OP_N,
                    (int)M, N, (int)K,
                    &alphas[g],
                    d_db,     CUDA_R_8I, (int)K,
                    d_q[g],   CUDA_R_8I, (int)K,
                    &betas[g],
                    d_output, CUDA_R_32I, (int)M,
                    CUBLAS_COMPUTE_32I,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float elapsed_ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        double avg_ms = elapsed_ms / iters;

        double db_bytes = double(M) * K;  // int8
        double queries_per_sec = N / (avg_ms / 1000.0);
        double eff_tput_gbs = (db_bytes * N) / (avg_ms / 1000.0) / 1.0e9;

        // HW BW: DB read 4x, 4 query byte arrays, output: 1 write + 3 RMW
        double hw_bw_gbs = (
            4.0 * db_bytes +
            4.0 * double(K) * N +
            (1 + 3 * 2) * double(M) * N * sizeof(int32_t)
        ) / (avg_ms / 1000.0) / 1.0e9;

        std::cout << std::setw(8) << N
                  << std::setw(14) << std::fixed << std::setprecision(2) << avg_ms
                  << std::setw(14) << std::fixed << std::setprecision(0) << queries_per_sec
                  << std::setw(18) << std::fixed << std::setprecision(1) << eff_tput_gbs
                  << std::setw(18) << std::fixed << std::setprecision(1) << hw_bw_gbs
                  << std::endl;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        CUDA_CHECK(cudaFree(d_query));
        for (int i = 0; i < 4; i++)
            CUDA_CHECK(cudaFree(d_q[i]));
        CUDA_CHECK(cudaFree(d_output));
    }

    CUDA_CHECK(cudaFree(d_db));
    cublasDestroy(handle);
    std::cout << std::string(72, '-') << std::endl;
    std::cout << "Eff Tput = (DB_int8_size * batch) / time  [amortized]" << std::endl;
    std::cout << "HW BW   = (4*DB + 4*q_bytes + output_writes) / time  [no L2 cache assumed]" << std::endl;
}

// ---------- Argument parsing ----------

std::vector<int> parse_batch_list(const std::string& s) {
    std::vector<int> result;
    std::istringstream iss(s);
    std::string token;
    while (std::getline(iss, token, ',')) {
        result.push_back(std::stoi(token));
    }
    return result;
}

void print_usage(const char* prog) {
    std::cerr << "Usage: " << prog << " [options]" << std::endl;
    std::cerr << "  --mode    MODE           One of: db32_q32, db32_q64, db16_q32, db16_q64, db16_crt, db8_q32, db32_crt_i64, db8_q32_tc, all" << std::endl;
    std::cerr << "                           (default: all)" << std::endl;
    std::cerr << "  --batches 1,2,4,...      Comma-separated batch sizes" << std::endl;
    std::cerr << "  --log_dim N              Log2 of matrix dimension (default: 15)" << std::endl;
    std::cerr << "  --warmup  N              Warmup iterations (default: 3)" << std::endl;
    std::cerr << "  --iters   N              Timed iterations (default: 10)" << std::endl;
}

int main(int argc, char** argv) {
    std::string mode = "all";
    int log_dim = 15;
    int warmup = 3;
    int iters = 10;
    std::vector<int> batches = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if (arg == "--mode" && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--batches" && i + 1 < argc) {
            batches = parse_batch_list(argv[++i]);
        } else if (arg == "--log_dim" && i + 1 < argc) {
            log_dim = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    uint64_t dim = 1ULL << log_dim;
    bool any = false;

    if (mode == "db32_q32" || mode == "all") {
        PirBench<uint32_t, uint32_t>::run(dim, dim, batches, warmup, iters,
            "uint32 DB x uint32 query");
        any = true;
    }
    if (mode == "db32_q64" || mode == "all") {
        PirBench<uint32_t, uint64_t>::run(dim, dim, batches, warmup, iters,
            "uint32 DB x uint64 query");
        any = true;
    }
    if (mode == "db16_q32" || mode == "all") {
        PirBench<uint16_t, uint32_t>::run(dim, dim, batches, warmup, iters,
            "uint16 DB x uint32 query");
        any = true;
    }
    if (mode == "db16_q64" || mode == "all") {
        PirBench<uint16_t, uint64_t>::run(dim, dim, batches, warmup, iters,
            "uint16 DB x uint64 query");
        any = true;
    }
    if (mode == "db16_crt" || mode == "all") {
        run_crt(dim, dim, batches, warmup, iters);
        any = true;
    }
    if (mode == "db8_q32" || mode == "all") {
        run_db8_q32(dim, dim, batches, warmup, iters);
        any = true;
    }
    if (mode == "db32_crt_i64" || mode == "all") {
        run_crt_i64(dim, dim, batches, warmup, iters);
        any = true;
    }
    if (mode == "db8_q32_tc" || mode == "all") {
        run_db8_q32_tensor(dim, dim, batches, warmup, iters);
        any = true;
    }

    if (!any) {
        std::cerr << "Unknown --mode: " << mode << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    return 0;
}
