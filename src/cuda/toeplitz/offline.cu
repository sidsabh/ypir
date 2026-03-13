#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#define CUDA_CHECK(call) do { \
    cudaError_t err_ = (call); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
    } \
} while(0)

// CUTLASS SIMT fallback: uint8 D^T × uint32 A^T → uint32 result (mod 2^32)
// Same type as online_kernel.cu CutlassGemm
using CutlassGemmSimt = cutlass::gemm::device::Gemm<
    uint8_t,                                // ElementA (D^T, row-major)
    cutlass::layout::RowMajor,              // LayoutA
    uint32_t,                               // ElementB (A^T, col-major)
    cutlass::layout::ColumnMajor,           // LayoutB
    uint32_t,                               // ElementC (Output, col-major)
    cutlass::layout::ColumnMajor,           // LayoutC
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

// CUTLASS uint8 TC GEMM: uint8 × uint8 → int32 (tensor cores)

// SM80+ (Ampere): wider instruction shape
using CutlassGemmU8TC_Sm80 = cutlass::gemm::device::Gemm<
    uint8_t, cutlass::layout::RowMajor,
    uint8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 3
>;

// SM75 (Turing): u8 IMMA with 8×8×16 instruction shape
using CutlassGemmU8TC_Sm75 = cutlass::gemm::device::Gemm<
    uint8_t, cutlass::layout::RowMajor,
    uint8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, 256, 64>,
    cutlass::gemm::GemmShape<64, 64, 64>,
    cutlass::gemm::GemmShape<8, 8, 16>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 4, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 2
>;

// Helper: run the correct TC GEMM based on SM version
static cutlass::Status run_u8tc_gemm(bool is_sm80,
    int M, int wide_N, int K,
    const uint8_t* A, int lda,
    const uint8_t* B, int ldb,
    int32_t* C, int ldc)
{
    int32_t alpha = 1, beta = 0;
    cutlass::gemm::GemmCoord problem_size(M, wide_N, K);
    if (is_sm80) {
        CutlassGemmU8TC_Sm80::Arguments args{
            problem_size, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta}, 1
        };
        CutlassGemmU8TC_Sm80 gemm_op;
        auto s = gemm_op.initialize(args, nullptr);
        if (s != cutlass::Status::kSuccess) return s;
        return gemm_op();
    } else {
        CutlassGemmU8TC_Sm75::Arguments args{
            problem_size, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta}, 1
        };
        CutlassGemmU8TC_Sm75 gemm_op;
        auto s = gemm_op.initialize(args, nullptr);
        if (s != cutlass::Status::kSuccess) return s;
        return gemm_op();
    }
}

extern "C" {

// Toeplitz builder for negacyclic polynomial multiplication
// Builds n×n block in column-major: matrix[row + col * n]
__global__ void build_toeplitz_matrix_negacyclic_u32(
    const uint32_t* __restrict__ poly,
    uint32_t* __restrict__ matrix,
    uint32_t n
)
{
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= n) return;

    for (uint32_t row = 0; row < n; row++) {
        uint32_t val;
        if (row >= col) {
            val = poly[row - col];
        } else {
            val = ~poly[n + row - col] + 1;
        }
        matrix[row + col * n] = val;
    }
}

// Fused transpose + byte-decompose (tensor core path):
// Reads A in n×l1 col-major, writes 4 packed uint8 byte slices contiguously
// in l1×(4*n) col-major (= A^T layout with 4 byte slices stacked along columns)
__global__ void transpose_and_decompose_packed(
    uint8_t* __restrict__ out,   // l1 × (4*n) col-major
    const uint32_t* __restrict__ A,
    uint32_t n, uint32_t l1)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)l1 * n;
    if (idx >= total) return;

    // idx is linear index in l1×n col-major (A^T layout)
    size_t i = idx % l1;   // row in A^T = col in A
    size_t j = idx / l1;   // col in A^T = row in A

    uint32_t v = A[j + i * n];  // read from n×l1 col-major

    size_t stride = (size_t)l1 * n;  // elements per byte slice
    out[idx + 0 * stride] = (uint8_t)(v & 0xFF);
    out[idx + 1 * stride] = (uint8_t)((v >> 8) & 0xFF);
    out[idx + 2 * stride] = (uint8_t)((v >> 16) & 0xFF);
    out[idx + 3 * stride] = (uint8_t)((v >> 24) & 0xFF);
}

// Plain transpose (SIMT path):
// Reads A in n×l1 col-major, writes A^T in l1×n col-major as uint32
__global__ void transpose_u32(
    const uint32_t* __restrict__ src,
    uint32_t* __restrict__ dst,
    uint32_t n, uint32_t l1)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = (size_t)l1 * n;
    if (idx >= total) return;

    size_t i = idx % l1;   // row in dst (= col in src)
    size_t j = idx / l1;   // col in dst (= row in src)

    dst[idx] = src[j + i * n];
}

// Accumulate 1 wide GEMM output (4 byte slices of A^T) into uint32 result
// gemm_out is col-major M × (4*N), stride M
__global__ void accumulate_toeplitz_bytes_kernel(
    uint32_t* __restrict__ result,         // M × N col-major
    const int32_t* __restrict__ gemm_out,  // M × (4*N) col-major
    size_t M, size_t N)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;
    size_t m = idx % M;
    size_t n = idx / M;

    uint32_t acc = 0;
    #pragma unroll
    for (int b = 0; b < 4; b++) {
        uint32_t val = (uint32_t)gemm_out[m + (b * N + n) * M];
        acc += val << (8 * b);
    }
    result[idx] = acc;
}

// Widen int32/uint32 to uint64 (no transpose — layout already correct)
__global__ void widen_to_u64(
    const uint32_t* __restrict__ src,
    uint64_t* __restrict__ dst,
    uint32_t total)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    dst[idx] = (uint64_t)src[idx];
}

// Context

struct ToeplitzContext
{
    // Tensor core path (CUTLASS uint8 TC, SM75+)
    bool has_tensor_cores;
    bool is_sm80;
    uint8_t* d_A_bytes_packed;  // l1 × (4*n) packed uint8 byte slices of A^T, col-major
    int32_t* d_gemm_out;        // l2 × (4*n) GEMM output, col-major

    // SIMT path (CUTLASS uint8×uint32)
    uint32_t* d_A_T;        // l1×n col-major uint32 (A transposed)
    void* d_gemm_workspace;

    // Common
    uint8_t*  d_D;           // l1×l2 DB as uint8, col-major
    uint32_t* d_result;      // l2×n col-major = n×l2 row-major

    uint32_t n;
    uint32_t l1;
    uint32_t l2;
};

// free_toeplitz_context
void free_toeplitz_context(void* context)
{
    ToeplitzContext* ctx = (ToeplitzContext*)context;
    if (!ctx) return;

    if (ctx->d_A_bytes_packed) cudaFree(ctx->d_A_bytes_packed);
    if (ctx->d_gemm_out)      cudaFree(ctx->d_gemm_out);
    if (ctx->d_A_T)           cudaFree(ctx->d_A_T);
    if (ctx->d_gemm_workspace) cudaFree(ctx->d_gemm_workspace);
    if (ctx->d_D)             cudaFree(ctx->d_D);
    if (ctx->d_result)        cudaFree(ctx->d_result);

    delete ctx;
}

// init_toeplitz_context

void* init_toeplitz_context(
    const uint8_t* db,
    const uint32_t* v_nega_perm_a,
    const uint64_t* moduli,
    const uint64_t* barrett_cr,
    uint32_t db_rows,
    uint32_t db_rows_padded,
    uint32_t db_cols,
    uint32_t n,
    uint32_t crt_count,
    uint32_t max_adds,
    uint64_t mod0_inv_mod1,
    uint64_t mod1_inv_mod0,
    uint64_t barrett_cr_0_modulus,
    uint64_t barrett_cr_1_modulus
)
{
    (void)moduli;
    (void)barrett_cr;
    (void)db_rows_padded;
    (void)crt_count;
    (void)max_adds;
    (void)mod0_inv_mod1;
    (void)mod1_inv_mod0;
    (void)barrett_cr_0_modulus;
    (void)barrett_cr_1_modulus;

    ToeplitzContext* ctx = new ToeplitzContext();
    if (!ctx) return nullptr;

    ctx->n  = n;
    ctx->l1 = db_rows;
    ctx->l2 = db_cols;
    ctx->has_tensor_cores = false;
    ctx->is_sm80 = false;
    ctx->d_A_bytes_packed = nullptr;
    ctx->d_gemm_out = nullptr;
    ctx->d_A_T = nullptr;
    ctx->d_gemm_workspace = nullptr;
    ctx->d_D = nullptr;
    ctx->d_result = nullptr;

    size_t A_elems = (size_t)n * db_rows;  // = l1 * n

    // ---- Detect tensor cores (SM >= 80 for CUTLASS uint8 TC) ----
    {
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        int major, minor;
        CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
        int sm = major * 10 + minor;
        ctx->has_tensor_cores = (sm >= 75);
        ctx->is_sm80 = (sm >= 80);

        if (ctx->has_tensor_cores) {
            printf("Toeplitz GEMM (CUTLASS uint8 TC %s): n=%u, l1=%u, l2=%u  =>  M=%u, K=%u, N=%u\n",
                   ctx->is_sm80 ? "Sm80" : "Sm75",
                   n, db_rows, db_cols, db_cols, db_rows, n);
        } else {
            printf("Toeplitz GEMM (CUTLASS SIMT): n=%u, l1=%u, l2=%u  =>  M=%u, K=%u, N=%u\n",
                   n, db_rows, db_cols, db_cols, db_rows, n);
        }
    }

    // ---- Build Toeplitz (uint32, temporary, n×l1 col-major) ----
    uint32_t* d_A_tmp = nullptr;
    if (cudaMalloc(&d_A_tmp, A_elems * sizeof(uint32_t)) != cudaSuccess) {
        printf("ERROR: Failed to allocate Toeplitz matrix.\n");
        free_toeplitz_context(ctx);
        return nullptr;
    }

    {
        uint32_t* d_poly = nullptr;
        cudaMalloc(&d_poly, db_rows * sizeof(uint32_t));
        cudaMemcpy(d_poly, v_nega_perm_a,
                   db_rows * sizeof(uint32_t),
                   cudaMemcpyHostToDevice);

        uint32_t num_polys = db_rows / n;
        dim3 block(256);
        dim3 grid((n + 255) / 256);
        for (uint32_t i = 0; i < num_polys; i++) {
            build_toeplitz_matrix_negacyclic_u32<<<grid, block>>>(
                d_poly + i * n,
                d_A_tmp + (size_t)i * n * n,
                n
            );
        }
        cudaFree(d_poly);
        cudaDeviceSynchronize();
    }

    // ---- Path-specific: transpose + prepare A ----
    if (ctx->has_tensor_cores) {
        // Fused transpose + decompose into packed uint8: l1 × (4*n) col-major
        size_t packed_size = A_elems * 4;  // 4 byte slices
        if (cudaMalloc(&ctx->d_A_bytes_packed, packed_size) != cudaSuccess) {
            printf("ERROR: Failed to allocate packed byte slices.\n");
            cudaFree(d_A_tmp);
            free_toeplitz_context(ctx);
            return nullptr;
        }

        // Allocate GEMM output: l2 × (4*n) int32
        size_t gemm_out_size = (size_t)db_cols * 4 * n * sizeof(int32_t);
        if (cudaMalloc(&ctx->d_gemm_out, gemm_out_size) != cudaSuccess) {
            printf("ERROR: Failed to allocate GEMM output.\n");
            cudaFree(d_A_tmp);
            free_toeplitz_context(ctx);
            return nullptr;
        }

        int threads = 256;
        int blocks = (A_elems + threads - 1) / threads;
        transpose_and_decompose_packed<<<blocks, threads>>>(
            ctx->d_A_bytes_packed, d_A_tmp, n, db_rows);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // Plain transpose to l1×n col-major uint32
        if (cudaMalloc(&ctx->d_A_T, A_elems * sizeof(uint32_t)) != cudaSuccess) {
            printf("ERROR: Failed to allocate transposed Toeplitz.\n");
            cudaFree(d_A_tmp);
            free_toeplitz_context(ctx);
            return nullptr;
        }

        int threads = 256;
        int blocks = (A_elems + threads - 1) / threads;
        transpose_u32<<<blocks, threads>>>(d_A_tmp, ctx->d_A_T, n, db_rows);
        CUDA_CHECK(cudaGetLastError());
    }

    // Free temporary uint32 Toeplitz
    cudaFree(d_A_tmp);
    cudaDeviceSynchronize();

    // ---- Upload DB (uint8, l1×l2 col-major) ----
    size_t size_D = (size_t)db_rows * db_cols;
    if (cudaMalloc(&ctx->d_D, size_D) != cudaSuccess) {
        printf("ERROR: Failed to allocate DB.\n");
        free_toeplitz_context(ctx);
        return nullptr;
    }
    cudaMemcpy(ctx->d_D, db, size_D, cudaMemcpyHostToDevice);

    // ---- Result buffer: l2×n col-major = n×l2 row-major ----
    size_t size_result = (size_t)n * db_cols * sizeof(uint32_t);
    if (cudaMalloc(&ctx->d_result, size_result) != cudaSuccess) {
        printf("ERROR: Failed to allocate result.\n");
        free_toeplitz_context(ctx);
        return nullptr;
    }

    // ---- Pre-allocate CUTLASS workspace (SIMT only) ----
    if (!ctx->has_tensor_cores) {
        // D^T(l2×l1, uint8 RowMajor) × A^T(l1×n, uint32 ColMajor) → C(l2×n, uint32 ColMajor)
        cutlass::gemm::GemmCoord problem_size(db_cols, n, db_rows);
        uint32_t alpha = 1, beta = 0;
        CutlassGemmSimt::Arguments args{
            problem_size,
            {ctx->d_D, (int)db_rows},        // A = D^T: RowMajor, lda = l1
            {ctx->d_A_T, (int)db_rows},      // B = A^T: ColMajor, ldb = l1
            {ctx->d_result, (int)db_cols},    // C: ColMajor, ldc = l2
            {ctx->d_result, (int)db_cols},    // D: same
            {alpha, beta}, 1
        };
        size_t ws = CutlassGemmSimt::get_workspace_size(args);
        if (ws > 0) {
            CUDA_CHECK(cudaMalloc(&ctx->d_gemm_workspace, ws));
        }
    }

    return ctx;
}

// compute_hint_0_toeplitz

int compute_hint_0_toeplitz(void* context, uint64_t* hint_0)
{
    ToeplitzContext* ctx = (ToeplitzContext*)context;
    if (!ctx) return -1;

    if (ctx->has_tensor_cores) {
        // ---- CUTLASS uint8 TC: 1 wide GEMM ----
        // D^T(l2×l1, uint8 RowMajor) × A^T_packed(l1×(4*n), uint8 ColMajor)
        //   → gemm_out(l2×(4*n), int32 ColMajor)
        // Then accumulate 4 byte products into uint32 result
        int M = (int)ctx->l2;
        int N = (int)ctx->n;
        int K = (int)ctx->l1;
        int wide_N = 4 * N;

        cutlass::Status status = run_u8tc_gemm(ctx->is_sm80,
            M, wide_N, K,
            ctx->d_D, K,
            ctx->d_A_bytes_packed, K,
            ctx->d_gemm_out, M);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS TC GEMM failed: %s\n", cutlassGetStatusString(status));
            return -1;
        }

        // Accumulate 4 byte products → uint32
        size_t total_result = (size_t)M * N;
        int acc_threads = 256;
        int acc_blocks = (total_result + acc_threads - 1) / acc_threads;
        accumulate_toeplitz_bytes_kernel<<<acc_blocks, acc_threads>>>(
            ctx->d_result, ctx->d_gemm_out, M, N);
        CUDA_CHECK(cudaGetLastError());
    } else {
        // ---- CUTLASS SIMT fallback: single uint8×uint32 → uint32 GEMM ----
        // D^T(l2×l1, uint8 RowMajor) × A^T(l1×n, uint32 ColMajor) → C(l2×n, uint32 ColMajor)
        cutlass::gemm::GemmCoord problem_size(ctx->l2, ctx->n, ctx->l1);
        uint32_t alpha = 1, beta = 0;
        CutlassGemmSimt::Arguments args{
            problem_size,
            {ctx->d_D, (int)ctx->l1},          // A = D^T: RowMajor, lda = l1
            {ctx->d_A_T, (int)ctx->l1},        // B = A^T: ColMajor, ldb = l1
            {ctx->d_result, (int)ctx->l2},     // C: ColMajor, ldc = l2
            {ctx->d_result, (int)ctx->l2},     // D: same
            {alpha, beta}, 1
        };
        CutlassGemmSimt gemm_op;
        cutlass::Status status = gemm_op.initialize(args, ctx->d_gemm_workspace);
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS init failed: %s\n", cutlassGetStatusString(status));
            return -1;
        }
        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            fprintf(stderr, "CUTLASS GEMM failed: %s\n", cutlassGetStatusString(status));
            return -1;
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Widen uint32 → uint64 (no transpose needed, layout is already row-major)
    uint32_t total = ctx->n * ctx->l2;

    uint64_t* d_hint_0;
    cudaMalloc(&d_hint_0, total * sizeof(uint64_t));

    {
        dim3 block(256);
        dim3 grid((total + 255) / 256);
        widen_to_u64<<<grid, block>>>(ctx->d_result, d_hint_0, total);
    }

    cudaMemcpy(hint_0, d_hint_0,
               total * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);

    cudaFree(d_hint_0);
    return 0;
}

} // extern "C"
