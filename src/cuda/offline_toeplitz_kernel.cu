#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cublas_v2.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#define CUBLAS_CHECK(call) do { \
    cublasStatus_t status_ = (call); \
    if (status_ != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__, (int)status_); \
    } \
} while(0)

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
// Reads A in n×l1 col-major, writes 4 int8 byte slices in l1×n col-major (= A^T layout)
__global__ void transpose_and_decompose(
    int8_t* __restrict__ b0, int8_t* __restrict__ b1,
    int8_t* __restrict__ b2, int8_t* __restrict__ b3,
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

    b0[idx] = (int8_t)(v & 0xFF);
    b1[idx] = (int8_t)((v >> 8) & 0xFF);
    b2[idx] = (int8_t)((v >> 16) & 0xFF);
    b3[idx] = (int8_t)((v >> 24) & 0xFF);
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
    // Tensor core path (cuBLAS int8)
    bool has_tensor_cores;
    cublasHandle_t cublas_handle;
    int8_t* d_A_bytes[4];   // l1×n byte slices of A^T, col-major

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

    for (int i = 0; i < 4; i++) {
        if (ctx->d_A_bytes[i]) cudaFree(ctx->d_A_bytes[i]);
    }
    if (ctx->d_A_T)            cudaFree(ctx->d_A_T);
    if (ctx->d_gemm_workspace) cudaFree(ctx->d_gemm_workspace);
    if (ctx->d_D)              cudaFree(ctx->d_D);
    if (ctx->d_result)         cudaFree(ctx->d_result);
    if (ctx->cublas_handle)    cublasDestroy(ctx->cublas_handle);

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
    ctx->cublas_handle = nullptr;
    ctx->d_A_T = nullptr;
    ctx->d_gemm_workspace = nullptr;
    ctx->d_D = nullptr;
    ctx->d_result = nullptr;
    for (int i = 0; i < 4; i++) ctx->d_A_bytes[i] = nullptr;

    size_t A_elems = (size_t)n * db_rows;  // = l1 * n

    // ---- Detect tensor cores (SM >= 72) ----
    {
        int device;
        CUDA_CHECK(cudaGetDevice(&device));
        int major, minor;
        CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
        CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
        int sm = major * 10 + minor;
        ctx->has_tensor_cores = (sm >= 72);

        if (ctx->has_tensor_cores) {
            printf("Toeplitz GEMM (tensor core): n=%u, l1=%u, l2=%u  =>  M=%u, K=%u, N=%u\n",
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
        // Fused transpose + decompose into 4 int8 byte slices (l1×n col-major)
        for (int i = 0; i < 4; i++) {
            if (cudaMalloc(&ctx->d_A_bytes[i], A_elems) != cudaSuccess) {
                printf("ERROR: Failed to allocate byte slice %d.\n", i);
                cudaFree(d_A_tmp);
                free_toeplitz_context(ctx);
                return nullptr;
            }
        }

        int threads = 256;
        int blocks = (A_elems + threads - 1) / threads;
        transpose_and_decompose<<<blocks, threads>>>(
            ctx->d_A_bytes[0], ctx->d_A_bytes[1],
            ctx->d_A_bytes[2], ctx->d_A_bytes[3],
            d_A_tmp, n, db_rows);
        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(cublasCreate(&ctx->cublas_handle));
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
        // ---- Tensor core path: 4× cuBLAS int8 GEMM with alpha/beta folding ----
        // D^T(l2×l1) × A^T_byte_g(l1×n) → C(l2×n)
        // M=l2, N=n, K=l1

        int32_t alphas[4] = {1, 256, 65536, 16777216};
        int32_t betas[4]  = {0, 1, 1, 1};
        int M = (int)ctx->l2;
        int N = (int)ctx->n;
        int K = (int)ctx->l1;

        for (int g = 0; g < 4; g++) {
            CUBLAS_CHECK(cublasGemmEx(ctx->cublas_handle,
                CUBLAS_OP_T, CUBLAS_OP_N,
                M, N, K,
                &alphas[g],
                ctx->d_D,          CUDA_R_8I, K,   // D: l1×l2 col-major, OP_T → l2×l1
                ctx->d_A_bytes[g], CUDA_R_8I, K,   // A^T: l1×n col-major, OP_N → l1×n
                &betas[g],
                (int32_t*)ctx->d_result, CUDA_R_32I, M,
                CUBLAS_COMPUTE_32I,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
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
