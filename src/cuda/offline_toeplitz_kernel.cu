#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

extern "C" {

// Kernels

// Toeplitz builder for negacyclic polynomial multiplication
__global__ void build_toeplitz_matrix_negacyclic_u32(
    const uint32_t* __restrict__ poly,
    uint32_t* __restrict__ matrix,
    uint32_t n
) {
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

// New GEMM kernel reading DB as uint8_t

#define TILE 32

__global__ void gemm_mod32_tiled_u8(
    const uint32_t* __restrict__ A,   // (n × l1) col-major
    const uint8_t*  __restrict__ D_u8,// (l1 × l2) col-major, bytes
    uint32_t*       __restrict__ C,   // (n × l2) col-major output
    uint32_t n,
    uint32_t l1,
    uint32_t l2
) {
    uint32_t row = blockIdx.y * TILE + threadIdx.y;
    uint32_t col = blockIdx.x * TILE + threadIdx.x;

    if (row >= n || col >= l2)
        return;

    __shared__ uint32_t As[TILE][TILE];
    __shared__ uint32_t Ds[TILE][TILE];

    uint64_t acc = 0;

    for (uint32_t t = 0; t < l1; t += TILE) {

        uint32_t kA = t + threadIdx.x;
        uint32_t kD = t + threadIdx.y;

        // Tile of A
        As[threadIdx.y][threadIdx.x] =
            (row < n && kA < l1)
                ? A[row + (size_t)kA * n]
                : 0;

        // Tile of D, widening uint8 -> uint32
        uint8_t d_val =
            (kD < l1 && col < l2)
                ? D_u8[kD + (size_t)col * l1]
                : 0;

        Ds[threadIdx.y][threadIdx.x] = (uint32_t)d_val;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; k++) {
            acc += (uint64_t)As[threadIdx.y][k] *
                   (uint64_t)Ds[k][threadIdx.x];
        }

        acc = (uint32_t)acc;  // periodic mod 2^32 - unnecessary unless DB dim > 2^22
        __syncthreads();
    }

    C[row * l2 + col] = (uint32_t)acc;
}

#undef TILE

// Convert result to uint64
__global__ void result_to_uint64(
    const uint32_t* result_u32,
    uint64_t* result_u64,
    uint32_t total
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        result_u64[idx] = (uint64_t)result_u32[idx];
    }
}

// Context

struct ToeplitzContext {

    uint32_t* d_A;     // n × l1 Toeplitz, column major
    uint8_t*  d_D;     // l1 × l2 DB **stored as bytes now**
    uint32_t* d_result;

    uint32_t n;
    uint32_t l1;
    uint32_t l2;
};

void free_toeplitz_context(void* context);

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
) {
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
    ctx->d_A = nullptr;
    ctx->d_D = nullptr;
    ctx->d_result = nullptr;

    printf("Batched GEMM: n=%u, l1=%u, l2=%u\n", n, db_rows, db_cols);

    size_t size_A      = (size_t)n * db_rows * sizeof(uint32_t);
    size_t size_result = (size_t)n * db_cols * sizeof(uint32_t);
    size_t size_D_u8   = (size_t)db_rows * db_cols * sizeof(uint8_t);

    // Allocate A, D_u8, result
    if (cudaMalloc(&ctx->d_A, size_A) != cudaSuccess ||
        cudaMalloc(&ctx->d_D, size_D_u8) != cudaSuccess ||
        cudaMalloc(&ctx->d_result, size_result) != cudaSuccess) {
        printf("ERROR: Failed to allocate GPU memory.\n");
        free_toeplitz_context(ctx);
        return nullptr;
    }

    // ---- Build Toeplitz ----
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
            ctx->d_A + (size_t)i * n * n,
            n
        );
    }

    cudaFree(d_poly);
    cudaDeviceSynchronize();

    // ---- Upload DB as bytes ----
    cudaMemcpy(ctx->d_D, db, size_D_u8, cudaMemcpyHostToDevice);

    return ctx;
}

// compute_hint_0_toeplitz

int compute_hint_0_toeplitz(void* context, uint64_t* hint_0) {
    ToeplitzContext* ctx = (ToeplitzContext*)context;
    if (!ctx) return -1;

    printf("Computing with tiled GEMM (uint8 DB)...\n");

    dim3 block(32, 32);
    dim3 grid((ctx->l2 + 31) / 32, (ctx->n + 31) / 32);

    gemm_mod32_tiled_u8<<<grid, block>>>(
        ctx->d_A,
        ctx->d_D,     // now uint8_t*
        ctx->d_result,
        ctx->n,
        ctx->l1,
        ctx->l2
    );

    if (cudaDeviceSynchronize() != cudaSuccess) {
        printf("ERROR: gemm_mod32_tiled_u8 failed.\n");
        return -1;
    }

    // Convert to uint64
    uint32_t total = ctx->n * ctx->l2;

    uint64_t* d_hint_0;
    cudaMalloc(&d_hint_0, total * sizeof(uint64_t));

    dim3 block2(256);
    dim3 grid2((total + 255) / 256);

    result_to_uint64<<<grid2, block2>>>(
        ctx->d_result, d_hint_0, total
    );

    cudaMemcpy(hint_0, d_hint_0,
               total * sizeof(uint64_t),
               cudaMemcpyDeviceToHost);

    cudaFree(d_hint_0);
    return 0;
}

// free_toeplitz_context
void free_toeplitz_context(void* context) {
    ToeplitzContext* ctx = (ToeplitzContext*)context;
    if (!ctx) return;

    if (ctx->d_A)      cudaFree(ctx->d_A);
    if (ctx->d_D)      cudaFree(ctx->d_D);   // now uint8_t*
    if (ctx->d_result) cudaFree(ctx->d_result);

    delete ctx;
}

} // extern "C"
