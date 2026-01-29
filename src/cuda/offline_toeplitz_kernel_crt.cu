#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <cstdio>

extern "C" {
// CRT Constants (2 primes around 2^28 for optimal coverage)
// P1*P2 ≈ 2^56 > 2^55 (max unreduced dot product fits in CRT modulus)
// Max per-element: 2^15 × 2^28 × 2^8 = 2^51 < 2^53 (safe in double)
static const uint64_t P1 = 268435399;  // 2^28 - 57 (prime)
static const uint64_t P2 = 268435459;  // 2^28 + 3 (prime)


// Build Toeplitz matrix directly as double mod p
__global__ void build_toeplitz_matrix_negacyclic_double(
    const uint32_t* __restrict__ poly,
    double* __restrict__ matrix,
    uint32_t n,
    uint64_t p
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
        matrix[row + col * n] = (double)(val % p);
    }
}

// Convert DB uint8 -> double mod p
__global__ void db_u8_to_double(
    const uint8_t* __restrict__ db,
    double* __restrict__ dst,
    uint32_t total,
    uint64_t p
) 
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        dst[idx] = (double)((uint32_t)db[idx] % p);
    }
}

// CRT Reconstruction: (r1, r2) -> result mod 2^32
__global__ void crt_reconstruct_2_double(
    const double* __restrict__ r1_ptr,
    const double* __restrict__ r2_ptr,
    uint64_t* __restrict__ dst,
    uint32_t n,
    uint32_t l2,
    uint64_t p1, uint64_t p2,
    uint64_t inv1, uint64_t inv2
) 
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = n * l2;
    if (idx >= total) return;

    // Input is column-major
    uint32_t row = idx % n;
    uint32_t col = idx / n;

    // Read double results - should be exact integers (any fractional part is error)
    int64_t raw1 = (int64_t)r1_ptr[idx];
    uint64_t a1 = ((raw1 % (int64_t)p1) + p1) % p1;
    uint64_t a2 = (uint64_t)r2_ptr[idx] % p2;

    // CRT: x ≡ a1 (mod p1), x ≡ a2 (mod p2)
    // x = (a1 * p2 * inv(p2, p1) + a2 * p1 * inv(p1, p2)) mod (p1*p2)
    // Use uint128 to avoid overflow
    unsigned __int128 M1 = p2;
    unsigned __int128 M2 = p1;
    unsigned __int128 M_total = (unsigned __int128)p1 * p2;

    unsigned __int128 term1 = (unsigned __int128)a1 * M1 * inv1;
    unsigned __int128 term2 = (unsigned __int128)a2 * M2 * inv2;
    
    unsigned __int128 sum = (term1 + term2) % M_total;

    // Output Row-Major, take mod 2^32
    dst[row * l2 + col] = (uint32_t)sum;
}

// Context
struct ToeplitzContext 
{
    cublasHandle_t handle;

    uint8_t* d_db_u8;
    
    double* d_A_p1;
    double* d_A_p2;
    
    double* d_C_p1;
    double* d_C_p2;

    uint32_t n;
    uint32_t l1;
    uint32_t l2;
    
    uint64_t inv1;
    uint64_t inv2;
};

void free_toeplitz_context(void* context) 
{
    ToeplitzContext* ctx = (ToeplitzContext*)context;
    if (!ctx) return;

    cudaFree(ctx->d_db_u8);
    cudaFree(ctx->d_A_p1);
    cudaFree(ctx->d_A_p2);
    cudaFree(ctx->d_C_p1);
    cudaFree(ctx->d_C_p2);

    cublasDestroy(ctx->handle);
    delete ctx;
}

// Modular inverse using extended Euclidean algorithm
uint64_t mod_inverse_u64(uint64_t a, uint64_t m) 
{
    int64_t m0 = m;
    int64_t t, q;
    int64_t x0 = 0, x1 = 1;
    if (m == 1) return 0;
    while (a > 1) {
        q = a / m;
        t = m;
        m = a % m;
        a = t;
        t = x0;
        x0 = x1 - q * x0;
        x1 = t;
    }
    if (x1 < 0) x1 += m0;
    return x1;
}

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
    (void)moduli; (void)barrett_cr; (void)db_rows_padded;
    (void)crt_count; (void)max_adds; (void)mod0_inv_mod1;
    (void)mod1_inv_mod0; (void)barrett_cr_0_modulus; (void)barrett_cr_1_modulus;

    ToeplitzContext* ctx = new ToeplitzContext();
    if (cublasCreate(&ctx->handle) != CUBLAS_STATUS_SUCCESS) {
        printf("ERROR: cublasCreate failed.\n");
        delete ctx;
        return nullptr;
    }

    ctx->n = n;
    ctx->l1 = db_rows;
    ctx->l2 = db_cols;

    printf("CRT+cuBLAS (2 primes): n=%u, l1=%u, l2=%u\n", n, db_rows, db_cols);

    // Allocate and copy DB (uint8)
    size_t bytes_db = (size_t)db_rows * db_cols * sizeof(uint8_t);
    if (cudaMalloc(&ctx->d_db_u8, bytes_db) != cudaSuccess) {
        printf("ERROR: Alloc d_db_u8 failed.\n");
        cublasDestroy(ctx->handle);
        delete ctx;
        return nullptr;
    }
    cudaMemcpy(ctx->d_db_u8, db, bytes_db, cudaMemcpyHostToDevice);

    // Build A matrices (double mod p) for both primes
    size_t size_A_d = (size_t)n * db_rows * sizeof(double);
    cudaMalloc(&ctx->d_A_p1, size_A_d);
    cudaMalloc(&ctx->d_A_p2, size_A_d);

    uint32_t* d_poly;
    cudaMalloc(&d_poly, db_rows * sizeof(uint32_t));
    cudaMemcpy(d_poly, v_nega_perm_a, db_rows * sizeof(uint32_t), cudaMemcpyHostToDevice);
    
    uint32_t num_polys = db_rows / n;
    dim3 block(256);
    dim3 grid((n + 255) / 256);
    
    for (uint32_t i = 0; i < num_polys; i++) {
        size_t offset = (size_t)i * n * n;
        build_toeplitz_matrix_negacyclic_double<<<grid, block>>>(
            d_poly + i*n, ctx->d_A_p1 + offset, n, P1);
        build_toeplitz_matrix_negacyclic_double<<<grid, block>>>(
            d_poly + i*n, ctx->d_A_p2 + offset, n, P2);
    }
    cudaFree(d_poly);

    // Allocate result buffers
    size_t size_C_d = (size_t)n * db_cols * sizeof(double);
    cudaMalloc(&ctx->d_C_p1, size_C_d);
    cudaMalloc(&ctx->d_C_p2, size_C_d);

    // Precompute CRT inverses
    ctx->inv1 = mod_inverse_u64(P2, P1);  // inv(P2) mod P1
    ctx->inv2 = mod_inverse_u64(P1, P2);  // inv(P1) mod P2

    cudaDeviceSynchronize();
    return ctx;
}

int compute_hint_0_toeplitz(void* context, uint64_t* hint_0) 
{
    ToeplitzContext* ctx = (ToeplitzContext*)context;
    if (!ctx) return -1;

    printf("Computing with CRT+cuBLAS (2 primes, single D buffer)...\n");

    // Allocate temporary D buffer (reused for both primes)
    double* d_D_temp;
    size_t size_D_d = (size_t)ctx->l1 * ctx->l2 * sizeof(double);
    if (cudaMalloc(&d_D_temp, size_D_d) != cudaSuccess) {
        printf("ERROR: Alloc d_D_temp failed.\n");
        return -1;
    }

    double alpha = 1.0;
    double beta = 0.0;
    uint32_t total_db = ctx->l1 * ctx->l2;
    dim3 block(256);
    dim3 grid_db((total_db + 255) / 256);

    // Process P1
    db_u8_to_double<<<grid_db, block>>>(ctx->d_db_u8, d_D_temp, total_db, P1);
    if (cublasDgemm(ctx->handle, CUBLAS_OP_N, CUBLAS_OP_N,
        ctx->n, ctx->l2, ctx->l1,
        &alpha, ctx->d_A_p1, ctx->n, d_D_temp, ctx->l1,
        &beta, ctx->d_C_p1, ctx->n) != CUBLAS_STATUS_SUCCESS) {
        printf("ERROR: cublasDgemm P1 failed.\n");
        cudaFree(d_D_temp);
        return -1;
    }

    // Process P2
    db_u8_to_double<<<grid_db, block>>>(ctx->d_db_u8, d_D_temp, total_db, P2);
    if (cublasDgemm(ctx->handle, CUBLAS_OP_N, CUBLAS_OP_N,
        ctx->n, ctx->l2, ctx->l1,
        &alpha, ctx->d_A_p2, ctx->n, d_D_temp, ctx->l1,
        &beta, ctx->d_C_p2, ctx->n) != CUBLAS_STATUS_SUCCESS) {
        printf("ERROR: cublasDgemm P2 failed.\n");
        cudaFree(d_D_temp);
        return -1;
    }

    cudaFree(d_D_temp);

    // CRT Reconstruction
    uint32_t total = ctx->n * ctx->l2;
    uint64_t* d_final;
    cudaMalloc(&d_final, total * sizeof(uint64_t));

    dim3 grid((total + 255) / 256);
    crt_reconstruct_2_double<<<grid, block>>>(
        ctx->d_C_p1, ctx->d_C_p2, d_final,
        ctx->n, ctx->l2,
        P1, P2, ctx->inv1, ctx->inv2
    );

    cudaMemcpy(hint_0, d_final, total * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_final);

    return 0;
}


} // extern "C"
