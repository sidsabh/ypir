/**
 * Tensor Core InspiRING Packing — Implementation
 *
 * Key insight: precompute M = Σ_i diag(T_i)·P_{σ_i} offline.
 * M[z, z'] = Σ_{i: σ_i(z)=z'} T[it+k, z], absorbing permutations into server data.
 * Online packing becomes a standard GEMM: result = Y · M, no permutation gathering.
 *
 * With byte decomposition (30-bit CRT residues → 4 uint8), the GEMM uses tensor cores.
 * 16 cross-product GEMMs × 2 CRT = 32 CUTLASS calls, same dimensions.
 *
 * GEMM shape: [B × tN] × [tN × ρN] → [B × ρN]
 *   B = batch (32-64), tN = 6144, ρN = 32*2048 = 65536
 */

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

// CUTLASS
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

#include "common/ntt.cuh"
#include "inspiring/tc_packing.cuh"
#include "inspiring/packing.cuh"

// ═══════════════════════════════════════════════════════════════
// CUTLASS GEMM types: uint8 × uint8 → int32
//   SM80+ : Tensor cores (Ampere, wider tiles)
//   SM75  : Tensor cores (Turing)
//   SM50  : SIMT fallback (any GPU, no tensor cores needed)
// ═══════════════════════════════════════════════════════════════

// Ampere+ tensor cores
using TcGemm_Sm80 = cutlass::gemm::device::Gemm<
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

// Turing tensor cores
using TcGemm_Sm75 = cutlass::gemm::device::Gemm<
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

// SIMT fallback (any GPU — no tensor cores needed)
using SimtGemm_Sm50 = cutlass::gemm::device::Gemm<
    uint8_t, cutlass::layout::RowMajor,
    uint8_t, cutlass::layout::ColumnMajor,
    int32_t, cutlass::layout::ColumnMajor,
    int32_t,
    cutlass::arch::OpClassSimt, cutlass::arch::Sm50,
    cutlass::gemm::GemmShape<64, 64, 8>,
    cutlass::gemm::GemmShape<32, 32, 8>,
    cutlass::gemm::GemmShape<1, 1, 1>,
    cutlass::epilogue::thread::LinearCombination<int32_t, 1, int32_t, int32_t>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>, 2
>;

// gpu_tier: 0 = CUTLASS SIMT (sm<75), 1 = SM75 TC, 2 = SM80+ TC
// A: [M × K] RowMajor, B: [K × N] ColumnMajor, C: [M × N] ColumnMajor
static cutlass::Status run_tc_gemm(int gpu_tier,
    int M, int N, int K,
    const uint8_t* A, int lda,
    const uint8_t* B, int ldb,
    int32_t* C, int ldc,
    int32_t alpha, int32_t beta)
{
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    if (gpu_tier >= 2) {
        TcGemm_Sm80::Arguments args{
            problem_size, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta}, 1
        };
        TcGemm_Sm80 gemm_op;
        auto s = gemm_op.initialize(args, nullptr);
        if (s != cutlass::Status::kSuccess) return s;
        return gemm_op();
    } else if (gpu_tier == 1) {
        TcGemm_Sm75::Arguments args{
            problem_size, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta}, 1
        };
        TcGemm_Sm75 gemm_op;
        auto s = gemm_op.initialize(args, nullptr);
        if (s != cutlass::Status::kSuccess) return s;
        return gemm_op();
    } else {
        SimtGemm_Sm50::Arguments args{
            problem_size, {A, lda}, {B, ldb}, {C, ldc}, {C, ldc}, {alpha, beta}, 1
        };
        SimtGemm_Sm50 gemm_op;
        auto s = gemm_op.initialize(args, nullptr);
        if (s != cutlass::Status::kSuccess) return s;
        return gemm_op();
    }
}

// ═══════════════════════════════════════════════════════════════
// GPU Kernels
// ═══════════════════════════════════════════════════════════════

/**
 * Build full M_combined from bold_t, bold_t_bar (fast path, needs 2×K×N uint64 temp).
 * Grid: (ceil(N/256), ρ, t)
 */
__global__ void build_M_combined_kernel(
    uint64_t* __restrict__ d_M_crt0,
    uint64_t* __restrict__ d_M_crt1,
    const uint64_t* __restrict__ d_bold_t,
    const uint64_t* __restrict__ d_bold_t_bar,
    const uint32_t* __restrict__ d_tables,
    const uint32_t* __restrict__ d_gen_pows,
    size_t num_iter, size_t t_exp_left, size_t poly_len,
    size_t num_outputs, size_t K_gemm)
{
    size_t z = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= poly_len) return;

    size_t o = blockIdx.y;
    size_t k = blockIdx.z;
    size_t D = num_iter * t_exp_left;
    size_t col = o * poly_len + z;

    for (size_t i = 0; i < num_iter; i++) {
        uint32_t gp = d_gen_pows[i];
        uint32_t z_prime     = d_tables[((gp - 1) / 2) * poly_len + z];
        uint32_t z_prime_bar = d_tables[((2 * (uint32_t)poly_len - gp - 1) / 2) * poly_len + z];

        size_t bold_t_idx = o * D * poly_len + (i * t_exp_left + k) * poly_len + z;
        uint64_t t_val     = d_bold_t[bold_t_idx];
        uint64_t t_bar_val = d_bold_t_bar[bold_t_idx];

        uint32_t t0  = (uint32_t)(t_val & 0xFFFFFFFF);
        uint32_t t1  = (uint32_t)(t_val >> 32);
        uint32_t tb0 = (uint32_t)(t_bar_val & 0xFFFFFFFF);
        uint32_t tb1 = (uint32_t)(t_bar_val >> 32);

        size_t idx1 = (k * poly_len + z_prime) + col * K_gemm;
        atomicAdd((unsigned long long*)&d_M_crt0[idx1], (unsigned long long)t0);
        atomicAdd((unsigned long long*)&d_M_crt1[idx1], (unsigned long long)t1);

        size_t idx2 = (k * poly_len + z_prime_bar) + col * K_gemm;
        atomicAdd((unsigned long long*)&d_M_crt0[idx2], (unsigned long long)tb0);
        atomicAdd((unsigned long long*)&d_M_crt1[idx2], (unsigned long long)tb1);
    }
}

/**
 * Barrett-reduce and byte-decompose a full CRT array.
 */
__global__ void reduce_and_byte_decompose_kernel(
    uint8_t* __restrict__ out,
    const uint64_t* __restrict__ in,
    size_t count,
    uint64_t mod,
    uint64_t barrett_cr)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t val = barrett_raw_u64(in[idx], barrett_cr, mod);
    uint32_t v = (uint32_t)val;

    out[0 * count + idx] = (uint8_t)(v & 0xFF);
    out[1 * count + idx] = (uint8_t)((v >> 8) & 0xFF);
    out[2 * count + idx] = (uint8_t)((v >> 16) & 0xFF);
    out[3 * count + idx] = (uint8_t)((v >> 24) & 0xFF);
}

/**
 * Build one tile of M_combined: single CRT component, single output.
 *
 * For output o, CRT component crt_idx, gadget k:
 *   tile[k*N + σ_i(z), z] += T_o[it+k, z]_{crt}     (Term 1)
 *   tile[k*N + σ̄_i(z), z] += T̄_o[it+k, z]_{crt}    (Term 2)
 *
 * Tile is ColumnMajor [K_gemm × N] uint64, ldb = K_gemm.
 * Grid: (ceil(N/256), 1, t)
 */
__global__ void build_M_tile_kernel(
    uint64_t* __restrict__ d_tile,                  // [K_gemm × N] ColMaj, zero-initialized
    const uint64_t* __restrict__ d_bold_t,          // [ρ × D × N] condensed
    const uint64_t* __restrict__ d_bold_t_bar,      // [ρ × D × N] condensed
    const uint32_t* __restrict__ d_tables,          // [num_tables × N]
    const uint32_t* __restrict__ d_gen_pows,        // [R]
    size_t num_iter,        // R
    size_t t_exp_left,      // t
    size_t poly_len,        // N
    size_t output_idx,      // which output o
    size_t K_gemm,          // t × N
    int crt_idx)            // 0 = low 32 bits, 1 = high 32 bits
{
    size_t z = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= poly_len) return;

    size_t k = blockIdx.z;      // gadget index
    size_t o = output_idx;
    size_t D = num_iter * t_exp_left;

    for (size_t i = 0; i < num_iter; i++) {
        uint32_t gp = d_gen_pows[i];
        uint32_t tidx_sigma     = (gp - 1) / 2;
        uint32_t tidx_sigma_bar = (2 * (uint32_t)poly_len - gp - 1) / 2;

        uint32_t z_prime     = d_tables[tidx_sigma     * poly_len + z];
        uint32_t z_prime_bar = d_tables[tidx_sigma_bar * poly_len + z];

        size_t bold_t_idx = o * D * poly_len + (i * t_exp_left + k) * poly_len + z;
        uint64_t t_val     = d_bold_t[bold_t_idx];
        uint64_t t_bar_val = d_bold_t_bar[bold_t_idx];

        uint32_t t_crt  = (crt_idx == 0) ? (uint32_t)(t_val & 0xFFFFFFFF)     : (uint32_t)(t_val >> 32);
        uint32_t tb_crt = (crt_idx == 0) ? (uint32_t)(t_bar_val & 0xFFFFFFFF) : (uint32_t)(t_bar_val >> 32);

        // Term 1: tile[k*N + σ_i(z), z]
        size_t idx1 = (k * poly_len + z_prime) + z * K_gemm;
        atomicAdd((unsigned long long*)&d_tile[idx1], (unsigned long long)t_crt);

        // Term 2: tile[k*N + σ̄_i(z), z]
        size_t idx2 = (k * poly_len + z_prime_bar) + z * K_gemm;
        atomicAdd((unsigned long long*)&d_tile[idx2], (unsigned long long)tb_crt);
    }
}

/**
 * Barrett-reduce and byte-decompose a tile into the full M byte-slice array.
 *
 * Reads from tile [K × N] uint64, writes 4 byte slices into the correct
 * offset of the full [K × N_gemm] byte-slice allocation.
 */
__global__ void reduce_and_byte_decompose_tile_kernel(
    uint8_t* __restrict__ out,      // base of full [4 × K × N_gemm] byte-slice allocation
    const uint64_t* __restrict__ in,
    size_t tile_elems,              // K × N (elements in this tile)
    size_t full_elems,              // K × N_gemm (elements in full M)
    size_t tile_offset,             // o × N × K (column offset in ColumnMajor)
    uint64_t mod,
    uint64_t barrett_cr)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= tile_elems) return;

    uint64_t val = barrett_raw_u64(in[idx], barrett_cr, mod);
    uint32_t v = (uint32_t)val;

    size_t out_idx = tile_offset + idx;
    out[0 * full_elems + out_idx] = (uint8_t)(v & 0xFF);
    out[1 * full_elems + out_idx] = (uint8_t)((v >> 8) & 0xFF);
    out[2 * full_elems + out_idx] = (uint8_t)((v >> 16) & 0xFF);
    out[3 * full_elems + out_idx] = (uint8_t)((v >> 24) & 0xFF);
}

/**
 * Byte-decompose condensed uint64 array into 8 uint8 slices (4 per CRT).
 * Input and output have the same element count; layout is preserved.
 *
 * Output: out[slice * count + idx] = byte slice of in[idx].
 *   Slices 0-3: CRT0 bytes 0-3
 *   Slices 4-7: CRT1 bytes 0-3
 */
__global__ void byte_decompose_condensed_kernel(
    uint8_t* __restrict__ out,      // [8 × count] contiguous
    const uint64_t* __restrict__ in,
    size_t count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t val = in[idx];
    uint32_t crt0 = (uint32_t)(val & 0xFFFFFFFF);
    uint32_t crt1 = (uint32_t)(val >> 32);

    out[0 * count + idx] = (uint8_t)(crt0 & 0xFF);
    out[1 * count + idx] = (uint8_t)((crt0 >> 8) & 0xFF);
    out[2 * count + idx] = (uint8_t)((crt0 >> 16) & 0xFF);
    out[3 * count + idx] = (uint8_t)((crt0 >> 24) & 0xFF);
    out[4 * count + idx] = (uint8_t)(crt1 & 0xFF);
    out[5 * count + idx] = (uint8_t)((crt1 >> 8) & 0xFF);
    out[6 * count + idx] = (uint8_t)((crt1 >> 16) & 0xFF);
    out[7 * count + idx] = (uint8_t)((crt1 >> 24) & 0xFF);
}

/**
 * Accumulate shift: result[idx] += (uint64)(uint32)G[idx] * shift_const
 */
__global__ void accumulate_shift_kernel(
    uint64_t* __restrict__ result,
    const int32_t* __restrict__ G,
    uint64_t shift_const,
    size_t count)
{
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    uint64_t g = (uint64_t)(uint32_t)G[idx];
    result[idx] += g * shift_const;
}

/**
 * Finalize: format result_crt0/crt1 into scratch for inspir_post_process,
 * adding Term 3 (T̂ · z_body) and reducing mod q0/q1.
 *
 * For each (client c, output o, coeff z):
 *   crt0 = result_crt0[c * N_gemm + o*N + z]  +  Σ_k T̂_o[k,z]_lo · z_k[c,z]_lo
 *   crt1 = result_crt1[c * N_gemm + o*N + z]  +  Σ_k T̂_o[k,z]_hi · z_k[c,z]_hi
 *   scratch[c*scratch_stride + o*spo + z]     = crt0 mod q0
 *   scratch[c*scratch_stride + o*spo + N + z] = crt1 mod q1
 */
__global__ void finalize_to_scratch_kernel(
    uint64_t* __restrict__ d_scratch,
    const uint64_t* __restrict__ d_result_crt0,
    const uint64_t* __restrict__ d_result_crt1,
    const uint64_t* __restrict__ d_bold_t_hat,    // [ρ × t × N] condensed
    const uint64_t* __restrict__ d_z_body,        // [B × t × N] condensed
    size_t result_stride,       // column-major stride of d_result (may be padded)
    size_t num_outputs,
    size_t poly_len,
    size_t t_exp_left,
    size_t N_gemm,
    size_t scratch_stride,      // num_outputs * inspir_spo
    size_t inspir_spo,          // 4 * poly_len
    uint64_t mod0,
    uint64_t mod1,
    uint64_t barrett_cr0,
    uint64_t barrett_cr1)
{
    // Grid: (ceil(N/256), ρ, B)
    size_t z = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (z >= poly_len) return;

    size_t o = blockIdx.y;
    size_t c = blockIdx.z;

    // Read GEMM result (ColumnMajor: stride = result_stride, index = c + col * stride)
    size_t gemm_idx = c + (o * poly_len + z) * result_stride;
    uint64_t acc0 = d_result_crt0[gemm_idx];
    uint64_t acc1 = d_result_crt1[gemm_idx];

    // Add Term 3: Σ_k T̂_o[k,z] · z_k[c,z]
    size_t z_body_stride = t_exp_left * poly_len;
    for (size_t k = 0; k < t_exp_left; k++) {
        uint64_t that_val = d_bold_t_hat[o * t_exp_left * poly_len + k * poly_len + z];
        uint64_t z_val    = d_z_body[c * z_body_stride + k * poly_len + z];

        uint32_t t0 = (uint32_t)(that_val & 0xFFFFFFFF);
        uint32_t t1 = (uint32_t)(that_val >> 32);
        uint32_t z0 = (uint32_t)(z_val & 0xFFFFFFFF);
        uint32_t z1 = (uint32_t)(z_val >> 32);

        acc0 += (uint64_t)t0 * z0;
        acc1 += (uint64_t)t1 * z1;
    }

    // Barrett reduce mod q0, q1
    uint64_t r0 = barrett_raw_u64(acc0, barrett_cr0, mod0);
    uint64_t r1 = barrett_raw_u64(acc1, barrett_cr1, mod1);

    // Write to scratch in inspir_post_process format
    size_t scratch_base = c * scratch_stride + o * inspir_spo;
    d_scratch[scratch_base + z]            = r0;
    d_scratch[scratch_base + poly_len + z] = r1;
}

// ═══════════════════════════════════════════════════════════════
// Cross-product table: (byte_i, byte_j) pairs grouped by shift s = i+j
// ═══════════════════════════════════════════════════════════════

struct CrossProduct { int i, j; };

static const CrossProduct CROSS_PRODUCTS[16] = {
    // shift 0
    {0, 0},
    // shift 1
    {0, 1}, {1, 0},
    // shift 2
    {0, 2}, {1, 1}, {2, 0},
    // shift 3
    {0, 3}, {1, 2}, {2, 1}, {3, 0},
    // shift 4
    {1, 3}, {2, 2}, {3, 1},
    // shift 5
    {2, 3}, {3, 2},
    // shift 6
    {3, 3}
};

// Start index in CROSS_PRODUCTS for each shift s=0..6
static const int SHIFT_START[8] = {0, 1, 3, 6, 10, 13, 15, 16};

// ═══════════════════════════════════════════════════════════════
// Extern "C" API
// ═══════════════════════════════════════════════════════════════

extern "C" {

void* tc_packing_init(
    uint64_t* d_bold_t_condensed,
    uint64_t* d_bold_t_bar_condensed,
    const uint64_t* d_bold_t_hat_condensed,
    const uint64_t* d_a_hat,
    const uint32_t* d_tables,
    const uint32_t* d_gen_pows,
    size_t num_iter,
    size_t t_exp_left,
    size_t poly_len,
    size_t num_outputs,
    size_t max_batch_size,
    uint64_t rlwe_q_prime_1,
    uint64_t rlwe_q_prime_2,
    size_t response_bytes_per_output,
    NTTParams ntt_params,
    int gpu_tier)
{
    TcPackingContext* ctx = new TcPackingContext();

    ctx->poly_len = poly_len;
    ctx->t_exp_left = t_exp_left;
    ctx->num_iter = num_iter;
    ctx->num_outputs = num_outputs;
    ctx->max_batch_size = max_batch_size;
    ctx->K_gemm = t_exp_left * poly_len;
    ctx->N_gemm = num_outputs * poly_len;
    ctx->gpu_tier = gpu_tier;
    ctx->rlwe_q_prime_1 = rlwe_q_prime_1;
    ctx->rlwe_q_prime_2 = rlwe_q_prime_2;
    ctx->response_bytes_per_output = response_bytes_per_output;
    ctx->ntt_params = ntt_params;

    // Copy moduli/barrett from device to host-side fields
    CUDA_ASSERT(cudaMemcpy(&ctx->mod0, ntt_params.moduli + 0, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(&ctx->mod1, ntt_params.moduli + 1, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(&ctx->barrett_cr0, ntt_params.barrett_cr + 0, sizeof(uint64_t), cudaMemcpyDeviceToHost));
    CUDA_ASSERT(cudaMemcpy(&ctx->barrett_cr1, ntt_params.barrett_cr + 1, sizeof(uint64_t), cudaMemcpyDeviceToHost));

    size_t K = ctx->K_gemm;
    size_t N = ctx->N_gemm;
    size_t M_elems = K * N;  // total elements in M

    size_t tile_elems = K * poly_len;  // elements per tile (one output)

    // Check available VRAM to decide between fast (full alloc) and tiled (low-memory) paths.
    // Fast path needs: 2 × M_elems × 8 (d_M_crt0/crt1) + 2 × 4 × M_elems (byte slices)
    // Tiled path needs: tile_elems × 8 (one tile) + 2 × 4 × M_elems (byte slices)
    size_t fast_path_temp = 2 * M_elems * sizeof(uint64_t);  // d_M_crt0 + d_M_crt1
    size_t free_mem, total_mem;
    CUDA_ASSERT(cudaMemGetInfo(&free_mem, &total_mem));
    bool use_tiled = (free_mem < fast_path_temp + 8 * M_elems + (256ULL << 20));  // need temp + bytes + 256MB headroom

    printf("TcPacking init: building M_combined [%zu × %zu] = %.1f MB per CRT (%s path, %.1f MB free)\n",
           K, N, M_elems * 8.0 / 1e6, use_tiled ? "tiled" : "fast", free_mem / 1e6);

    GpuTimer timer;
    timer.tic();

    // ── Allocate final M byte-slices (persistent, both paths) ──

    uint8_t* d_M_bytes_crt0;
    uint8_t* d_M_bytes_crt1;
    CUDA_ASSERT(cudaMalloc(&d_M_bytes_crt0, 4 * M_elems));
    CUDA_ASSERT(cudaMalloc(&d_M_bytes_crt1, 4 * M_elems));

    if (use_tiled) {
        // ── Tiled path: one CRT × one output at a time ──
        // Peak = d_M_bytes + bold_t + tile (~100 MB). Fits on 16 GB GPUs.

        uint64_t* d_tile;
        CUDA_ASSERT(cudaMalloc(&d_tile, tile_elems * sizeof(uint64_t)));

        int threads = 256;
        int blocks_z = ((int)poly_len + threads - 1) / threads;
        dim3 build_grid(blocks_z, 1, (int)t_exp_left);
        int decomp_blocks = ((int)tile_elems + threads - 1) / threads;

        for (int crt = 0; crt < 2; crt++) {
            uint8_t* d_M_bytes_crt = (crt == 0) ? d_M_bytes_crt0 : d_M_bytes_crt1;
            uint64_t mod       = (crt == 0) ? ctx->mod0       : ctx->mod1;
            uint64_t barrett_cr = (crt == 0) ? ctx->barrett_cr0 : ctx->barrett_cr1;

            for (size_t o = 0; o < num_outputs; o++) {
                CUDA_ASSERT(cudaMemset(d_tile, 0, tile_elems * sizeof(uint64_t)));

                build_M_tile_kernel<<<build_grid, threads>>>(
                    d_tile,
                    d_bold_t_condensed,
                    d_bold_t_bar_condensed,
                    d_tables, d_gen_pows,
                    num_iter, t_exp_left, poly_len, o, K, crt);
                CUDA_ASSERT(cudaGetLastError());

                size_t tile_offset = o * poly_len * K;
                reduce_and_byte_decompose_tile_kernel<<<decomp_blocks, threads>>>(
                    d_M_bytes_crt, d_tile, tile_elems, M_elems, tile_offset,
                    mod, barrett_cr);
                CUDA_ASSERT(cudaGetLastError());
            }
        }

        CUDA_ASSERT(cudaDeviceSynchronize());
        CUDA_ASSERT(cudaFree(d_tile));

    } else {
        // ── Fast path: build both CRTs at once, then byte-decompose ──

        uint64_t* d_M_crt0;
        uint64_t* d_M_crt1;
        CUDA_ASSERT(cudaMalloc(&d_M_crt0, M_elems * sizeof(uint64_t)));
        CUDA_ASSERT(cudaMalloc(&d_M_crt1, M_elems * sizeof(uint64_t)));
        CUDA_ASSERT(cudaMemset(d_M_crt0, 0, M_elems * sizeof(uint64_t)));
        CUDA_ASSERT(cudaMemset(d_M_crt1, 0, M_elems * sizeof(uint64_t)));

        {
            int threads = 256;
            int blocks_z = ((int)poly_len + threads - 1) / threads;
            dim3 grid(blocks_z, (int)num_outputs, (int)t_exp_left);

            build_M_combined_kernel<<<grid, threads>>>(
                d_M_crt0, d_M_crt1,
                d_bold_t_condensed, d_bold_t_bar_condensed,
                d_tables, d_gen_pows,
                num_iter, t_exp_left, poly_len, num_outputs, K);
            CUDA_ASSERT(cudaGetLastError());
            CUDA_ASSERT(cudaDeviceSynchronize());
        }

        {
            int threads = 256;
            int blocks = ((int)M_elems + threads - 1) / threads;
            reduce_and_byte_decompose_kernel<<<blocks, threads>>>(
                d_M_bytes_crt0, d_M_crt0, M_elems, ctx->mod0, ctx->barrett_cr0);
            reduce_and_byte_decompose_kernel<<<blocks, threads>>>(
                d_M_bytes_crt1, d_M_crt1, M_elems, ctx->mod1, ctx->barrett_cr1);
            CUDA_ASSERT(cudaGetLastError());
        }

        CUDA_ASSERT(cudaFree(d_M_crt0));
        CUDA_ASSERT(cudaFree(d_M_crt1));
    }

    // bold_t and bold_t_bar are fully absorbed into M — free to reduce memory
    if (d_bold_t_condensed)     { CUDA_ASSERT(cudaFree(d_bold_t_condensed)); }
    if (d_bold_t_bar_condensed) { CUDA_ASSERT(cudaFree(d_bold_t_bar_condensed)); }

    // Set up pointer array: d_M_bytes[crt][byte_idx]
    for (int b = 0; b < 4; b++) {
        ctx->d_M_bytes[0][b] = d_M_bytes_crt0 + (size_t)b * M_elems;
        ctx->d_M_bytes[1][b] = d_M_bytes_crt1 + (size_t)b * M_elems;
    }

    // ── Step 3: Copy T̂ and â (small, keep as condensed) ──

    size_t t_hat_size = num_outputs * t_exp_left * poly_len;
    CUDA_ASSERT(cudaMalloc(&ctx->d_bold_t_hat_condensed, t_hat_size * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMemcpy(ctx->d_bold_t_hat_condensed, d_bold_t_hat_condensed,
                           t_hat_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));

    size_t a_hat_size = num_outputs * poly_len;
    CUDA_ASSERT(cudaMalloc(&ctx->d_a_hat, a_hat_size * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMemcpy(ctx->d_a_hat, d_a_hat,
                           a_hat_size * sizeof(uint64_t), cudaMemcpyDeviceToDevice));

    // ── Step 4: Allocate online scratch ──

    // A byte-slices: 8 × [max_B × K_gemm] uint8
    size_t A_slice_size = max_batch_size * K;
    CUDA_ASSERT(cudaMalloc(&ctx->d_A_bytes_buf, 8 * A_slice_size));

    // Shift accumulator: [max_B × N_gemm] int32
    CUDA_ASSERT(cudaMalloc(&ctx->d_G, max_batch_size * N * sizeof(int32_t)));

    // CRT running totals: [max_B × N_gemm] uint64 each
    CUDA_ASSERT(cudaMalloc(&ctx->d_result_crt0, max_batch_size * N * sizeof(uint64_t)));
    CUDA_ASSERT(cudaMalloc(&ctx->d_result_crt1, max_batch_size * N * sizeof(uint64_t)));

    // Post-process scratch: [max_B × ρ × 4N] uint64
    size_t inspir_spo = 4 * poly_len;
    size_t scratch_total = max_batch_size * num_outputs * inspir_spo;
    CUDA_ASSERT(cudaMalloc(&ctx->d_scratch, scratch_total * sizeof(uint64_t)));

    float ms = timer.toc_ms();
    printf("TcPacking init done: M build + byte-decompose in %.1f ms\n", ms);
    printf("  M byte-slices: %.1f MB, A scratch: %.1f MB, G: %.1f MB, result_crt: %.1f MB\n",
           8.0 * M_elems / 1e6, 8.0 * A_slice_size / 1e6,
           max_batch_size * N * 4.0 / 1e6,
           2.0 * max_batch_size * N * 8.0 / 1e6);

    return ctx;
}

void tc_packing_run(
    void* context,
    const uint64_t* d_intermediate,
    const uint64_t* d_y_body_condensed,
    const uint64_t* d_z_body_condensed,
    uint8_t* d_response_out,
    size_t batch_size)
{
    TcPackingContext* ctx = (TcPackingContext*)context;
    if (!ctx || batch_size == 0) return;

    size_t N      = ctx->poly_len;
    size_t K      = ctx->K_gemm;    // t × N
    size_t Ng     = ctx->N_gemm;    // ρ × N
    size_t B      = batch_size;
    size_t rho    = ctx->num_outputs;
    size_t t      = ctx->t_exp_left;

    GpuTimer timer;
    timer.tic();

    // ── Step 1: Byte-decompose client keys A = [y_0|y_1|...|y_{t-1}] ──
    // y_body is [B × t × N] condensed uint64, already contiguous as [B × K] RowMajor
    size_t A_elems = B * K;
    {
        int threads = 256;
        int blocks = ((int)A_elems + threads - 1) / threads;
        byte_decompose_condensed_kernel<<<blocks, threads>>>(
            ctx->d_A_bytes_buf, d_y_body_condensed, A_elems);
        CUDA_ASSERT(cudaGetLastError());
    }

    // Pointers to A byte-slices: A_bytes[crt][byte_idx] each [B × K] uint8
    uint8_t* A_bytes[2][4];
    for (int crt = 0; crt < 2; crt++)
        for (int b = 0; b < 4; b++)
            A_bytes[crt][b] = ctx->d_A_bytes_buf + (size_t)(crt * 4 + b) * A_elems;

    // ── Step 2: Cross-product GEMMs with shift accumulation ──
    // For each CRT component, process shifts 0..6 incrementally.
    // GEMM shape: [B × K] × [K × Ng] → [B × Ng]
    // A: RowMajor, lda = K.  M: ColumnMajor, ldb = K.  C: ColumnMajor, ldc = B.

    // CUTLASS epilogue does vectorized int32 stores (width 4 for TC, 1 for SIMT).
    // ldc must be a multiple of this width, otherwise we get misaligned address errors.
    // Round M up to next multiple of 4 when using tensor cores.
    int align = (ctx->gpu_tier >= 1) ? 4 : 1;
    int M_dim = ((int)B + align - 1) / align * align;
    int K_dim = (int)K;
    int N_dim = (int)Ng;
    size_t BN = (size_t)M_dim * Ng;  // padded batch × Ng for GEMM output/G/result

    uint64_t mod0 = ctx->mod0;
    uint64_t mod1 = ctx->mod1;

    // Precompute 2^{8s} mod q for reconstruction
    uint64_t shift_mod[2][7];
    for (int s = 0; s < 7; s++) {
        uint64_t pow = 1;
        for (int p = 0; p < 8 * s; p++) pow = (pow * 2) % mod0;
        shift_mod[0][s] = pow;
        pow = 1;
        for (int p = 0; p < 8 * s; p++) pow = (pow * 2) % mod1;
        shift_mod[1][s] = pow;
    }

    for (int crt = 0; crt < 2; crt++) {
        uint64_t* d_result = (crt == 0) ? ctx->d_result_crt0 : ctx->d_result_crt1;
        CUDA_ASSERT(cudaMemset(d_result, 0, BN * sizeof(uint64_t)));

        for (int s = 0; s < 7; s++) {
            int start = SHIFT_START[s];
            int end   = SHIFT_START[s + 1];
            if (start == end) continue;

            for (int cp = start; cp < end; cp++) {
                int bi = CROSS_PRODUCTS[cp].i;
                int bj = CROSS_PRODUCTS[cp].j;
                int32_t beta = (cp == start) ? 0 : 1;

                auto status = run_tc_gemm(ctx->gpu_tier,
                    M_dim, N_dim, K_dim,
                    A_bytes[crt][bi], K_dim,
                    ctx->d_M_bytes[crt][bj], K_dim,
                    ctx->d_G, M_dim,
                    1, beta);

                if (status != cutlass::Status::kSuccess) {
                    fprintf(stderr, "TcPacking GEMM failed: crt=%d shift=%d cp=(%d,%d)\n",
                            crt, s, bi, bj);
                    return;
                }
            }

            // Accumulate: result += G * 2^{8s} mod q
            {
                int threads = 256;
                int blocks = ((int)BN + threads - 1) / threads;
                accumulate_shift_kernel<<<blocks, threads>>>(
                    d_result, ctx->d_G, shift_mod[crt][s], BN);
                CUDA_ASSERT(cudaGetLastError());
            }
        }
    }

    // ── Step 3: Finalize — format into scratch + Term 3 + mod reduce ──
    {
        size_t inspir_spo = 4 * N;
        size_t scratch_stride = rho * inspir_spo;

        int threads = 256;
        int blocks_z = ((int)N + threads - 1) / threads;
        dim3 grid(blocks_z, (int)rho, (int)B);

        finalize_to_scratch_kernel<<<grid, threads>>>(
            ctx->d_scratch,
            ctx->d_result_crt0,
            ctx->d_result_crt1,
            ctx->d_bold_t_hat_condensed,
            d_z_body_condensed,
            (size_t)M_dim, rho, N, t, Ng,
            scratch_stride, inspir_spo,
            mod0, mod1,
            ctx->barrett_cr0,
            ctx->barrett_cr1);
        CUDA_ASSERT(cudaGetLastError());
    }

    // ── Step 4: Post-process per client (INTT + CRT compose + add b + modswitch + bitpack) ──
    {
        size_t inspir_spo = 4 * N;
        size_t scratch_stride = rho * inspir_spo;
        size_t M_inter = rho * N;  // intermediate stride per client

        for (size_t b = 0; b < B; b++) {
            inspir_post_process<<<(int)rho, 1024>>>(
                d_response_out + b * rho * ctx->response_bytes_per_output,
                d_intermediate + b * M_inter,
                ctx->d_a_hat,
                ctx->d_scratch + b * scratch_stride,
                rho,
                ctx->rlwe_q_prime_1,
                ctx->rlwe_q_prime_2,
                ctx->response_bytes_per_output,
                inspir_spo,
                ctx->ntt_params);
            CUDA_ASSERT(cudaGetLastError());
        }
    }

    CUDA_ASSERT(cudaDeviceSynchronize());

    float ms = timer.toc_ms();
    printf("TcPacking run: %.1f ms (%zu clients, %zu outputs, GEMM [%d × %d × %d])\n",
           ms, B, rho, M_dim, K_dim, N_dim);
}

void tc_packing_free(void* context)
{
    TcPackingContext* ctx = (TcPackingContext*)context;
    if (!ctx) return;

    // M byte-slices are two contiguous allocations (one per CRT)
    if (ctx->d_M_bytes[0][0]) cudaFree(ctx->d_M_bytes[0][0]);
    if (ctx->d_M_bytes[1][0]) cudaFree(ctx->d_M_bytes[1][0]);

    if (ctx->d_bold_t_hat_condensed) cudaFree(ctx->d_bold_t_hat_condensed);
    if (ctx->d_a_hat)                cudaFree(ctx->d_a_hat);
    if (ctx->d_A_bytes_buf)          cudaFree(ctx->d_A_bytes_buf);
    if (ctx->d_G)                    cudaFree(ctx->d_G);
    if (ctx->d_result_crt0)          cudaFree(ctx->d_result_crt0);
    if (ctx->d_result_crt1)          cudaFree(ctx->d_result_crt1);
    if (ctx->d_scratch)              cudaFree(ctx->d_scratch);

    delete ctx;
}

} // extern "C"
