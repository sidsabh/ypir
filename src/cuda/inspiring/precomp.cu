// ============================================================================
// GPU InspiRING Offline Precomputation
//
// Replaces the CPU full_packing_with_preprocessing_offline (packing.rs:2247)
// for word SimplePIR. Keeps hint_0 on GPU (no D2H transfer), computes bold_t
// results directly on GPU (no H2D transfer), and is ~100x faster.
//
// Algorithm:
//   Phase 1 (parallel): compute r_all[i] for i=0..1023 via monomial*a_ct_tilde
//                        multiply-adds + automorphism, per output
//   Phase 2 (sequential): backward recursion i=1022..0:
//                          gadget_invert(r_all[i+1]) -> bold_t[i]
//                          r_all[i] += w_all[i] · bold_t[i]
//   Final: bold_t_hat = gadget_invert(r_bar_all[0]),
//          r_all[0] += v_mask · bold_t_hat -> a_hat
// ============================================================================

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdint.h>
#include "common/ntt.cuh"

// ============================================================================
// Context
// ============================================================================

struct InspirPrecompContext {
    uint32_t poly_len;         // 2048
    uint32_t crt_count;        // 2
    uint32_t t_exp_left;       // 3
    uint32_t bits_per;         // floor(modulus_log2 / t_exp_left) + 1
    uint32_t num_outputs;
    uint32_t num_to_pack_half; // poly_len / 2
    uint32_t num_iter;         // poly_len / 2 - 1
    uint32_t q2_bits;

    NTTParams ntt_params;

    uint32_t* d_gen_pows;      // [poly_len]
    uint32_t* d_tables;        // [num_tables * poly_len]
    uint32_t num_tables;

    uint64_t* d_monomial_ntts;     // [poly_len * crt_count * poly_len]
    uint64_t* d_neg_monomial_ntts; // same
    uint64_t* d_mod_inv_poly;      // [crt_count * poly_len]
    uint64_t* d_a_ct_tilde;       // [num_outputs * poly_len * crt_count * poly_len]
    uint64_t* d_w_all;            // [num_iter * t_exp_left * crt_count * poly_len]
    uint64_t* d_w_bar_all;        // same
    uint64_t* d_v_mask;           // [t_exp_left * crt_count * poly_len]
    uint64_t* d_r_all;            // [num_outputs * num_to_pack_half * crt_count * poly_len]
    uint64_t* d_r_bar_all;        // same

    // Outputs (condensed, stay on GPU)
    uint64_t* d_bold_t_condensed;      // [num_outputs * num_iter * t_exp_left * poly_len]
    uint64_t* d_bold_t_bar_condensed;  // same
    uint64_t* d_bold_t_hat_condensed;  // [num_outputs * t_exp_left * poly_len]
    uint64_t* d_a_hat;                 // [num_outputs * poly_len]

    const uint64_t* d_hint_0;  // borrowed
    uint32_t db_cols;

    cudaStream_t stream;
};

// ============================================================================
// Kernel 1: Compute Monomial NTTs
// Each block computes NTT(X^j) and NTT(-X^j) for one j.
// Block: 1024 threads, Grid: poly_len, Smem: crt_count * poly_len u64
// ============================================================================

__global__ void inspir_compute_monomials(
    uint64_t* __restrict__ d_monomial_ntts,
    uint64_t* __restrict__ d_neg_monomial_ntts,
    NTTParams params)
{
    uint32_t j = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t poly_len = params.poly_len;
    uint32_t crt_count = params.crt_count;
    uint32_t cpn = crt_count * poly_len;

    extern __shared__ uint64_t smem[];

    // Build raw monomial e_j per CRT modulus (value is 0 or 1, same for all moduli)
    for (uint32_t m = 0; m < crt_count; m++) {
        for (uint32_t k = tid; k < poly_len; k += blockDim.x) {
            smem[m * poly_len + k] = (k == j) ? 1ULL : 0ULL;
        }
    }
    __syncthreads();

    for (uint32_t m = 0; m < crt_count; m++) {
        ntt_forward_kernel_parallel(smem + m * poly_len, &params, m, tid, blockDim.x);
    }

    uint64_t* dst = d_monomial_ntts + (size_t)j * cpn;
    for (uint32_t k = tid; k < cpn; k += blockDim.x) {
        dst[k] = smem[k];
    }

    // Negate: -X^j in NTT domain = modulus - coeff for each CRT
    for (uint32_t m = 0; m < crt_count; m++) {
        uint64_t mod_m = params.moduli[m];
        for (uint32_t k = tid; k < poly_len; k += blockDim.x) {
            uint64_t val = smem[m * poly_len + k];
            smem[m * poly_len + k] = (val == 0) ? 0 : (mod_m - val);
        }
    }
    __syncthreads();

    uint64_t* dst_neg = d_neg_monomial_ntts + (size_t)j * cpn;
    for (uint32_t k = tid; k < cpn; k += blockDim.x) {
        dst_neg[k] = smem[k];
    }
}

// ============================================================================
// Kernel 2: Prep Pack LWEs
// Converts hint_0 columns into a_ct_tilde NTT polynomials.
// Each block: one (output, column_j) pair.
// Block: 1024 threads, Grid: (poly_len, num_outputs), Smem: cpn u64
// ============================================================================

__global__ void inspir_prep_pack_lwes(
    uint64_t* __restrict__ d_a_ct_tilde,
    const uint64_t* __restrict__ d_hint_0,
    uint32_t db_cols,
    uint64_t modulus_Q,
    NTTParams params)
{
    uint32_t col_j = blockIdx.x;
    uint32_t output = blockIdx.y;
    uint32_t tid = threadIdx.x;
    uint32_t poly_len = params.poly_len;
    uint32_t crt_count = params.crt_count;
    uint32_t cpn = crt_count * poly_len;

    extern __shared__ uint64_t smem[];

    // Extract column, negacyclic_perm, CRT decompose — all in one pass.
    // negacyclic_perm(shift=0): out[0]=a[0]; out[k]=Q-a[N-k] for k>0
    uint32_t col_in_hint = output * poly_len + col_j;
    for (uint32_t k = tid; k < poly_len; k += blockDim.x) {
        uint64_t val;
        if (k == 0) {
            val = d_hint_0[col_in_hint];
        } else {
            uint64_t raw = d_hint_0[(size_t)(poly_len - k) * db_cols + col_in_hint];
            val = (raw == 0) ? 0 : (modulus_Q - raw);
        }
        for (uint32_t m = 0; m < crt_count; m++) {
            smem[m * poly_len + k] = val % params.moduli[m];
        }
    }
    __syncthreads();

    for (uint32_t m = 0; m < crt_count; m++) {
        ntt_forward_kernel_parallel(smem + m * poly_len, &params, m, tid, blockDim.x);
    }

    uint64_t* dst = d_a_ct_tilde + ((size_t)output * poly_len + col_j) * cpn;
    for (uint32_t k = tid; k < cpn; k += blockDim.x) {
        dst[k] = smem[k];
    }
}

// ============================================================================
// Kernel 3: Generate W Rotations
// Applies automorphisms to w_mask to produce w_all and w_bar_all.
// Block: 1024 threads, Grid: ceil(num_iter / ROTS_PER_BLK)
// ============================================================================

#define ROTS_PER_BLK 8

__global__ void inspir_generate_w_rotations(
    uint64_t* __restrict__ d_w_all,
    uint64_t* __restrict__ d_w_bar_all,
    const uint64_t* __restrict__ d_w_mask,
    const uint32_t* __restrict__ d_tables,
    const uint32_t* __restrict__ d_gen_pows,
    uint32_t t_exp_left,
    uint32_t poly_len,
    uint32_t crt_count,
    uint32_t num_iter)
{
    extern __shared__ uint64_t s_wmask[];
    uint32_t rot_base = blockIdx.x * ROTS_PER_BLK;
    uint32_t cpn = crt_count * poly_len;

    for (uint32_t k = 0; k < t_exp_left; k++) {
        for (uint32_t m = 0; m < crt_count; m++) {
            const uint64_t* src = d_w_mask + k * cpn + m * poly_len;
            for (uint32_t z = threadIdx.x; z < poly_len; z += blockDim.x)
                s_wmask[z] = src[z];
            __syncthreads();

            for (uint32_t r = 0; r < ROTS_PER_BLK; r++) {
                uint32_t rot = rot_base + r;
                if (rot >= num_iter) break;

                uint32_t t = d_gen_pows[rot];
                uint32_t tidx1 = (t - 1) / 2;
                uint32_t tidx2 = (2 * poly_len - t - 1) / 2;
                const uint32_t* tab1 = d_tables + (size_t)tidx1 * poly_len;
                const uint32_t* tab2 = d_tables + (size_t)tidx2 * poly_len;

                uint64_t* dst1 = d_w_all + ((size_t)rot * t_exp_left + k) * cpn + m * poly_len;
                uint64_t* dst2 = d_w_bar_all + ((size_t)rot * t_exp_left + k) * cpn + m * poly_len;

                for (uint32_t z = threadIdx.x; z < poly_len; z += blockDim.x) {
                    dst1[z] = s_wmask[__ldg(&tab1[z])];
                    dst2[z] = s_wmask[__ldg(&tab2[z])];
                }
            }
            __syncthreads();
        }
    }
}

// ============================================================================
// Kernel 4: Compute r_all (Phase 1)
//
// Each thread handles 2 coefficient positions (tid, tid+1024) across all CRT
// moduli. Accumulators stay in registers: 2 pos × 2 CRT × 2 paths = 8 regs.
// Grid: (num_to_pack_half, num_outputs), Block: 1024
// Smem: cpn u64 (for automorphism gather)
// ============================================================================

__global__ void inspir_compute_r_all(
    uint64_t* __restrict__ d_r_all,
    uint64_t* __restrict__ d_r_bar_all,
    const uint64_t* __restrict__ d_monomial_ntts,
    const uint64_t* __restrict__ d_neg_monomial_ntts,
    const uint64_t* __restrict__ d_a_ct_tilde,
    const uint64_t* __restrict__ d_mod_inv_poly,
    const uint32_t* __restrict__ d_tables,
    const uint32_t* __restrict__ d_gen_pows,
    NTTParams params,
    uint32_t q2_bits)
{
    uint32_t i = blockIdx.x;
    uint32_t output = blockIdx.y;
    uint32_t tid = threadIdx.x;
    uint32_t poly_len = params.poly_len;
    uint32_t crt_count = params.crt_count;
    uint32_t cpn = crt_count * poly_len;

    extern __shared__ uint64_t smem[];

    uint32_t gp_val = d_gen_pows[(poly_len - i) % poly_len];
    uint32_t reduction_steps = 1u << (64 - 2 * q2_bits - 1);
    if (reduction_steps < 1) reduction_steps = 1;

    const uint64_t* a_ct_base = d_a_ct_tilde + (size_t)output * poly_len * cpn;

    // Register accumulators: [crt][pos] for r and r_bar
    uint64_t acc_r[4] = {0, 0, 0, 0};   // [m0_z0, m0_z1, m1_z0, m1_z1]
    uint64_t acc_rb[4] = {0, 0, 0, 0};

    for (uint32_t j = 0; j < poly_len; j++) {
        uint32_t index = (uint32_t)(((uint64_t)j * gp_val) % (2 * poly_len));
        uint32_t index_bar = (uint32_t)((2 * poly_len
            - ((uint64_t)j * gp_val) % (2 * poly_len)) % (2 * poly_len));

        const uint64_t* mono = (index < poly_len)
            ? d_monomial_ntts + (size_t)(index % poly_len) * cpn
            : d_neg_monomial_ntts + (size_t)(index % poly_len) * cpn;
        const uint64_t* mono_bar = (index_bar < poly_len)
            ? d_monomial_ntts + (size_t)(index_bar % poly_len) * cpn
            : d_neg_monomial_ntts + (size_t)(index_bar % poly_len) * cpn;

        const uint64_t* a_ct_j = a_ct_base + (size_t)j * cpn;

        for (uint32_t m = 0; m < crt_count; m++) {
            uint32_t base = m * poly_len;
            {
                uint64_t a = __ldg(&a_ct_j[base + tid]);
                acc_r[m * 2]  += __ldg(&mono[base + tid]) * a;
                acc_rb[m * 2] += __ldg(&mono_bar[base + tid]) * a;
            }
            if (tid + 1024 < poly_len) {
                uint64_t a = __ldg(&a_ct_j[base + tid + 1024]);
                acc_r[m * 2 + 1]  += __ldg(&mono[base + tid + 1024]) * a;
                acc_rb[m * 2 + 1] += __ldg(&mono_bar[base + tid + 1024]) * a;
            }
        }

        if ((j + 1) % reduction_steps == 0) {
            for (uint32_t m = 0; m < crt_count; m++) {
                uint64_t mod_m = params.moduli[m];
                uint64_t cr = params.barrett_cr[m];
                acc_r[m*2]     = barrett_raw_u64(acc_r[m*2], cr, mod_m);
                acc_r[m*2+1]   = barrett_raw_u64(acc_r[m*2+1], cr, mod_m);
                acc_rb[m*2]    = barrett_raw_u64(acc_rb[m*2], cr, mod_m);
                acc_rb[m*2+1]  = barrett_raw_u64(acc_rb[m*2+1], cr, mod_m);
            }
        }
    }

    // Final reduction + multiply by mod_inv_poly
    for (uint32_t m = 0; m < crt_count; m++) {
        uint64_t mod_m = params.moduli[m];
        uint64_t cr = params.barrett_cr[m];
        acc_r[m*2]   = barrett_raw_u64(acc_r[m*2], cr, mod_m);
        acc_r[m*2+1] = barrett_raw_u64(acc_r[m*2+1], cr, mod_m);
        acc_rb[m*2]  = barrett_raw_u64(acc_rb[m*2], cr, mod_m);
        acc_rb[m*2+1]= barrett_raw_u64(acc_rb[m*2+1], cr, mod_m);

        uint64_t inv0 = __ldg(&d_mod_inv_poly[m * poly_len + tid]);
        acc_r[m*2]  = barrett_raw_u64(acc_r[m*2] * inv0, cr, mod_m);
        acc_rb[m*2] = barrett_raw_u64(acc_rb[m*2] * inv0, cr, mod_m);
        if (tid + 1024 < poly_len) {
            uint64_t inv1 = __ldg(&d_mod_inv_poly[m * poly_len + tid + 1024]);
            acc_r[m*2+1]  = barrett_raw_u64(acc_r[m*2+1] * inv1, cr, mod_m);
            acc_rb[m*2+1] = barrett_raw_u64(acc_rb[m*2+1] * inv1, cr, mod_m);
        }
    }

    // Automorphism for r: τ_{gen_pows[i]}
    // Write to smem, then gather via table
    for (uint32_t m = 0; m < crt_count; m++) {
        smem[m * poly_len + tid] = acc_r[m * 2];
        if (tid + 1024 < poly_len)
            smem[m * poly_len + tid + 1024] = acc_r[m * 2 + 1];
    }
    __syncthreads();

    uint32_t t_val = d_gen_pows[i];
    uint32_t tidx1 = (t_val - 1) / 2;
    const uint32_t* tab1 = d_tables + (size_t)tidx1 * poly_len;

    size_t r_off = ((size_t)output * gridDim.x + i) * cpn;
    for (uint32_t m = 0; m < crt_count; m++) {
        d_r_all[r_off + m * poly_len + tid] = smem[m * poly_len + __ldg(&tab1[tid])];
        if (tid + 1024 < poly_len)
            d_r_all[r_off + m * poly_len + tid + 1024] = smem[m * poly_len + __ldg(&tab1[tid + 1024])];
    }
    __syncthreads();

    // Automorphism for r_bar: τ_{2*poly_len - gen_pows[i]}
    for (uint32_t m = 0; m < crt_count; m++) {
        smem[m * poly_len + tid] = acc_rb[m * 2];
        if (tid + 1024 < poly_len)
            smem[m * poly_len + tid + 1024] = acc_rb[m * 2 + 1];
    }
    __syncthreads();

    uint32_t t_val2 = 2 * poly_len - t_val;
    uint32_t tidx2 = (t_val2 - 1) / 2;
    const uint32_t* tab2 = d_tables + (size_t)tidx2 * poly_len;

    for (uint32_t m = 0; m < crt_count; m++) {
        d_r_bar_all[r_off + m * poly_len + tid] = smem[m * poly_len + __ldg(&tab2[tid])];
        if (tid + 1024 < poly_len)
            d_r_bar_all[r_off + m * poly_len + tid + 1024] = smem[m * poly_len + __ldg(&tab2[tid + 1024])];
    }
}

// ============================================================================
// Kernel 5: Backward Reduction (Phase 2)
//
// Persistent kernel: each block runs num_iter sequential iterations for one output.
// Grid: num_outputs, Block: 1024
// Smem: cpn u64 (= 32 KB for poly_len=2048, crt_count=2)
//
// Key design: processes gadget digits one at a time to avoid 128 KB smem.
// After INTT + CRT compose, composed values live in registers (2 per thread).
// Each digit is extracted, NTT'd in ntt_buf, condensed+stored, and accumulated.
// ============================================================================

__global__ void inspir_backward_reduction(
    uint64_t* __restrict__ d_r_all,
    uint64_t* __restrict__ d_r_bar_all,
    uint64_t* __restrict__ d_bold_t_condensed,
    uint64_t* __restrict__ d_bold_t_bar_condensed,
    uint64_t* __restrict__ d_bold_t_hat_condensed,
    uint64_t* __restrict__ d_a_hat,
    const uint64_t* __restrict__ d_w_all,
    const uint64_t* __restrict__ d_w_bar_all,
    const uint64_t* __restrict__ d_v_mask,
    NTTParams params,
    uint32_t t_exp_left,
    uint32_t bits_per,
    uint32_t num_to_pack_half)
{
    uint32_t output = blockIdx.x;
    uint32_t tid = threadIdx.x;
    uint32_t poly_len = params.poly_len;
    uint32_t crt_count = params.crt_count;
    uint32_t cpn = crt_count * poly_len;
    uint32_t num_iter = num_to_pack_half - 1;
    uint64_t bit_mask = (1ULL << bits_per) - 1;

    extern __shared__ uint64_t smem[];
    uint64_t* ntt_buf = smem;  // cpn u64

    size_t r_base = (size_t)output * num_to_pack_half * cpn;
    size_t bt_base = (size_t)output * num_iter * t_exp_left * poly_len;

    // Two paths: p=0 is r/w/bold_t, p=1 is r_bar/w_bar/bold_t_bar
    uint64_t* d_r_ptrs[2]  = {d_r_all, d_r_bar_all};
    const uint64_t* d_w_ptrs[2]  = {d_w_all, d_w_bar_all};
    uint64_t* d_bt_ptrs[2] = {d_bold_t_condensed, d_bold_t_bar_condensed};

    // ---- Main backward loop ----
    for (int ii = (int)num_iter - 1; ii >= 0; ii--) {
        uint32_t i = (uint32_t)ii;

        for (uint32_t p = 0; p < 2; p++) {
            // Load r[i+1] into ntt_buf
            uint64_t* r_ip1 = d_r_ptrs[p] + r_base + (size_t)(i + 1) * cpn;
            for (uint32_t kk = tid; kk < cpn; kk += blockDim.x)
                ntt_buf[kk] = r_ip1[kk];
            __syncthreads();

            // INTT
            for (uint32_t m = 0; m < crt_count; m++)
                ntt_inverse_kernel_parallel(ntt_buf + m * poly_len, &params, m, tid, blockDim.x);

            // CRT compose -> registers (2 values per thread, positions tid and tid+1024)
            uint64_t comp0 = crt_compose_2(ntt_buf[tid], ntt_buf[poly_len + tid], &params);
            uint64_t comp1 = crt_compose_2(ntt_buf[tid + 1024], ntt_buf[poly_len + tid + 1024], &params);

            // Accumulators for w multiply-add (per CRT × position)
            uint64_t macc[4] = {0, 0, 0, 0};

            // Process each gadget digit one at a time
            for (uint32_t k = 0; k < t_exp_left; k++) {
                uint32_t boff = k * bits_per;
                uint64_t d0 = (boff >= 64) ? 0ULL : ((comp0 >> boff) & bit_mask);
                uint64_t d1 = (boff >= 64) ? 0ULL : ((comp1 >> boff) & bit_mask);

                // Write digit to ntt_buf (same value for both CRT slots)
                ntt_buf[tid] = d0;
                ntt_buf[tid + 1024] = d1;
                ntt_buf[poly_len + tid] = d0;
                ntt_buf[poly_len + tid + 1024] = d1;
                __syncthreads();

                // Forward NTT per CRT
                for (uint32_t m = 0; m < crt_count; m++)
                    ntt_forward_kernel_parallel(ntt_buf + m * poly_len, &params, m, tid, blockDim.x);

                // Condense & store bold_t[i][k]
                uint64_t* bt_dst = d_bt_ptrs[p] + bt_base + ((size_t)i * t_exp_left + k) * poly_len;
                bt_dst[tid] = ntt_buf[tid] | (ntt_buf[poly_len + tid] << 32);
                bt_dst[tid + 1024] = ntt_buf[tid + 1024] | (ntt_buf[poly_len + tid + 1024] << 32);

                // Accumulate w[i,k] * ntt_val
                for (uint32_t m = 0; m < crt_count; m++) {
                    size_t w_off = ((size_t)i * t_exp_left + k) * cpn + m * poly_len;
                    macc[m * 2]     += __ldg(&d_w_ptrs[p][w_off + tid])     * ntt_buf[m * poly_len + tid];
                    macc[m * 2 + 1] += __ldg(&d_w_ptrs[p][w_off + tid + 1024]) * ntt_buf[m * poly_len + tid + 1024];
                }
            }

            // Barrett reduce and add to r[i]
            for (uint32_t m = 0; m < crt_count; m++) {
                uint64_t mod_m = params.moduli[m];
                uint64_t cr = params.barrett_cr[m];
                uint64_t rv0 = barrett_raw_u64(macc[m * 2], cr, mod_m);
                uint64_t rv1 = barrett_raw_u64(macc[m * 2 + 1], cr, mod_m);

                uint64_t* r_i = d_r_ptrs[p] + r_base + (size_t)i * cpn + m * poly_len;
                uint64_t new0 = r_i[tid] + rv0;       if (new0 >= mod_m) new0 -= mod_m;
                uint64_t new1 = r_i[tid + 1024] + rv1; if (new1 >= mod_m) new1 -= mod_m;
                r_i[tid] = new0;
                r_i[tid + 1024] = new1;
            }
        }
    }

    // ---- Final: bold_t_hat from r_bar_all[0] ----
    {
        uint64_t* rb_0 = d_r_bar_all + r_base;
        for (uint32_t kk = tid; kk < cpn; kk += blockDim.x)
            ntt_buf[kk] = rb_0[kk];
        __syncthreads();

        for (uint32_t m = 0; m < crt_count; m++)
            ntt_inverse_kernel_parallel(ntt_buf + m * poly_len, &params, m, tid, blockDim.x);

        uint64_t comp0 = crt_compose_2(ntt_buf[tid], ntt_buf[poly_len + tid], &params);
        uint64_t comp1 = crt_compose_2(ntt_buf[tid + 1024], ntt_buf[poly_len + tid + 1024], &params);

        // v_mask accumulators (for updating r_all[0])
        uint64_t vacc[4] = {0, 0, 0, 0};
        size_t bt_hat_base = (size_t)output * t_exp_left * poly_len;

        for (uint32_t k = 0; k < t_exp_left; k++) {
            uint32_t boff = k * bits_per;
            uint64_t d0 = (boff >= 64) ? 0ULL : ((comp0 >> boff) & bit_mask);
            uint64_t d1 = (boff >= 64) ? 0ULL : ((comp1 >> boff) & bit_mask);

            ntt_buf[tid] = d0;
            ntt_buf[tid + 1024] = d1;
            ntt_buf[poly_len + tid] = d0;
            ntt_buf[poly_len + tid + 1024] = d1;
            __syncthreads();

            for (uint32_t m = 0; m < crt_count; m++)
                ntt_forward_kernel_parallel(ntt_buf + m * poly_len, &params, m, tid, blockDim.x);

            // Condense bold_t_hat[k]
            uint64_t* bth_dst = d_bold_t_hat_condensed + bt_hat_base + k * poly_len;
            bth_dst[tid] = ntt_buf[tid] | (ntt_buf[poly_len + tid] << 32);
            bth_dst[tid + 1024] = ntt_buf[tid + 1024] | (ntt_buf[poly_len + tid + 1024] << 32);

            // Accumulate v_mask[k] * ntt_val
            for (uint32_t m = 0; m < crt_count; m++) {
                size_t v_off = k * cpn + m * poly_len;
                vacc[m * 2]     += __ldg(&d_v_mask[v_off + tid])     * ntt_buf[m * poly_len + tid];
                vacc[m * 2 + 1] += __ldg(&d_v_mask[v_off + tid + 1024]) * ntt_buf[m * poly_len + tid + 1024];
            }
        }

        // Add v_mask accumulator to r_all[0]
        for (uint32_t m = 0; m < crt_count; m++) {
            uint64_t mod_m = params.moduli[m];
            uint64_t cr = params.barrett_cr[m];
            uint64_t rv0 = barrett_raw_u64(vacc[m * 2], cr, mod_m);
            uint64_t rv1 = barrett_raw_u64(vacc[m * 2 + 1], cr, mod_m);

            uint64_t* r_0 = d_r_all + r_base + m * poly_len;
            uint64_t new0 = r_0[tid] + rv0;       if (new0 >= mod_m) new0 -= mod_m;
            uint64_t new1 = r_0[tid + 1024] + rv1; if (new1 >= mod_m) new1 -= mod_m;
            r_0[tid] = new0;
            r_0[tid + 1024] = new1;
        }
        __syncthreads();

        // a_hat = INTT(r_all[0]) -> CRT compose
        uint64_t* r0_ptr = d_r_all + r_base;
        for (uint32_t kk = tid; kk < cpn; kk += blockDim.x)
            ntt_buf[kk] = r0_ptr[kk];
        __syncthreads();

        for (uint32_t m = 0; m < crt_count; m++)
            ntt_inverse_kernel_parallel(ntt_buf + m * poly_len, &params, m, tid, blockDim.x);

        uint64_t* a_hat_out = d_a_hat + (size_t)output * poly_len;
        a_hat_out[tid] = crt_compose_2(ntt_buf[tid], ntt_buf[poly_len + tid], &params);
        a_hat_out[tid + 1024] = crt_compose_2(ntt_buf[tid + 1024], ntt_buf[poly_len + tid + 1024], &params);
    }
}

// ============================================================================
// C API
// ============================================================================

extern "C" {

void* inspir_precomp_init(
    const uint64_t* d_hint_0,
    uint32_t db_cols,
    uint32_t poly_len,
    uint32_t crt_count,
    uint32_t t_exp_left,
    uint32_t modulus_log2,
    uint32_t q2_bits,
    uint32_t num_outputs,
    const uint64_t* moduli,
    const uint64_t* barrett_cr,
    const uint64_t* forward_table,
    const uint64_t* forward_prime_table,
    const uint64_t* inverse_table,
    const uint64_t* inverse_prime_table,
    uint64_t mod0_inv_mod1,
    uint64_t mod1_inv_mod0,
    uint64_t barrett_cr_0_modulus,
    uint64_t barrett_cr_1_modulus,
    uint64_t modulus,
    const uint64_t* w_mask,        // host [t_exp * cpn]
    const uint64_t* v_mask,        // host [t_exp * cpn]
    const uint64_t* mod_inv_poly,  // host [cpn]
    const uint32_t* tables,        // host [num_tables * poly_len]
    uint32_t num_tables,
    const uint32_t* gen_pows,      // host [poly_len]
    uint32_t gen_pows_len)
{
    InspirPrecompContext* ctx = new InspirPrecompContext();
    ctx->poly_len = poly_len;
    ctx->crt_count = crt_count;
    ctx->t_exp_left = t_exp_left;
    ctx->num_outputs = num_outputs;
    ctx->num_to_pack_half = poly_len / 2;
    ctx->num_iter = poly_len / 2 - 1;
    ctx->d_hint_0 = d_hint_0;
    ctx->db_cols = db_cols;
    ctx->bits_per = modulus_log2 / t_exp_left + 1;
    ctx->q2_bits = q2_bits;

    CUDA_ASSERT(cudaStreamCreate(&ctx->stream));

    uint32_t cpn = crt_count * poly_len;
    NTTParams& np = ctx->ntt_params;
    np.poly_len = poly_len;
    np.log2_poly_len = 0;
    for (uint32_t v = poly_len; v > 1; v >>= 1) np.log2_poly_len++;
    np.crt_count = crt_count;
    np.mod0_inv_mod1 = mod0_inv_mod1;
    np.mod1_inv_mod0 = mod1_inv_mod0;
    np.barrett_cr_0_modulus = barrett_cr_0_modulus;
    np.barrett_cr_1_modulus = barrett_cr_1_modulus;
    np.modulus = modulus;

    CUDA_ALLOC_AND_COPY(np.moduli, moduli, crt_count * sizeof(uint64_t));
    CUDA_ALLOC_AND_COPY(np.barrett_cr, barrett_cr, crt_count * sizeof(uint64_t));
    CUDA_ALLOC_AND_COPY(np.forward_table, forward_table, cpn * sizeof(uint64_t));
    CUDA_ALLOC_AND_COPY(np.forward_prime_table, forward_prime_table, cpn * sizeof(uint64_t));
    CUDA_ALLOC_AND_COPY(np.inverse_table, inverse_table, cpn * sizeof(uint64_t));
    CUDA_ALLOC_AND_COPY(np.inverse_prime_table, inverse_prime_table, cpn * sizeof(uint64_t));

    CUDA_ALLOC_AND_COPY(ctx->d_tables, tables, (size_t)num_tables * poly_len * sizeof(uint32_t));
    ctx->num_tables = num_tables;
    CUDA_ALLOC_AND_COPY(ctx->d_gen_pows, gen_pows, gen_pows_len * sizeof(uint32_t));

    size_t mask_size = (size_t)t_exp_left * cpn * sizeof(uint64_t);
    uint64_t* d_w_mask;
    CUDA_ALLOC_AND_COPY(d_w_mask, w_mask, mask_size);
    CUDA_ALLOC_AND_COPY(ctx->d_v_mask, v_mask, mask_size);
    CUDA_ALLOC_AND_COPY(ctx->d_mod_inv_poly, mod_inv_poly, cpn * sizeof(uint64_t));

    size_t mono_size = (size_t)poly_len * cpn * sizeof(uint64_t);
    CUDA_ASSERT(cudaMalloc(&ctx->d_monomial_ntts, mono_size));
    CUDA_ASSERT(cudaMalloc(&ctx->d_neg_monomial_ntts, mono_size));

    CUDA_ASSERT(cudaMalloc(&ctx->d_a_ct_tilde,
        (size_t)num_outputs * poly_len * cpn * sizeof(uint64_t)));

    size_t w_size = (size_t)ctx->num_iter * t_exp_left * cpn * sizeof(uint64_t);
    CUDA_ASSERT(cudaMalloc(&ctx->d_w_all, w_size));
    CUDA_ASSERT(cudaMalloc(&ctx->d_w_bar_all, w_size));

    size_t r_size = (size_t)num_outputs * ctx->num_to_pack_half * cpn * sizeof(uint64_t);
    CUDA_ASSERT(cudaMalloc(&ctx->d_r_all, r_size));
    CUDA_ASSERT(cudaMalloc(&ctx->d_r_bar_all, r_size));

    size_t bt_size = (size_t)num_outputs * ctx->num_iter * t_exp_left * poly_len * sizeof(uint64_t);
    CUDA_ASSERT(cudaMalloc(&ctx->d_bold_t_condensed, bt_size));
    CUDA_ASSERT(cudaMalloc(&ctx->d_bold_t_bar_condensed, bt_size));

    size_t bth_size = (size_t)num_outputs * t_exp_left * poly_len * sizeof(uint64_t);
    CUDA_ASSERT(cudaMalloc(&ctx->d_bold_t_hat_condensed, bth_size));

    size_t ah_size = (size_t)num_outputs * poly_len * sizeof(uint64_t);
    CUDA_ASSERT(cudaMalloc(&ctx->d_a_hat, ah_size));

    printf("InspiRING GPU precomp init: poly_len=%u, crt=%u, t_exp=%u, bits_per=%u, outputs=%u\n",
           poly_len, crt_count, t_exp_left, ctx->bits_per, num_outputs);
    printf("  monomials=%.1f MB, a_ct=%.1f MB, w_all=%.1f MB, r_all=%.1f MB, bold_t=%.1f MB\n",
           2 * mono_size / 1e6,
           (size_t)num_outputs * poly_len * cpn * 8 / 1e6,
           2 * w_size / 1e6, 2 * r_size / 1e6, 2 * bt_size / 1e6);

    // Run Kernel 1: monomial NTTs
    {
        size_t smem = cpn * sizeof(uint64_t);
        inspir_compute_monomials<<<poly_len, 1024, smem, ctx->stream>>>(
            ctx->d_monomial_ntts, ctx->d_neg_monomial_ntts, ctx->ntt_params);
        CUDA_ASSERT(cudaGetLastError());
    }

    // Run Kernel 3: w rotations
    {
        uint32_t num_blocks = (ctx->num_iter + ROTS_PER_BLK - 1) / ROTS_PER_BLK;
        size_t smem = poly_len * sizeof(uint64_t);
        inspir_generate_w_rotations<<<num_blocks, 1024, smem, ctx->stream>>>(
            ctx->d_w_all, ctx->d_w_bar_all, d_w_mask,
            ctx->d_tables, ctx->d_gen_pows,
            t_exp_left, poly_len, crt_count, ctx->num_iter);
        CUDA_ASSERT(cudaGetLastError());
    }

    CUDA_ASSERT(cudaFree(d_w_mask));

    return ctx;
}

void inspir_precomp_compute(void* context)
{
    InspirPrecompContext* ctx = (InspirPrecompContext*)context;
    if (!ctx) return;

    uint32_t poly_len = ctx->poly_len;
    uint32_t crt_count = ctx->crt_count;
    uint32_t cpn = crt_count * poly_len;
    uint32_t t_exp_left = ctx->t_exp_left;

    GpuTimer timer;

    // Kernel 2: prep pack LWEs
    timer.tic(ctx->stream);
    {
        size_t smem = cpn * sizeof(uint64_t);
        dim3 grid(poly_len, ctx->num_outputs);
        inspir_prep_pack_lwes<<<grid, 1024, smem, ctx->stream>>>(
            ctx->d_a_ct_tilde, ctx->d_hint_0,
            ctx->db_cols, ctx->ntt_params.modulus, ctx->ntt_params);
        CUDA_ASSERT(cudaGetLastError());
    }
    float prep_ms = timer.toc_ms(ctx->stream);
    printf("  Prep pack LWEs: %.2f ms\n", prep_ms);

    // Kernel 4: compute r_all (Phase 1)
    timer.tic(ctx->stream);
    {
        dim3 grid(ctx->num_to_pack_half, ctx->num_outputs);
        size_t smem = cpn * sizeof(uint64_t);
        inspir_compute_r_all<<<grid, 1024, smem, ctx->stream>>>(
            ctx->d_r_all, ctx->d_r_bar_all,
            ctx->d_monomial_ntts, ctx->d_neg_monomial_ntts,
            ctx->d_a_ct_tilde, ctx->d_mod_inv_poly,
            ctx->d_tables, ctx->d_gen_pows,
            ctx->ntt_params, ctx->q2_bits);
        CUDA_ASSERT(cudaGetLastError());
    }
    float phase1_ms = timer.toc_ms(ctx->stream);
    printf("  Phase 1 (compute r_all): %.2f ms\n", phase1_ms);

    // Free monomials and a_ct_tilde
    CUDA_ASSERT(cudaFree(ctx->d_monomial_ntts));  ctx->d_monomial_ntts = nullptr;
    CUDA_ASSERT(cudaFree(ctx->d_neg_monomial_ntts)); ctx->d_neg_monomial_ntts = nullptr;
    CUDA_ASSERT(cudaFree(ctx->d_a_ct_tilde)); ctx->d_a_ct_tilde = nullptr;

    // Kernel 5: backward reduction (Phase 2)
    timer.tic(ctx->stream);
    {
        size_t smem = (size_t)cpn * sizeof(uint64_t);  // 32 KB for poly_len=2048

        inspir_backward_reduction<<<ctx->num_outputs, 1024, smem, ctx->stream>>>(
            ctx->d_r_all, ctx->d_r_bar_all,
            ctx->d_bold_t_condensed, ctx->d_bold_t_bar_condensed,
            ctx->d_bold_t_hat_condensed, ctx->d_a_hat,
            ctx->d_w_all, ctx->d_w_bar_all,
            ctx->d_v_mask, ctx->ntt_params,
            t_exp_left, ctx->bits_per, ctx->num_to_pack_half);
        CUDA_ASSERT(cudaGetLastError());
    }
    float phase2_ms = timer.toc_ms(ctx->stream);
    printf("  Phase 2 (backward reduction): %.2f ms\n", phase2_ms);

    // Free intermediates
    CUDA_ASSERT(cudaFree(ctx->d_r_all)); ctx->d_r_all = nullptr;
    CUDA_ASSERT(cudaFree(ctx->d_r_bar_all)); ctx->d_r_bar_all = nullptr;
    CUDA_ASSERT(cudaFree(ctx->d_w_all)); ctx->d_w_all = nullptr;
    CUDA_ASSERT(cudaFree(ctx->d_w_bar_all)); ctx->d_w_bar_all = nullptr;
    CUDA_ASSERT(cudaFree(ctx->d_mod_inv_poly)); ctx->d_mod_inv_poly = nullptr;
    CUDA_ASSERT(cudaFree(ctx->d_v_mask)); ctx->d_v_mask = nullptr;

    printf("InspiRING GPU precomp: prep=%.2f, phase1=%.2f, phase2=%.2f, total=%.2f ms\n",
           prep_ms, phase1_ms, phase2_ms, prep_ms + phase1_ms + phase2_ms);
}

void inspir_precomp_get_results(
    void* context,
    uint64_t** out_bold_t_condensed,
    uint64_t** out_bold_t_bar_condensed,
    uint64_t** out_bold_t_hat_condensed,
    uint64_t** out_a_hat,
    size_t* out_bold_t_size,
    size_t* out_bold_t_bar_size,
    size_t* out_bold_t_hat_size,
    size_t* out_a_hat_size)
{
    InspirPrecompContext* ctx = (InspirPrecompContext*)context;
    *out_bold_t_condensed = ctx->d_bold_t_condensed;
    *out_bold_t_bar_condensed = ctx->d_bold_t_bar_condensed;
    *out_bold_t_hat_condensed = ctx->d_bold_t_hat_condensed;
    *out_a_hat = ctx->d_a_hat;

    uint32_t poly_len = ctx->poly_len;
    uint32_t t_exp = ctx->t_exp_left;
    uint32_t ni = ctx->num_iter;
    uint32_t no = ctx->num_outputs;

    *out_bold_t_size = (size_t)no * ni * t_exp * poly_len * sizeof(uint64_t);
    *out_bold_t_bar_size = *out_bold_t_size;
    *out_bold_t_hat_size = (size_t)no * t_exp * poly_len * sizeof(uint64_t);
    *out_a_hat_size = (size_t)no * poly_len * sizeof(uint64_t);
}

void inspir_precomp_free(void* context, bool free_outputs)
{
    InspirPrecompContext* ctx = (InspirPrecompContext*)context;
    if (!ctx) return;

    if (ctx->ntt_params.moduli) cudaFree(ctx->ntt_params.moduli);
    if (ctx->ntt_params.barrett_cr) cudaFree(ctx->ntt_params.barrett_cr);
    if (ctx->ntt_params.forward_table) cudaFree(ctx->ntt_params.forward_table);
    if (ctx->ntt_params.forward_prime_table) cudaFree(ctx->ntt_params.forward_prime_table);
    if (ctx->ntt_params.inverse_table) cudaFree(ctx->ntt_params.inverse_table);
    if (ctx->ntt_params.inverse_prime_table) cudaFree(ctx->ntt_params.inverse_prime_table);
    if (ctx->d_tables) cudaFree(ctx->d_tables);
    if (ctx->d_gen_pows) cudaFree(ctx->d_gen_pows);
    if (ctx->d_monomial_ntts) cudaFree(ctx->d_monomial_ntts);
    if (ctx->d_neg_monomial_ntts) cudaFree(ctx->d_neg_monomial_ntts);
    if (ctx->d_a_ct_tilde) cudaFree(ctx->d_a_ct_tilde);
    if (ctx->d_w_all) cudaFree(ctx->d_w_all);
    if (ctx->d_w_bar_all) cudaFree(ctx->d_w_bar_all);
    if (ctx->d_r_all) cudaFree(ctx->d_r_all);
    if (ctx->d_r_bar_all) cudaFree(ctx->d_r_bar_all);
    if (ctx->d_mod_inv_poly) cudaFree(ctx->d_mod_inv_poly);
    if (ctx->d_v_mask) cudaFree(ctx->d_v_mask);

    if (free_outputs) {
        if (ctx->d_bold_t_condensed) cudaFree(ctx->d_bold_t_condensed);
        if (ctx->d_bold_t_bar_condensed) cudaFree(ctx->d_bold_t_bar_condensed);
        if (ctx->d_bold_t_hat_condensed) cudaFree(ctx->d_bold_t_hat_condensed);
        if (ctx->d_a_hat) cudaFree(ctx->d_a_hat);
    }

    if (ctx->stream) cudaStreamDestroy(ctx->stream);
    delete ctx;
}

} // extern "C"
