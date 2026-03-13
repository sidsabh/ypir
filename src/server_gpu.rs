#![cfg(feature = "cuda")]

use std::time::Instant;

use log::debug;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use spiral_rs::{arith::*, poly::*};

use crate::measurement::Measurement;
use crate::params::GetQPrime;

use super::{
    client::*,
    convolution::{negacyclic_perm_u32, Convolution},
    lwe::*,
    packing::*,
    scheme::*,
    server::*,
};

impl<'a, T> YServer<'a, T>
where
    T: Sized + Copy + ToU64 + Default + Sync,
    *const T: ToM512,
{
    pub fn init_hint_0_gpu_context(&self, lwe_params: &LWEParams, conv: &Convolution) -> crate::cuda::OfflineComputeContext {
        use crate::cuda::OfflineComputeContext;

        let db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_cols = self.db_cols();
        let n = lwe_params.n;

        // Prepare v_nega_perm_a
        let mut rng_pub = ChaCha20Rng::from_seed(get_seed(SEED_0));
        let mut v_nega_perm_a_flat = Vec::new();
        for _ in 0..db_rows / n {
            let mut a = vec![0u32; n];
            for idx in 0..n {
                a[idx] = rng_pub.sample::<u32, _>(rand::distributions::Standard);
            }
            let nega_perm_a = negacyclic_perm_u32(&a);
            let nega_perm_a_ntt = conv.ntt(&nega_perm_a);
            v_nega_perm_a_flat.extend_from_slice(&nega_perm_a_ntt);
        }

        // Compute max_adds
        let log2_conv_output =
            log2(lwe_params.modulus) + log2(lwe_params.n as u64) + log2(lwe_params.pt_modulus);
        let log2_modulus = log2(conv.params().modulus);
        let log2_max_adds = log2_modulus - log2_conv_output - 1;
        let max_adds = 1 << log2_max_adds;

        // Convert DB to u8 slice
        let db_slice = self.db();
        let db_u8 = unsafe {
            std::slice::from_raw_parts(
                db_slice.as_ptr() as *const u8,
                db_slice.len() * std::mem::size_of::<T>(),
            )
        };

        // Extract NTT tables
        let crt_count = conv.params().crt_count;
        let poly_len = conv.params().poly_len;
        let mut moduli = Vec::new();
        let mut barrett_cr = Vec::new();
        let mut forward_table = Vec::new();
        let mut forward_prime_table = Vec::new();
        let mut inverse_table = Vec::new();
        let mut inverse_prime_table = Vec::new();

        for i in 0..crt_count {
            moduli.push(conv.params().moduli[i]);
            barrett_cr.push(conv.params().barrett_cr_1[i]);
            forward_table.extend_from_slice(conv.params().get_ntt_forward_table(i));
            forward_prime_table.extend_from_slice(conv.params().get_ntt_forward_prime_table(i));
            inverse_table.extend_from_slice(conv.params().get_ntt_inverse_table(i));
            inverse_prime_table.extend_from_slice(conv.params().get_ntt_inverse_prime_table(i));
        }

        // Initialize GPUContext (DB upload happens here)
        OfflineComputeContext::new(
            db_rows as u32,
            self.db_rows_padded() as u32,
            db_cols as u32,
            n as u32,
            poly_len as u32,
            crt_count as u32,
            max_adds as u32,
            db_u8,
            &v_nega_perm_a_flat,
            &moduli,
            &barrett_cr,
            &forward_table,
            &forward_prime_table,
            &inverse_table,
            &inverse_prime_table,
            conv.params().mod0_inv_mod1,
            conv.params().mod1_inv_mod0,
            conv.params().barrett_cr_0_modulus,
            conv.params().barrett_cr_1_modulus,
        ).expect("Failed to initialize GPU context")
    }

    pub fn compute_hint_0_with_context(&self, gpu_ctx: &crate::cuda::OfflineComputeContext) -> Vec<u64> {
        gpu_ctx.compute_hint_0().expect("GPU computation failed")
    }

    /// Initialize Toeplitz GPU context for matrix-based multiplication
    /// Uses coefficient-form polynomials instead of NTT
    #[cfg(feature = "toeplitz")]
    pub fn init_toeplitz_context(&self, lwe_params: &LWEParams, conv: &Convolution) -> crate::cuda::ToeplitzContext {
        let db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_cols = self.db_cols();
        let n = lwe_params.n;

        // Prepare database as u8
        let db_u8: Vec<u8> = self
            .db()
            .iter()
            .map(|&x| x.to_u64() as u8)
            .collect();

        // Compute max_adds
        let log2_conv_output =
            log2(lwe_params.modulus) + log2(lwe_params.n as u64) + log2(lwe_params.pt_modulus);
        let log2_modulus = log2(conv.params().modulus);
        let log2_max_adds = log2_modulus - log2_conv_output - 1;
        let max_adds = 1 << log2_max_adds;

        // Generate v_nega_perm_a in COEFFICIENT FORM (not NTT'd)
        let mut rng_pub = ChaCha20Rng::from_seed(get_seed(SEED_0));
        let mut v_nega_perm_a_flat = Vec::new();

        for _ in 0..db_rows / n {
            let mut a = vec![0u32; n];
            for idx in 0..n {
                a[idx] = rng_pub.sample::<u32, _>(rand::distributions::Standard);
            }

            // For Toeplitz: apply negacyclic permutation to match CPU reference
            let nega_perm_a = negacyclic_perm_u32(&a);
            v_nega_perm_a_flat.extend_from_slice(&nega_perm_a);
        }

        // Extract NTT tables and parameters
        let crt_count = conv.params().crt_count;
        let mut moduli = Vec::new();
        let mut barrett_cr = Vec::new();

        for i in 0..crt_count {
            moduli.push(conv.params().moduli[i]);
            barrett_cr.push(conv.params().barrett_cr_1[i]);
        }

        // Initialize ToeplitzContext (note: poly_len is passed as crt_count param, but ignored in Toeplitz)
        crate::cuda::ToeplitzContext::new(
            db_rows as u32,
            self.db_rows_padded() as u32,
            db_cols as u32,
            n as u32,
            crt_count as u32,
            max_adds as u32,
            &db_u8,
            &v_nega_perm_a_flat,
            &moduli,
            &barrett_cr,
            conv.params().mod0_inv_mod1,
            conv.params().mod1_inv_mod0,
            conv.params().barrett_cr_0_modulus,
            conv.params().barrett_cr_1_modulus,
        ).expect("Failed to initialize Toeplitz GPU context")
    }

    #[cfg(feature = "toeplitz")]
    pub fn compute_hint_0_with_toeplitz(&self, toeplitz_ctx: &crate::cuda::ToeplitzContext) -> Vec<u64> {
        toeplitz_ctx.compute_hint_0().expect("Toeplitz GPU computation failed")
    }

    pub fn init_hint_0_simplepir_gpu_context(&self) -> crate::cuda::SPOfflineContext {
        let db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_cols = self.db_cols();

        // Generate pseudorandom query and convert to NTT form
        let preprocessed_query = self.generate_pseudorandom_query(SEED_0);

        // Flatten query_ntt: db_rows_poly x crt_count x poly_len
        let db_rows_poly = db_rows / self.params.poly_len;
        let mut query_ntt_flat = Vec::with_capacity(db_rows_poly * self.params.crt_count * self.params.poly_len);
        for query_poly in &preprocessed_query {
            // Each query_poly is a 1x1 PolyMatrixNTT with crt_count * poly_len elements
            query_ntt_flat.extend_from_slice(query_poly.as_slice());
        }

        // Get DB as u8 slice
        let db_u16: &[u16] = unsafe {
            std::slice::from_raw_parts(
                self.db().as_ptr() as *const u16,
                self.db().len() * std::mem::size_of::<T>(),
            )
        };

        crate::cuda::SPOfflineContext::new(
            db_u16,
            db_rows,
            self.db_rows_padded(),
            db_cols,
            &query_ntt_flat,
            self.params,
        ).expect("Failed to initialize SimplePIR GPU context")
    }

    /// Compute hint_0 using GPU for SimplePIR
    pub fn compute_hint_0_simplepir_gpu(&self, gpu_ctx: &crate::cuda::SPOfflineContext) -> Vec<u64> {
        gpu_ctx.compute_hint_0().expect("SimplePIR GPU hint_0 computation failed")
    }

    pub fn perform_online_computation_simplepir(
        &self,
        first_dim_queries_packed: &[&[u64]],
        offline_vals: &OfflinePrecomputedValues<'a>,
        packing_keys: &mut [PackingKeys<'a>],
        mut measurement: Option<&mut Measurement>,
    ) -> Vec<Vec<Vec<u8>>> {
        assert!(self.ypir_params.is_simplepir);
        let params = self.params;
        let batch_size = first_dim_queries_packed.len();

        let online_start = Instant::now();

        let ctx = offline_vals.sp_cuda_context.as_ref()
            .expect("CUDA context not initialized");

        // Flatten queries: batch_size * db_rows_padded
        let mut queries_flat: Vec<u64> = Vec::with_capacity(batch_size * self.db_rows_padded());
        for query in first_dim_queries_packed {
            queries_flat.extend_from_slice(query);
        }

        // Flatten CDKS pack_pub_params_row_1s
        let mut pub_params_flat: Vec<u64> = Vec::new();
        for pk in packing_keys.iter() {
            for key in pk.pack_pub_params_row_1s.iter() {
                let slc = key.as_slice();
                for col in 0..params.t_exp_left {
                    let start = col * 2 * params.poly_len;
                    pub_params_flat.extend_from_slice(&slc[start..start + params.poly_len]);
                }
            }
        }

        let q_1_bits = (params.get_q_prime_2() as f64).log2().ceil() as usize;
        let q_2_bits = (params.get_q_prime_1() as f64).log2().ceil() as usize;
        let response_bytes_per_output = ((q_1_bits + q_2_bits) * params.poly_len + 7) / 8;

        // Run GPU computation
        let result = ctx.compute_batch(&queries_flat, &pub_params_flat, response_bytes_per_output, batch_size);

        let online_time_ms = online_start.elapsed().as_millis();
        debug!("SimplePIR GPU online time: {} ms", online_time_ms);

        if let Some(ref mut m) = measurement {
            m.online.first_pass_time_ms = online_time_ms as usize;
        }

        result
    }

    /// Online computation for word-based SimplePIR (GPU path).
    /// CDKS: GEMM + packing all on GPU.
    /// InspiRING: GEMM on GPU, packing on CPU.
    pub fn perform_online_computation_simplepir_word(
        &self,
        word_queries: &[&[u64]],
        offline_vals: &OfflinePrecomputedValues<'a>,
        packing_keys: &mut [PackingKeys<'a>],
        mut measurement: Option<&mut Measurement>,
    ) -> Vec<Vec<Vec<u8>>> {
        assert!(self.ypir_params.is_simplepir);
        let params = self.params;
        let batch_size = word_queries.len();

        let ctx = offline_vals.word_cuda_context.as_ref()
            .expect("Word CUDA context not initialized");

        // Flatten queries: batch_size * db_rows
        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let mut queries_flat: Vec<u64> = Vec::with_capacity(batch_size * db_rows);
        for query in word_queries {
            queries_flat.extend_from_slice(query);
        }

        if offline_vals.packing_type == PackingType::InspiRING {
            // InspiRING: GEMM + packing all on GPU
            let online_start = Instant::now();

            let q_1_bits = (params.get_q_prime_2() as f64).log2().ceil() as usize;
            let q_2_bits = (params.get_q_prime_1() as f64).log2().ceil() as usize;
            let response_bytes_per_output = ((q_1_bits + q_2_bits) * params.poly_len + 7) / 8;

            // Flatten y_body and z_body (tiny, ~48KB each per client)
            // GPU will expand y_body → y_all + y_bar_all via permutation tables
            let y_body_per_client = params.t_exp_left * params.poly_len;
            let z_body_per_client = params.t_exp_left * params.poly_len;

            let mut y_body_flat: Vec<u64> = Vec::with_capacity(batch_size * y_body_per_client);
            let mut z_body_flat: Vec<u64> = Vec::with_capacity(batch_size * z_body_per_client);

            for pk in packing_keys.iter() {
                let y_body = pk.y_body_condensed.as_ref().unwrap();
                let z_body = pk.z_body_condensed.as_ref().unwrap();

                for row in 0..y_body.rows {
                    for col in 0..y_body.cols {
                        y_body_flat.extend_from_slice(&y_body.get_poly(row, col)[..params.poly_len]);
                    }
                }
                for row in 0..z_body.rows {
                    for col in 0..z_body.cols {
                        z_body_flat.extend_from_slice(&z_body.get_poly(row, col)[..params.poly_len]);
                    }
                }
            }

            let result = ctx.compute_batch_inspir(
                &queries_flat,
                &y_body_flat,
                &z_body_flat,
                response_bytes_per_output,
                batch_size,
            );

            let online_time_ms = online_start.elapsed().as_millis();
            debug!("Word GPU InspiRING online time: {} ms", online_time_ms);

            if let Some(ref mut m) = measurement {
                m.online.first_pass_time_ms = online_time_ms as usize;
            }

            result
        } else {
            // CDKS: full GPU path (matmul + packing on GPU)
            let online_start = Instant::now();

            // Flatten CDKS pack_pub_params_row_1s
            let mut pub_params_flat: Vec<u64> = Vec::new();
            for pk in packing_keys.iter() {
                for key in pk.pack_pub_params_row_1s.iter() {
                    let slc = key.as_slice();
                    for col in 0..params.t_exp_left {
                        let start = col * 2 * params.poly_len;
                        pub_params_flat.extend_from_slice(&slc[start..start + params.poly_len]);
                    }
                }
            }

            let q_1_bits = (params.get_q_prime_2() as f64).log2().ceil() as usize;
            let q_2_bits = (params.get_q_prime_1() as f64).log2().ceil() as usize;
            let response_bytes_per_output = ((q_1_bits + q_2_bits) * params.poly_len + 7) / 8;

            let result = ctx.compute_batch(&queries_flat, &pub_params_flat, response_bytes_per_output, batch_size);

            let online_time_ms = online_start.elapsed().as_millis();
            debug!("Word GPU online time: {} ms", online_time_ms);

            if let Some(ref mut m) = measurement {
                m.online.first_pass_time_ms = online_time_ms as usize;
            }

            result
        }
    }

    /// CUDA-accelerated online computation
    pub fn perform_online_computation<const K: usize>(
        &self,
        offline_vals: &mut OfflinePrecomputedValues<'a>,
        first_dim_queries_packed: &[u32],
        second_dim_query_cols: &[&[u64]],
        packing_keys: &mut [PackingKeys<'a>],
        _measurement: Option<&mut Measurement>,
    ) -> Vec<Vec<Vec<u8>>> {
        debug!("CUDA-accelerated online computation");
        // Set up parameters (same as CPU version)
        let params = self.params;
        let db_cols = self.db_cols();

        // RLWE reduced moduli
        let rlwe_q_prime_1 = params.get_q_prime_1();
        let rlwe_q_prime_2 = params.get_q_prime_2();


        let mut query_q2_batch = Vec::with_capacity(second_dim_query_cols.len() * db_cols);
        let mut pack_pub_params_row_1s_batch: Vec<u64> = Vec::with_capacity(second_dim_query_cols.len() * params.poly_len_log2 * params.t_exp_left);

        for (i, packed_query_col) in second_dim_query_cols.iter().enumerate() {
            query_q2_batch.extend_from_slice(packed_query_col);
            let automorph_keys = &packing_keys[i].pack_pub_params_row_1s;
            assert_eq!(automorph_keys.len(), params.poly_len_log2);
            for key in automorph_keys {
                let slc = key.as_slice();
                for col in 0..params.t_exp_left {
                    let start = col * 2 * params.poly_len;
                    pack_pub_params_row_1s_batch.extend_from_slice(&slc[start..start + params.poly_len]);
                }
            }
        }

        let batch_size = second_dim_query_cols.len();

        let q_1_bits = (rlwe_q_prime_1 as f64).log2().ceil() as usize;
        let q_2_bits = (rlwe_q_prime_2 as f64).log2().ceil() as usize;
        let total_sz_bits = (q_1_bits + q_2_bits) * self.params.poly_len;
        let total_sz_bytes = (total_sz_bits + 7) / 8;
        let mut responses = vec![0u8; batch_size*total_sz_bytes];

        // Begin online computation
        let online_phase = Instant::now();
        if let Some(ref ctx) = offline_vals.cuda_context {
            ctx.compute_full_batch(
                first_dim_queries_packed,
                &query_q2_batch,
                &pack_pub_params_row_1s_batch,
                batch_size,
                &mut responses
            );
        }

        debug!(
            "Total online time: {} us",
            online_phase.elapsed().as_micros()
        );
        debug!("");

        responses
            .chunks(total_sz_bytes)
            .map(|a| vec![a.to_vec()])
            .collect()
    }
}
