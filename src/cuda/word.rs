/// CUDA FFI bindings for Word-based SimplePIR offline and online computation.

use super::flatten_ntt_tables;

// ==================== Word Offline ====================

#[cfg(feature = "cuda")]
extern "C" {
    fn init_word_offline_context(
        db: *const u16,
        A: *const u64,
        db_rows: u32,
        db_rows_padded: u32,
        db_cols: u32,
        poly_len: u32,
        modulus: u64,
        mod0: u64,
        mod1: u64,
        inv_n: u64,
    ) -> *mut std::ffi::c_void;

    fn compute_hint_0_word_gpu(context: *mut std::ffi::c_void, hint_0_out: *mut u64) -> i32;

    fn ypir_word_offline_free_gemm_buffers(context: *mut std::ffi::c_void);
    fn ypir_word_offline_get_hint_device_ptr(context: *mut std::ffi::c_void) -> *mut u64;
    fn ypir_word_offline_take_hint_device_ptr(context: *mut std::ffi::c_void) -> *mut u64;
    fn ypir_word_offline_take_db_device_ptrs(
        context: *mut std::ffi::c_void,
        out_db_stacked: *mut *mut u8,
        out_db_u16: *mut *mut u16,
    );

    fn free_word_offline_context(context: *mut std::ffi::c_void);
}

#[cfg(feature = "cuda")]
pub struct WordOfflineContext {
    ctx: *mut std::ffi::c_void,
    poly_len: usize,
    db_cols: usize,
}
#[cfg(feature = "cuda")]
unsafe impl Send for WordOfflineContext {}
#[cfg(feature = "cuda")]
unsafe impl Sync for WordOfflineContext {}

#[cfg(feature = "cuda")]
impl WordOfflineContext {
    pub fn new(
        db: &[u16],
        a_matrix: &[u64],
        db_rows: u32,
        db_rows_padded: u32,
        db_cols: u32,
        poly_len: u32,
        modulus: u64,
        mod0: u64,
        mod1: u64,
        apply_inv_n: bool,
    ) -> Result<Self, String> {
        let inv_n = if apply_inv_n {
            spiral_rs::number_theory::invert_uint_mod(poly_len as u64, modulus)
                .expect("Failed to compute inv_N mod Q")
        } else {
            1u64
        };
        let ctx = unsafe {
            init_word_offline_context(
                db.as_ptr(),
                a_matrix.as_ptr(),
                db_rows,
                db_rows_padded,
                db_cols,
                poly_len,
                modulus,
                mod0,
                mod1,
                inv_n,
            )
        };
        if ctx.is_null() {
            Err("Failed to initialize Word offline GPU context".to_string())
        } else {
            Ok(Self { ctx, poly_len: poly_len as usize, db_cols: db_cols as usize })
        }
    }

    pub fn compute_hint_0(&self) -> Result<Vec<u64>, String> {
        let mut hint_0 = vec![0u64; self.poly_len * self.db_cols];
        let result = unsafe { compute_hint_0_word_gpu(self.ctx, hint_0.as_mut_ptr()) };
        if result == 0 {
            Ok(hint_0)
        } else {
            Err("Word GPU hint_0 computation failed".to_string())
        }
    }

    /// Free GEMM-specific buffers that are no longer needed after compute_hint_0().
    /// Keeps DB and hint_0 alive.
    pub fn free_gemm_buffers(&self) {
        unsafe { ypir_word_offline_free_gemm_buffers(self.ctx) }
    }

    pub fn get_hint_device_ptr(&self) -> *mut u64 {
        unsafe { ypir_word_offline_get_hint_device_ptr(self.ctx) }
    }

    /// Take ownership of the device pointer to hint_0.
    /// After this call, the offline context no longer owns d_out.
    pub fn take_hint_device_ptr(&self) -> *mut u64 {
        unsafe { ypir_word_offline_take_hint_device_ptr(self.ctx) }
    }

    /// Take ownership of DB device pointers (d_db_stacked, d_db_u16).
    pub fn take_db_device_ptrs(&self) -> (*mut u8, *mut u16) {
        let mut stacked: *mut u8 = std::ptr::null_mut();
        let mut u16_ptr: *mut u16 = std::ptr::null_mut();
        unsafe {
            ypir_word_offline_take_db_device_ptrs(self.ctx, &mut stacked, &mut u16_ptr);
        }
        (stacked, u16_ptr)
    }
}

#[cfg(feature = "cuda")]
impl Drop for WordOfflineContext {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { free_word_offline_context(self.ctx); }
        }
    }
}

// ==================== Word Online ====================

#[cfg(feature = "cuda")]
extern "C" {
    fn ypir_word_online_init(
        db: *const u16,
        db_rows: usize,
        db_rows_padded: usize,
        db_cols: usize,
        t_exp_left: usize,
        rlwe_q_prime_1: u64,
        rlwe_q_prime_2: u64,
        poly_len: u32,
        crt_count: u32,
        moduli: *const u64,
        barrett_cr: *const u64,
        forward_table: *const u64,
        forward_prime_table: *const u64,
        inverse_table: *const u64,
        inverse_prime_table: *const u64,
        mod0_inv_mod1: u64,
        mod1_inv_mod0: u64,
        barrett_cr_0_modulus: u64,
        barrett_cr_1_modulus: u64,
        max_batch_size: usize,
        inv_n: u64,
    ) -> *mut std::ffi::c_void;

    fn ypir_word_online_init_packing(
        context: *mut std::ffi::c_void,
        y_constants: *const u64, y_constants_size: usize,
        precomp_res: *const u64, precomp_res_size: usize,
        precomp_vals: *const u64, precomp_vals_size: usize,
        precomp_tables: *const u64, precomp_tables_size: usize,
    );

    fn ypir_word_online_compute_batch(
        context: *mut std::ffi::c_void,
        queries: *const u64,
        pack_pub_params_row_1s: *const u64,
        response_out: *mut u8,
        response_bytes_per_batch: usize,
        batch_size: usize,
    );

    fn ypir_word_online_compute_matmul_only(
        context: *mut std::ffi::c_void,
        queries: *const u64,
        intermediate_out: *mut u64,
        batch_size: usize,
    );

    fn ypir_word_online_init_packing_inspir(
        context: *mut std::ffi::c_void,
        bold_t_condensed: *const u64, bold_t_size: usize,
        bold_t_bar_condensed: *const u64, bold_t_bar_size: usize,
        bold_t_hat_condensed: *const u64, bold_t_hat_size: usize,
        a_hat: *const u64, a_hat_size: usize,
        num_iter: usize,
        tables: *const u32, num_tables: usize,
        gen_pows: *const u32, num_rotations: usize,
    );

    fn ypir_word_online_init_packing_inspir_from_gpu(
        context: *mut std::ffi::c_void,
        d_bold_t_condensed: *mut u64,
        d_bold_t_bar_condensed: *mut u64,
        d_bold_t_hat_condensed: *mut u64,
        d_a_hat: *mut u64,
        num_iter: usize,
        tables: *const u32, num_tables: usize,
        gen_pows: *const u32, num_rotations: usize,
    );

    fn ypir_word_online_compute_batch_inspir(
        context: *mut std::ffi::c_void,
        queries: *const u64,
        y_body_condensed: *const u64,
        z_body_condensed: *const u64,
        response_out: *mut u8,
        response_bytes_per_batch: usize,
        batch_size: usize,
    );

    fn ypir_word_online_adopt_db(
        context: *mut std::ffi::c_void,
        d_db_stacked: *mut u8,
        d_db_u16: *mut u16,
    );

    fn ypir_word_online_free(context: *mut std::ffi::c_void);
}

#[cfg(feature = "cuda")]
pub struct WordOnlineContext {
    ctx: *mut std::ffi::c_void,
    poly_len: usize,
    db_cols: usize,
    num_rlwe_outputs: usize,
}
#[cfg(feature = "cuda")]
unsafe impl Send for WordOnlineContext {}
#[cfg(feature = "cuda")]
unsafe impl Sync for WordOnlineContext {}

#[cfg(feature = "cuda")]
impl WordOnlineContext {
    pub fn new(
        db: &[u16],
        db_rows: usize,
        db_rows_padded: usize,
        db_cols: usize,
        t_exp_left: usize,
        rlwe_q_prime_1: u64,
        rlwe_q_prime_2: u64,
        params: &spiral_rs::params::Params,
        max_batch_size: usize,
        apply_inv_n: bool,
    ) -> Result<Self, String> {
        let poly_len = params.poly_len;
        let crt_count = params.crt_count;
        let num_rlwe_outputs = db_cols / poly_len;
        let inv_n = if apply_inv_n {
            spiral_rs::number_theory::invert_uint_mod(poly_len as u64, params.modulus)
                .expect("Failed to compute inv_N mod Q")
        } else {
            1u64
        };

        let (forward_table, forward_prime_table, inverse_table, inverse_prime_table) =
            flatten_ntt_tables(params);

        let ctx = unsafe {
            ypir_word_online_init(
                db.as_ptr(),
                db_rows,
                db_rows_padded,
                db_cols,
                t_exp_left,
                rlwe_q_prime_1,
                rlwe_q_prime_2,
                poly_len as u32,
                crt_count as u32,
                params.moduli.as_ptr(),
                params.barrett_cr_1.as_ptr(),
                forward_table.as_ptr(),
                forward_prime_table.as_ptr(),
                inverse_table.as_ptr(),
                inverse_prime_table.as_ptr(),
                params.mod0_inv_mod1,
                params.mod1_inv_mod0,
                params.barrett_cr_0_modulus,
                params.barrett_cr_1_modulus,
                max_batch_size,
                inv_n,
            )
        };

        if ctx.is_null() {
            return Err("Failed to initialize Word online GPU context".to_string());
        }

        Ok(Self { ctx, poly_len, db_cols, num_rlwe_outputs })
    }

    /// Like `new` but skips DB upload. Call `adopt_db` afterwards with device ptrs.
    pub fn new_no_db(
        db_rows: usize,
        db_rows_padded: usize,
        db_cols: usize,
        t_exp_left: usize,
        rlwe_q_prime_1: u64,
        rlwe_q_prime_2: u64,
        params: &spiral_rs::params::Params,
        max_batch_size: usize,
        apply_inv_n: bool,
    ) -> Result<Self, String> {
        let poly_len = params.poly_len;
        let crt_count = params.crt_count;
        let num_rlwe_outputs = db_cols / poly_len;
        let inv_n = if apply_inv_n {
            spiral_rs::number_theory::invert_uint_mod(poly_len as u64, params.modulus)
                .expect("Failed to compute inv_N mod Q")
        } else {
            1u64
        };

        let (forward_table, forward_prime_table, inverse_table, inverse_prime_table) =
            flatten_ntt_tables(params);

        let ctx = unsafe {
            ypir_word_online_init(
                std::ptr::null(),
                db_rows,
                db_rows_padded,
                db_cols,
                t_exp_left,
                rlwe_q_prime_1,
                rlwe_q_prime_2,
                poly_len as u32,
                crt_count as u32,
                params.moduli.as_ptr(),
                params.barrett_cr_1.as_ptr(),
                forward_table.as_ptr(),
                forward_prime_table.as_ptr(),
                inverse_table.as_ptr(),
                inverse_prime_table.as_ptr(),
                params.mod0_inv_mod1,
                params.mod1_inv_mod0,
                params.barrett_cr_0_modulus,
                params.barrett_cr_1_modulus,
                max_batch_size,
                inv_n,
            )
        };

        if ctx.is_null() {
            return Err("Failed to initialize Word online GPU context (no DB)".to_string());
        }

        Ok(Self { ctx, poly_len, db_cols, num_rlwe_outputs })
    }

    /// Adopt DB device pointers from offline context (avoids second DB upload).
    pub fn adopt_db(&self, d_db_stacked: *mut u8, d_db_u16: *mut u16) {
        unsafe {
            ypir_word_online_adopt_db(self.ctx, d_db_stacked, d_db_u16);
        }
    }

    pub fn init_packing(
        &self,
        y_constants: &[u64],
        precomp_res: &[u64],
        precomp_vals: &[u64],
        precomp_tables: &[u64],
    ) {
        unsafe {
            ypir_word_online_init_packing(
                self.ctx,
                y_constants.as_ptr(), y_constants.len() * std::mem::size_of::<u64>(),
                precomp_res.as_ptr(), precomp_res.len() * std::mem::size_of::<u64>(),
                precomp_vals.as_ptr(), precomp_vals.len() * std::mem::size_of::<u64>(),
                precomp_tables.as_ptr(), precomp_tables.len() * std::mem::size_of::<u64>(),
            );
        }
    }

    pub fn compute_batch(
        &self,
        queries: &[u64],
        pack_pub_params_row_1s: &[u64],
        response_bytes_per_output: usize,
        batch_size: usize,
    ) -> Vec<Vec<Vec<u8>>> {
        let response_bytes_per_batch = self.num_rlwe_outputs * response_bytes_per_output;
        let mut response_flat = vec![0u8; batch_size * response_bytes_per_batch];

        unsafe {
            ypir_word_online_compute_batch(
                self.ctx,
                queries.as_ptr(),
                pack_pub_params_row_1s.as_ptr(),
                response_flat.as_mut_ptr(),
                response_bytes_per_batch,
                batch_size,
            );
        }

        let mut result = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let batch_start = b * response_bytes_per_batch;
            let mut outputs = Vec::with_capacity(self.num_rlwe_outputs);
            for o in 0..self.num_rlwe_outputs {
                let output_start = batch_start + o * response_bytes_per_output;
                let output_end = output_start + response_bytes_per_output;
                outputs.push(response_flat[output_start..output_end].to_vec());
            }
            result.push(outputs);
        }
        result
    }

    pub fn init_packing_inspir(
        &self,
        bold_t_condensed: &[u64],
        bold_t_bar_condensed: &[u64],
        bold_t_hat_condensed: &[u64],
        a_hat: &[u64],
        num_iter: usize,
        tables: &[u32],
        num_tables: usize,
        gen_pows: &[u32],
    ) {
        unsafe {
            ypir_word_online_init_packing_inspir(
                self.ctx,
                bold_t_condensed.as_ptr(), bold_t_condensed.len() * std::mem::size_of::<u64>(),
                bold_t_bar_condensed.as_ptr(), bold_t_bar_condensed.len() * std::mem::size_of::<u64>(),
                bold_t_hat_condensed.as_ptr(), bold_t_hat_condensed.len() * std::mem::size_of::<u64>(),
                a_hat.as_ptr(), a_hat.len() * std::mem::size_of::<u64>(),
                num_iter,
                tables.as_ptr(), num_tables,
                gen_pows.as_ptr(), gen_pows.len(),
            );
        }
    }

    pub fn compute_batch_inspir(
        &self,
        queries: &[u64],
        y_body_condensed: &[u64],
        z_body_condensed: &[u64],
        response_bytes_per_output: usize,
        batch_size: usize,
    ) -> Vec<Vec<Vec<u8>>> {
        let response_bytes_per_batch = self.num_rlwe_outputs * response_bytes_per_output;
        let mut response_flat = vec![0u8; batch_size * response_bytes_per_batch];

        unsafe {
            ypir_word_online_compute_batch_inspir(
                self.ctx,
                queries.as_ptr(),
                y_body_condensed.as_ptr(),
                z_body_condensed.as_ptr(),
                response_flat.as_mut_ptr(),
                response_bytes_per_batch,
                batch_size,
            );
        }

        let mut result = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let batch_start = b * response_bytes_per_batch;
            let mut outputs = Vec::with_capacity(self.num_rlwe_outputs);
            for o in 0..self.num_rlwe_outputs {
                let output_start = batch_start + o * response_bytes_per_output;
                let output_end = output_start + response_bytes_per_output;
                outputs.push(response_flat[output_start..output_end].to_vec());
            }
            result.push(outputs);
        }
        result
    }

    /// Run only Step 1 (GEMM + modswitch + inv_N) and return CRT-packed intermediates.
    pub fn compute_matmul_only(&self, queries: &[u64], batch_size: usize) -> Vec<u64> {
        let total = self.db_cols * batch_size;
        let mut intermediate = vec![0u64; total];
        unsafe {
            ypir_word_online_compute_matmul_only(
                self.ctx,
                queries.as_ptr(),
                intermediate.as_mut_ptr(),
                batch_size,
            );
        }
        intermediate
    }

    /// Initialize InspiRING packing from GPU device pointers (zero-copy from precomp).
    pub fn init_packing_inspir_from_gpu(
        &self,
        precomp: &mut super::inspiring::InspirPrecompContext,
        tables: &[u32],
        num_tables: usize,
        gen_pows: &[u32],
    ) {
        let (d_bt, d_btb, d_bth, d_ah) = precomp.take_device_ptrs();
        let num_iter = self.poly_len / 2 - 1;
        unsafe {
            ypir_word_online_init_packing_inspir_from_gpu(
                self.ctx,
                d_bt, d_btb, d_bth, d_ah,
                num_iter,
                tables.as_ptr(), num_tables,
                gen_pows.as_ptr(), gen_pows.len(),
            );
        }
    }

    pub fn num_rlwe_outputs(&self) -> usize {
        self.num_rlwe_outputs
    }

    pub fn db_cols(&self) -> usize {
        self.db_cols
    }
}

#[cfg(feature = "cuda")]
impl Drop for WordOnlineContext {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe { ypir_word_online_free(self.ctx); }
        }
    }
}
