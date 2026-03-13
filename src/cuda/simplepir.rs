/// CUDA FFI bindings for SimplePIR offline and online computation.

use super::flatten_ntt_tables;

// ==================== SimplePIR Offline ====================

#[cfg(feature = "cuda")]
extern "C" {
    fn ypir_sp_offline_init(
        db: *const u16,
        db_rows: usize,
        db_rows_padded: usize,
        db_cols: usize,
        query_ntt: *const u64,
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
    ) -> *mut std::ffi::c_void;

    fn ypir_sp_compute_hint_0(context: *mut std::ffi::c_void, hint_out: *mut u64);

    fn ypir_sp_offline_free(context: *mut std::ffi::c_void);
}

#[cfg(feature = "cuda")]
pub struct SPOfflineContext {
    ctx: *mut std::ffi::c_void,
    poly_len: usize,
    db_cols: usize,
}
#[cfg(feature = "cuda")]
unsafe impl Send for SPOfflineContext {}
#[cfg(feature = "cuda")]
unsafe impl Sync for SPOfflineContext {}

#[cfg(feature = "cuda")]
impl SPOfflineContext {
    pub fn new(
        db: &[u16],
        db_rows: usize,
        db_rows_padded: usize,
        db_cols: usize,
        query_ntt: &[u64],
        params: &spiral_rs::params::Params,
    ) -> Result<Self, String> {
        let poly_len = params.poly_len;
        let crt_count = params.crt_count;

        let (forward_table, forward_prime_table, inverse_table, inverse_prime_table) =
            flatten_ntt_tables(params);

        let ctx = unsafe {
            ypir_sp_offline_init(
                db.as_ptr(),
                db_rows,
                db_rows_padded,
                db_cols,
                query_ntt.as_ptr(),
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
            )
        };

        if ctx.is_null() {
            return Err("Failed to initialize SimplePIR offline GPU context".to_string());
        }

        Ok(Self { ctx, poly_len, db_cols })
    }

    pub fn compute_hint_0(&self) -> Result<Vec<u64>, String> {
        let mut hint_gpu = vec![0u64; self.poly_len * self.db_cols];

        unsafe {
            ypir_sp_compute_hint_0(self.ctx, hint_gpu.as_mut_ptr());
        }

        // Transpose from db_cols x poly_len to poly_len x db_cols
        let mut hint_out = vec![0u64; self.poly_len * self.db_cols];
        for col in 0..self.db_cols {
            for z in 0..self.poly_len {
                hint_out[z * self.db_cols + col] = hint_gpu[col * self.poly_len + z];
            }
        }

        Ok(hint_out)
    }
}

#[cfg(feature = "cuda")]
impl Drop for SPOfflineContext {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe {
                ypir_sp_offline_free(self.ctx);
            }
        }
    }
}

// ==================== SimplePIR Online ====================

#[cfg(feature = "cuda")]
extern "C" {
    fn ypir_sp_online_init(
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
    ) -> *mut std::ffi::c_void;

    fn ypir_sp_online_init_packing(
        context: *mut std::ffi::c_void,
        y_constants: *const u64, y_constants_size: usize,
        precomp_res: *const u64, precomp_res_size: usize,
        precomp_vals: *const u64, precomp_vals_size: usize,
        precomp_tables: *const u64, precomp_tables_size: usize,
    );

    fn ypir_sp_online_compute_batch(
        context: *mut std::ffi::c_void,
        queries: *const u64,
        pack_pub_params_row_1s: *const u64,
        response_out: *mut u8,
        response_bytes_per_batch: usize,
        batch_size: usize,
    );

    fn ypir_sp_online_free(context: *mut std::ffi::c_void);
}

#[cfg(feature = "cuda")]
pub struct SPOnlineContext {
    ctx: *mut std::ffi::c_void,
    _poly_len: usize,
    _db_cols: usize,
    num_rlwe_outputs: usize,
    _rlwe_q_prime_1: u64,
    _rlwe_q_prime_2: u64,
}
#[cfg(feature = "cuda")]
unsafe impl Send for SPOnlineContext {}
#[cfg(feature = "cuda")]
unsafe impl Sync for SPOnlineContext {}

#[cfg(feature = "cuda")]
impl SPOnlineContext {
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
    ) -> Result<Self, String> {
        let poly_len = params.poly_len;
        let crt_count = params.crt_count;
        let num_rlwe_outputs = db_cols / poly_len;

        let (forward_table, forward_prime_table, inverse_table, inverse_prime_table) =
            flatten_ntt_tables(params);

        let ctx = unsafe {
            ypir_sp_online_init(
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
            )
        };

        if ctx.is_null() {
            return Err("Failed to initialize SimplePIR online GPU context".to_string());
        }

        Ok(Self { ctx, _poly_len: poly_len, _db_cols: db_cols, num_rlwe_outputs, _rlwe_q_prime_1: rlwe_q_prime_1, _rlwe_q_prime_2: rlwe_q_prime_2 })
    }

    pub fn init_packing(
        &self,
        y_constants: &[u64],
        precomp_res: &[u64],
        precomp_vals: &[u64],
        precomp_tables: &[u64],
    ) {
        unsafe {
            ypir_sp_online_init_packing(
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
            ypir_sp_online_compute_batch(
                self.ctx,
                queries.as_ptr(),
                pack_pub_params_row_1s.as_ptr(),
                response_flat.as_mut_ptr(),
                response_bytes_per_batch,
                batch_size,
            );
        }

        // Reshape: flat -> [batch][output][bytes]
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

    pub fn num_rlwe_outputs(&self) -> usize {
        self.num_rlwe_outputs
    }
}

#[cfg(feature = "cuda")]
impl Drop for SPOnlineContext {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            unsafe {
                ypir_sp_online_free(self.ctx);
            }
        }
    }
}
