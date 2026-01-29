/// CUDA-accelerated hint generation

use crate::measurement::Offline;

#[cfg(feature = "cuda")]
extern "C" {
    fn ypir_offline_init(
        db: *const u8,
        v_nega_perm_a: *const u32,
        moduli: *const u64,
        barrett_cr: *const u64,
        forward_table: *const u64,
        forward_prime_table: *const u64,
        inverse_table: *const u64,
        inverse_prime_table: *const u64,
        db_rows: u32,
        db_rows_padded: u32,
        db_cols: u32,
        n: u32,
        poly_len: u32,
        crt_count: u32,
        max_adds: u32,
        mod0_inv_mod1: u64,
        mod1_inv_mod0: u64,
        barrett_cr_0_modulus: u64,
        barrett_cr_1_modulus: u64,
    ) -> *mut std::ffi::c_void;

    fn compute_hint_0_cuda(context: *mut std::ffi::c_void, hint_0: *mut u64) -> i32;

    fn ypir_offline_free(context: *mut std::ffi::c_void);
}

#[cfg(feature = "cuda")]
pub struct OfflineComputeContext {
    ctx: *mut std::ffi::c_void,
    n: u32,
    db_cols: u32,
}

#[cfg(feature = "cuda")]
impl OfflineComputeContext {
    pub fn new(
        db_rows: u32,
        db_rows_padded: u32,
        db_cols: u32,
        n: u32,
        poly_len: u32,
        crt_count: u32,
        max_adds: u32,
        db: &[u8],
        v_nega_perm_a: &[u32],
        moduli: &[u64],
        barrett_cr: &[u64],
        forward_table: &[u64],
        forward_prime_table: &[u64],
        inverse_table: &[u64],
        inverse_prime_table: &[u64],
        mod0_inv_mod1: u64,
        mod1_inv_mod0: u64,
        barrett_cr_0_modulus: u64,
        barrett_cr_1_modulus: u64,
    ) -> Result<Self, String> {
        let ctx = unsafe {
            ypir_offline_init(
                db.as_ptr(),
                v_nega_perm_a.as_ptr(),
                moduli.as_ptr(),
                barrett_cr.as_ptr(),
                forward_table.as_ptr(),
                forward_prime_table.as_ptr(),
                inverse_table.as_ptr(),
                inverse_prime_table.as_ptr(),
                db_rows,
                db_rows_padded,
                db_cols,
                n,
                poly_len,
                crt_count,
                max_adds,
                mod0_inv_mod1,
                mod1_inv_mod0,
                barrett_cr_0_modulus,
                barrett_cr_1_modulus,
            )
        };

        if ctx.is_null() {
            Err("Failed to initialize GPU context".to_string())
        } else {
            Ok(OfflineComputeContext { ctx, n, db_cols })
        }
    }

    pub fn compute_hint_0(&self) -> Result<Vec<u64>, String> {
        let hint_size = (self.n * self.db_cols) as usize;
        let mut hint_0 = vec![0u64; hint_size];

        let result = unsafe { compute_hint_0_cuda(self.ctx, hint_0.as_mut_ptr()) };

        if result == 0 {
            Ok(hint_0)
        } else {
            Err("CUDA kernel failed".to_string())
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for OfflineComputeContext {
    fn drop(&mut self) {
        unsafe {
            ypir_offline_free(self.ctx);
        }
    }
}


#[cfg(all(feature = "cuda", feature = "toeplitz"))]
extern "C" {
    fn init_toeplitz_context(
        db: *const u8,
        v_nega_perm_a: *const u32,
        moduli: *const u64,
        barrett_cr: *const u64,
        db_rows: u32,
        db_rows_padded: u32,
        db_cols: u32,
        n: u32,
        crt_count: u32,
        max_adds: u32,
        mod0_inv_mod1: u64,
        mod1_inv_mod0: u64,
        barrett_cr_0_modulus: u64,
        barrett_cr_1_modulus: u64,
    ) -> *mut std::ffi::c_void;

    fn compute_hint_0_toeplitz(context: *mut std::ffi::c_void, hint_0: *mut u64) -> i32;

    fn free_toeplitz_context(context: *mut std::ffi::c_void);
}

#[cfg(all(feature = "cuda", feature = "toeplitz"))]
pub struct ToeplitzContext {
    ctx: *mut std::ffi::c_void,
    n: u32,
    db_cols: u32,
}

#[cfg(all(feature = "cuda", feature = "toeplitz"))]
impl ToeplitzContext {
    pub fn new(
        db_rows: u32,
        db_rows_padded: u32,
        db_cols: u32,
        n: u32,
        crt_count: u32,
        max_adds: u32,
        db: &[u8],
        v_nega_perm_a: &[u32],
        moduli: &[u64],
        barrett_cr: &[u64],
        mod0_inv_mod1: u64,
        mod1_inv_mod0: u64,
        barrett_cr_0_modulus: u64,
        barrett_cr_1_modulus: u64,
    ) -> Result<Self, String> {
        let ctx = unsafe {
            init_toeplitz_context(
                db.as_ptr(),
                v_nega_perm_a.as_ptr(),
                moduli.as_ptr(),
                barrett_cr.as_ptr(),
                db_rows,
                db_rows_padded,
                db_cols,
                n,
                crt_count,
                max_adds,
                mod0_inv_mod1,
                mod1_inv_mod0,
                barrett_cr_0_modulus,
                barrett_cr_1_modulus,
            )
        };

        if ctx.is_null() {
            Err("Failed to initialize Toeplitz GPU context".to_string())
        } else {
            Ok(ToeplitzContext { ctx, n, db_cols })
        }
    }

    pub fn compute_hint_0(&self) -> Result<Vec<u64>, String> {
        let hint_size = (self.n * self.db_cols) as usize;
        let mut hint_0 = vec![0u64; hint_size];

        let result = unsafe { compute_hint_0_toeplitz(self.ctx, hint_0.as_mut_ptr()) };

        if result == 0 {
            Ok(hint_0)
        } else {
            Err("CUDA Toeplitz kernel failed".to_string())
        }
    }
}

#[cfg(all(feature = "cuda", feature = "toeplitz"))]
impl Drop for ToeplitzContext {
    fn drop(&mut self) {
        unsafe {
            free_toeplitz_context(self.ctx);
        }
    }
}

#[cfg(feature = "cuda")]
extern "C" {
    fn ypir_online_init(
        db: *const u32,
        db_rows: usize,
        db_cols: usize,
        A2t: *const u64,
        smaller_db: *const u16,
        smaller_db_rows: usize,
    ) -> *mut std::ffi::c_void;

    fn ypir_online_init_ntt(
        context: *mut std::ffi::c_void,
        poly_len: u32,
        crt_count: u32,
        pt_bits: usize,
        lwe_modulus: u64,
        lwe_q_prime: u64,
        special_offs: usize,
        blowup_factor_ceil: usize,
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
    );

    fn ypir_init_packing_data(
        context: *mut std::ffi::c_void,
        y_constants: *const u64, y_constants_size: usize,
        prepacked_lwe: *const u64, prepacked_lwe_size: usize,
        precomp_res: *const u64, precomp_res_size: usize,
        precomp_vals: *const u64, precomp_vals_size: usize,
        precomp_tables: *const u64, precomp_tables_size: usize,
        fake_pack_pub_params: *const u64, fake_pack_pub_params_size: usize,
    );

    fn ypir_online_compute_full_batch(
        context: *mut std::ffi::c_void,
        query: *const u32,
        query_q2_batch: *const u64,
        batch_size: usize,
        all_hints_out: *mut u64,
        all_responses_out: *mut u64,
    ) -> i32;

    fn ypir_online_free(context: *mut std::ffi::c_void);
}

#[cfg(feature = "cuda")]
pub struct OnlineComputeContext {
    ctx: *mut std::ffi::c_void,
}

#[cfg(feature = "cuda")]
unsafe impl Send for OnlineComputeContext {}
#[cfg(feature = "cuda")]
unsafe impl Sync for OnlineComputeContext {}

#[cfg(feature = "cuda")]
impl OnlineComputeContext {
    pub fn new(
        db: &[u32],
        db_rows: usize,
        db_cols: usize,
        A2t: &[u64],
        smaller_db: &[u16],
        smaller_db_rows: usize,
    ) -> Result<Self, String> {
        let ctx = unsafe {
            ypir_online_init(
                db.as_ptr(),
                db_rows,
                db_cols,
                A2t.as_ptr(),
                smaller_db.as_ptr(),
                smaller_db_rows,
            )
        };

        if ctx.is_null() {
            Err("Failed to initialize Online GPU context".to_string())
        } else {
            Ok(OnlineComputeContext { ctx})
        }
    }

    pub fn init_ntt(
        &self,
        poly_len: u32,
        crt_count: u32,
        pt_bits: usize,
        lwe_modulus: u64,
        lwe_q_prime: u64,
        special_offs: usize,
        blowup_factor_ceil: usize,
        moduli: &[u64],
        barrett_cr: &[u64],
        forward_table: &[u64],
        forward_prime_table: &[u64],
        inverse_table: &[u64],
        inverse_prime_table: &[u64],
        mod0_inv_mod1: u64,
        mod1_inv_mod0: u64,
        barrett_cr_0_modulus: u64,
        barrett_cr_1_modulus: u64,
    ) {
        unsafe {
            ypir_online_init_ntt(
                self.ctx,
                poly_len,
                crt_count,
                pt_bits,
                lwe_modulus,
                lwe_q_prime,
                special_offs,
                blowup_factor_ceil,
                moduli.as_ptr(),
                barrett_cr.as_ptr(),
                forward_table.as_ptr(),
                forward_prime_table.as_ptr(),
                inverse_table.as_ptr(),
                inverse_prime_table.as_ptr(),
                mod0_inv_mod1,
                mod1_inv_mod0,
                barrett_cr_0_modulus,
                barrett_cr_1_modulus,
            );
        }
    }

    pub fn init_packing_data(
        &self,
        y_constants: &[u64],
        prepacked_lwe: &[u64],
        precomp_res: &[u64],
        precomp_vals: &[u64],
        precomp_tables: &[u64],
        fake_pack_pub_params: &[u64],
    ) {
        unsafe {
            ypir_init_packing_data(
                self.ctx,
                y_constants.as_ptr(), y_constants.len() * 8, // size in bytes
                prepacked_lwe.as_ptr(), prepacked_lwe.len() * 8,
                precomp_res.as_ptr(), precomp_res.len() * 8,
                precomp_vals.as_ptr(), precomp_vals.len() * 8,
                precomp_tables.as_ptr(), precomp_tables.len() * 8,
                fake_pack_pub_params.as_ptr(), fake_pack_pub_params.len() * 8,
            );
        }
    }

    pub fn compute_full_batch(
        &self,
        query: &[u32],
        query_q2_batch: &[u64],
        batch_size: usize,
        hint_size: usize,
        response_size: usize,
    ) -> Result<(Vec<u64>, Vec<u64>), String> {
        let mut all_hints = vec![0u64; batch_size * hint_size];
        let mut all_responses = vec![0u64; batch_size * response_size];

        let res = unsafe {
            ypir_online_compute_full_batch(
                self.ctx,
                query.as_ptr(),
                query_q2_batch.as_ptr(),
                batch_size,
                all_hints.as_mut_ptr(),
                all_responses.as_mut_ptr(),
            )
        };

        if res != 0 {
            return Err("ypir_online_compute_full_batch failed".to_string());
        }

        Ok((all_hints, all_responses))
    }

}

#[cfg(feature = "cuda")]
impl Drop for OnlineComputeContext {
    fn drop(&mut self) {
        unsafe {
            ypir_online_free(self.ctx);
        }
    }
}

