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
        rlwe_q_prime_1: u64,
        rlwe_q_prime_2: u64,
        special_offs: usize,
        t_exp_left : usize,
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
        pack_pub_params_row_1s_batch: *const u64,
        batch_size: usize,
        responses: *mut u8,
    );

    fn ypir_online_free(context: *mut std::ffi::c_void);
}

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
    /// Create a new SimplePIR offline GPU context
    ///
    /// # Arguments
    /// * `db` - Database as u8 slice, column-major: db_cols x db_rows_padded
    /// * `db_rows` - Actual number of rows
    /// * `db_rows_padded` - Padded number of rows
    /// * `db_cols` - Number of columns
    /// * `query_ntt` - Precomputed query in NTT form: db_rows_poly x crt_count x poly_len
    /// * `params` - Spiral parameters for NTT tables
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
        
        // Flatten NTT tables
        let mut forward_table = Vec::with_capacity(crt_count * poly_len);
        let mut forward_prime_table = Vec::with_capacity(crt_count * poly_len);
        let mut inverse_table = Vec::with_capacity(crt_count * poly_len);
        let mut inverse_prime_table = Vec::with_capacity(crt_count * poly_len);
        
        for i in 0..crt_count {
            forward_table.extend_from_slice(&params.ntt_tables[i][0]);
            forward_prime_table.extend_from_slice(&params.ntt_tables[i][1]);
            inverse_table.extend_from_slice(&params.ntt_tables[i][2]);
            inverse_prime_table.extend_from_slice(&params.ntt_tables[i][3]);
        }
        
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
    
    /// Compute hint_0 on GPU and return transposed result
    /// Returns: poly_len x db_cols (row-major, same as CPU version)
    pub fn compute_hint_0(&self) -> Result<Vec<u64>, String> {
        // GPU outputs db_cols x poly_len, we need to transpose to poly_len x db_cols
        let mut hint_gpu = vec![0u64; self.poly_len * self.db_cols];
        
        unsafe {
            ypir_sp_compute_hint_0(self.ctx, hint_gpu.as_mut_ptr());
        }
        
        // Transpose from db_cols x poly_len to poly_len x db_cols
        let mut hint_out = vec![0u64; self.poly_len * self.db_cols];
        for col in 0..self.db_cols {
            for z in 0..self.poly_len {
                // GPU stored: hint_gpu[col * poly_len + z]
                // CPU expects: hint_out[z * db_cols + col]
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
        rlwe_q_prime_1: u64,
        rlwe_q_prime_2: u64,
        special_offs: usize,
        t_exp_left: usize,
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
                rlwe_q_prime_1,
                rlwe_q_prime_2,
                special_offs,
                t_exp_left,
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
        pack_pub_params_row_1s_batch: &[u64],
        batch_size: usize,
        responses: &mut [u8]
    ) {
        unsafe {
            ypir_online_compute_full_batch(
                self.ctx,
                query.as_ptr(),
                query_q2_batch.as_ptr(),
                pack_pub_params_row_1s_batch.as_ptr(),
                batch_size,
                responses.as_mut_ptr(),
            )
        };
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

// Add these to your existing cuda/mod.rs

// ==================== SimplePIR Online FFI ====================
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

/// SimplePIR Online GPU Context
#[cfg(feature = "cuda")]
pub struct SPOnlineContext {
    ctx: *mut std::ffi::c_void,
    poly_len: usize,
    db_cols: usize,
    num_rlwe_outputs: usize,
    rlwe_q_prime_1: u64,
    rlwe_q_prime_2: u64,
}
#[cfg(feature = "cuda")]
unsafe impl Send for SPOnlineContext {}
#[cfg(feature = "cuda")]
unsafe impl Sync for SPOnlineContext {}

#[cfg(feature = "cuda")]
impl SPOnlineContext {
    /// Create SimplePIR online GPU context
    pub fn new(
        db: &[u16],
        db_rows: usize,
        db_rows_padded: usize,
        db_cols: usize,
        t_exp_left: usize,
        rlwe_q_prime_1: u64,
        rlwe_q_prime_2: u64,
        params: &spiral_rs::params::Params,
    ) -> Result<Self, String> {
        let poly_len = params.poly_len;
        let crt_count = params.crt_count;
        let num_rlwe_outputs = db_cols / poly_len;
        
        // Flatten NTT tables
        let mut forward_table = Vec::with_capacity(crt_count * poly_len);
        let mut forward_prime_table = Vec::with_capacity(crt_count * poly_len);
        let mut inverse_table = Vec::with_capacity(crt_count * poly_len);
        let mut inverse_prime_table = Vec::with_capacity(crt_count * poly_len);
        
        for i in 0..crt_count {
            forward_table.extend_from_slice(&params.ntt_tables[i][0]);
            forward_prime_table.extend_from_slice(&params.ntt_tables[i][1]);
            inverse_table.extend_from_slice(&params.ntt_tables[i][2]);
            inverse_prime_table.extend_from_slice(&params.ntt_tables[i][3]);
        }
        
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
            )
        };
        
        if ctx.is_null() {
            return Err("Failed to initialize SimplePIR online GPU context".to_string());
        }
        
        Ok(Self { 
            ctx, 
            poly_len, 
            db_cols, 
            num_rlwe_outputs,
            rlwe_q_prime_1,
            rlwe_q_prime_2,
        })
    }
    
    /// Upload packing data from offline precomputation
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
    
    /// Run online computation
    /// 
    /// # Arguments
    /// * `query` - Query vector (db_rows_padded u64s)
    /// * `pack_pub_params_row_1s` - Automorphism keys row 1s (condensed)
    /// 
    /// # Returns
    /// Response bytes for all RLWE outputs
    pub fn compute_batch(
        &self,
        queries: &[u64],                    // batch_size * db_rows_padded
        pack_pub_params_row_1s: &[u64],     // batch_size * ell * t_exp_left * poly_len
        response_bytes_per_output : usize,
        batch_size : usize
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