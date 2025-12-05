/// CUDA-accelerated hint generation

#[cfg(feature = "cuda")]
extern "C" {
    fn init_gpu_context(
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

    fn free_gpu_context(context: *mut std::ffi::c_void);
}

#[cfg(feature = "cuda")]
pub struct GPUContext {
    ctx: *mut std::ffi::c_void,
    n: u32,
    db_cols: u32,
}

#[cfg(feature = "cuda")]
impl GPUContext {
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
            init_gpu_context(
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
            Ok(GPUContext { ctx, n, db_cols })
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
impl Drop for GPUContext {
    fn drop(&mut self) {
        unsafe {
            free_gpu_context(self.ctx);
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub struct GPUContext;

#[cfg(not(feature = "cuda"))]
impl GPUContext {
    pub fn new(
        _db_rows: u32,
        _db_rows_padded: u32,
        _db_cols: u32,
        _n: u32,
        _poly_len: u32,
        _crt_count: u32,
        _max_adds: u32,
        _db: &[u8],
        _v_nega_perm_a: &[u32],
        _moduli: &[u64],
        _barrett_cr: &[u64],
        _forward_table: &[u64],
        _forward_prime_table: &[u64],
        _inverse_table: &[u64],
        _inverse_prime_table: &[u64],
        _mod0_inv_mod1: u64,
        _mod1_inv_mod0: u64,
        _barrett_cr_0_modulus: u64,
        _barrett_cr_1_modulus: u64,
    ) -> Result<Self, String> {
        Err("CUDA support not compiled".to_string())
    }

    pub fn compute_hint_0(&self) -> Result<Vec<u64>, String> {
        Err("CUDA support not compiled".to_string())
    }
}
