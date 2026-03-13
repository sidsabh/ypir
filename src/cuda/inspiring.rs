/// CUDA FFI bindings for InspiRING GPU precomputation.

use super::flatten_ntt_tables;

#[cfg(feature = "cuda")]
extern "C" {
    fn inspir_precomp_init(
        d_hint_0: *const u64,
        db_cols: u32,
        poly_len: u32,
        crt_count: u32,
        t_exp_left: u32,
        modulus_log2: u32,
        q2_bits: u32,
        num_outputs: u32,
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
        modulus: u64,
        w_mask: *const u64,
        v_mask: *const u64,
        mod_inv_poly: *const u64,
        tables: *const u32,
        num_tables: u32,
        gen_pows: *const u32,
        gen_pows_len: u32,
    ) -> *mut std::ffi::c_void;

    fn inspir_precomp_compute(context: *mut std::ffi::c_void);

    fn inspir_precomp_get_results(
        context: *mut std::ffi::c_void,
        out_bold_t: *mut *mut u64,
        out_bold_t_bar: *mut *mut u64,
        out_bold_t_hat: *mut *mut u64,
        out_a_hat: *mut *mut u64,
        out_bold_t_size: *mut usize,
        out_bold_t_bar_size: *mut usize,
        out_bold_t_hat_size: *mut usize,
        out_a_hat_size: *mut usize,
    );

    fn inspir_precomp_free(context: *mut std::ffi::c_void, free_outputs: bool);
}

#[cfg(feature = "cuda")]
pub struct InspirPrecompContext {
    ctx: *mut std::ffi::c_void,
    outputs_taken: bool,
}

#[cfg(feature = "cuda")]
unsafe impl Send for InspirPrecompContext {}
#[cfg(feature = "cuda")]
unsafe impl Sync for InspirPrecompContext {}

#[cfg(feature = "cuda")]
impl InspirPrecompContext {
    pub fn new(
        d_hint_0: *const u64,
        db_cols: u32,
        params: &spiral_rs::params::Params,
        num_outputs: u32,
        w_mask: &[u64],
        v_mask: &[u64],
        mod_inv_poly: &[u64],
        tables: &[u32],
        num_tables: u32,
        gen_pows: &[u32],
    ) -> Self {
        let poly_len = params.poly_len as u32;
        let crt_count = params.crt_count as u32;
        let t_exp_left = params.t_exp_left as u32;

        let (forward_table, forward_prime_table, inverse_table, inverse_prime_table) =
            flatten_ntt_tables(params);

        let ctx = unsafe {
            inspir_precomp_init(
                d_hint_0,
                db_cols,
                poly_len,
                crt_count,
                t_exp_left,
                params.modulus_log2 as u32,
                params.q2_bits as u32,
                num_outputs,
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
                params.modulus,
                w_mask.as_ptr(),
                v_mask.as_ptr(),
                mod_inv_poly.as_ptr(),
                tables.as_ptr(),
                num_tables,
                gen_pows.as_ptr(),
                gen_pows.len() as u32,
            )
        };

        Self { ctx, outputs_taken: false }
    }

    pub fn compute(&self) {
        unsafe { inspir_precomp_compute(self.ctx); }
    }

    /// Take ownership of the output device pointers.
    /// Returns (d_bold_t, d_bold_t_bar, d_bold_t_hat, d_a_hat).
    pub fn take_device_ptrs(&mut self) -> (*mut u64, *mut u64, *mut u64, *mut u64) {
        let mut d_bt: *mut u64 = std::ptr::null_mut();
        let mut d_btb: *mut u64 = std::ptr::null_mut();
        let mut d_bth: *mut u64 = std::ptr::null_mut();
        let mut d_ah: *mut u64 = std::ptr::null_mut();
        let mut sz1: usize = 0;
        let mut sz2: usize = 0;
        let mut sz3: usize = 0;
        let mut sz4: usize = 0;

        unsafe {
            inspir_precomp_get_results(
                self.ctx,
                &mut d_bt, &mut d_btb, &mut d_bth, &mut d_ah,
                &mut sz1, &mut sz2, &mut sz3, &mut sz4,
            );
        }
        self.outputs_taken = true;
        (d_bt, d_btb, d_bth, d_ah)
    }
}

#[cfg(feature = "cuda")]
impl Drop for InspirPrecompContext {
    fn drop(&mut self) {
        if !self.ctx.is_null() {
            // If outputs were taken (transferred to online context), don't free them
            unsafe { inspir_precomp_free(self.ctx, !self.outputs_taken); }
        }
    }
}
