#[cfg(target_feature = "avx2")]
use std::arch::x86_64::*;
use std::{marker::PhantomData, ops::Range, time::Instant};

use log::debug;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use spiral_rs::aligned_memory::AlignedMemory64;
use spiral_rs::{arith::*, client::*, params::*, poly::*};

use crate::convolution::naive_multiply_matrices;
use crate::measurement::Measurement;

use super::{
    bits::*,
    client::*,
    convolution::{negacyclic_perm_u32, Convolution},
    kernel::*,
    lwe::*,
    matmul::matmul_vec_packed,
    modulus_switch::ModulusSwitch,
    packing::*,
    params::*,
    scheme::*,
    transpose::*,
    util::*,
};

pub fn generate_y_constants<'a>(
    params: &'a Params,
) -> (Vec<PolyMatrixNTT<'a>>, Vec<PolyMatrixNTT<'a>>) {
    let mut y_constants = Vec::new();
    let mut neg_y_constants = Vec::new();
    for num_cts_log2 in 1..params.poly_len_log2 + 1 {
        let num_cts = 1 << num_cts_log2;

        // Y = X^(poly_len / num_cts)
        let mut y_raw = PolyMatrixRaw::zero(params, 1, 1);
        y_raw.data[params.poly_len / num_cts] = 1;
        let y = y_raw.ntt();

        let mut neg_y_raw = PolyMatrixRaw::zero(params, 1, 1);
        neg_y_raw.data[params.poly_len / num_cts] = params.modulus - 1;
        let neg_y = neg_y_raw.ntt();

        y_constants.push(y);
        neg_y_constants.push(neg_y);
    }

    (y_constants, neg_y_constants)
}

/// Takes a matrix of u64s and returns a matrix of T's.
///
/// Input is row x cols u64's.
/// Output is out_rows x cols T's.
pub fn split_alloc(
    buf: &[u64],
    special_bit_offs: usize,
    rows: usize,
    cols: usize,
    out_rows: usize,
    inp_mod_bits: usize,
    pt_bits: usize,
) -> Vec<u16> {
    let mut out = vec![0u16; out_rows * cols];

    assert!(out_rows >= rows);
    assert!(inp_mod_bits >= pt_bits);

    for j in 0..cols {
        let mut bytes_tmp = vec![0u8; out_rows * inp_mod_bits / 8];

        // read this column
        let mut bit_offs = 0;
        for i in 0..rows {
            // even though hint was stored in u64, it only needed u32, then mod switch down to u28, so we grab the value and only store those 28 bits
            let inp = buf[i * cols + j];
            // if j < 10 {
            //     debug!("({},{}) inp: {}", i, j, inp);
            // }

            if i == rows - 1 {
                // the reason we explicitly set this is because we ceil'd the division for the offset in each 
                bit_offs = special_bit_offs;
            }

            // if j == 4095 {
            //     debug!("write: {}/{} {}/{}", j, cols, i, rows);
            // }
            write_bits(&mut bytes_tmp, inp, bit_offs, inp_mod_bits);
            bit_offs += inp_mod_bits;
        }

        // debug!("stretch: {}", j);

        // now, 'stretch' the column vertically
        let mut bit_offs = 0;
        for i in 0..out_rows {
            // if j == 4095 {
            //     debug!("stretch: {}/{}", i, out_rows);
            //     debug!("reading at offs: {}, {} bits", bit_offs, pt_bits);
            //     debug!("into byte buffer of len: {}", bytes_tmp.len());
            //     debug!("writing at {} in out of len {}", i * cols + j, out.len());
            // }
            let out_val = read_bits(&bytes_tmp, bit_offs, pt_bits);
            out[i * cols + j] = out_val as u16;
            // if j == 4095 {
            //     debug!("wrote at {} in out of len {}", i * cols + j, out.len());
            // }
            bit_offs += pt_bits;
            if bit_offs >= out_rows * inp_mod_bits {
                break;
            }
        }

        // debug!("here {}", j);
        // debug!(
        //     "out {}",
        //     out[(special_bit_offs / pt_bits) * cols + j].to_u64()
        // );
        // debug!("buf {}", buf[(rows - 1) * cols + j] & ((1 << pt_bits) - 1));

        assert_eq!(
            out[(special_bit_offs / pt_bits) * cols + j] as u64,
            buf[(rows - 1) * cols + j] & ((1 << pt_bits) - 1)
        );
    }

    out
}

pub fn generate_fake_pack_pub_params<'a>(params: &'a Params) -> Vec<PolyMatrixNTT<'a>> {
    // sk is 0, since this is server pre-processing no client
    let pack_pub_params = raw_generate_expansion_params(
        &params,
        &PolyMatrixRaw::zero(&params, 1, 1),
        params.poly_len_log2,
        params.t_exp_left,
        &mut ChaCha20Rng::from_entropy(),
        &mut ChaCha20Rng::from_seed(STATIC_SEED_2),
    );
    pack_pub_params
}

pub type Precomp<'a> = Vec<(PolyMatrixNTT<'a>, Vec<PolyMatrixNTT<'a>>, Vec<Vec<usize>>)>;

#[derive(Clone)]
pub struct OfflinePrecomputedValues<'a> {
    pub hint_0: Vec<u64>,
    pub hint_1: Vec<u64>,
    pub pseudorandom_query_1: Vec<PolyMatrixNTT<'a>>,
    pub y_constants: (Vec<PolyMatrixNTT<'a>>, Vec<PolyMatrixNTT<'a>>),
    pub smaller_server: Option<YServer<'a, u16>>,
    pub prepacked_lwe: Vec<Vec<PolyMatrixNTT<'a>>>,
    pub fake_pack_pub_params: Vec<PolyMatrixNTT<'a>>,
    pub precomp: Precomp<'a>,
    #[cfg(feature = "cuda")]
    pub cuda_context: Option<std::sync::Arc<crate::cuda::OnlineComputeContext>>,
    #[cfg(feature = "cuda")]
    pub sp_cuda_context: Option<std::sync::Arc<crate::cuda::SPOnlineContext>>,
}

#[derive(Clone)]
pub struct YServer<'a, T> {
    params: &'a Params,
    smaller_params: Params,
    db_buf_aligned: AlignedMemory64, // db_buf: Vec<u8>, // stored transposed
    phantom: PhantomData<T>,
    pad_rows: bool,
    ypir_params: YPIRParams,
}

pub trait DbRowsPadded {
    fn db_rows_padded(&self) -> usize;
}

impl DbRowsPadded for Params {
    fn db_rows_padded(&self) -> usize {
        let db_rows = 1 << (self.db_dim_1 + self.poly_len_log2);
        db_rows
        // let db_rows_padded = db_rows + db_rows / (16 * 8);
        // db_rows_padded
    }
}

impl<'a, T> YServer<'a, T>
where
    T: Sized + Copy + ToU64 + Default,
    *const T: ToM512,
{
    pub fn new<'b, I>(
        params: &'a Params,
        mut db: I,
        is_simplepir: bool,
        inp_transposed: bool,
        pad_rows: bool,
    ) -> Self
    where
        I: Iterator<Item = T>,
    {
        // TODO: hack
        // let lwe_params = LWEParams::default();
        let mut ypir_params = YPIRParams::default();
        ypir_params.is_simplepir = is_simplepir;
        let bytes_per_pt_el = std::mem::size_of::<T>(); //1; //((lwe_params.pt_modulus as f64).log2() / 8.).ceil() as usize;

        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_rows_padded = if pad_rows {
            params.db_rows_padded()
        } else {
            db_rows
        };
        let db_cols = if is_simplepir {
            params.instances * params.poly_len
        } else {
            1 << (params.db_dim_2 + params.poly_len_log2)
        };

        let sz_bytes = db_rows_padded * db_cols * bytes_per_pt_el;

        let mut db_buf_aligned = AlignedMemory64::new(sz_bytes / 8);
        let db_buf_mut = as_bytes_mut(&mut db_buf_aligned);
        let db_buf_ptr = db_buf_mut.as_mut_ptr() as *mut T;

        // SID: this is where we load the database from the passed in iterator (random bits u8::sample())
        // the database is in column major format
        // SIDQ: why are the column dims db_dim_x + poly_len ? just because we need l1, l2 to be multiples of ring dim for NTT
        for i in 0..db_rows {
            for j in 0..db_cols {
                let idx = if inp_transposed {
                    i * db_cols + j
                } else {
                    j * db_rows_padded + i
                };

                unsafe {
                    *db_buf_ptr.add(idx) = db.next().unwrap();
                    // *db_buf_ptr.add(idx) = if i < db_rows {
                    //     db.next().unwrap()
                    // } else {
                    //     T::default()
                    // };
                }
            }
        }

        // Parameters for the second round (the "DoublePIR" round)
        let smaller_params = if is_simplepir {
            params.clone()
        } else {
            let lwe_params = LWEParams::default();
            let pt_bits = (params.pt_modulus as f64).log2().floor() as usize;
            let blowup_factor = lwe_params.q2_bits as f64 / pt_bits as f64;
            let mut smaller_params = params.clone();
            smaller_params.db_dim_1 = params.db_dim_2;
            smaller_params.db_dim_2 = ((blowup_factor * (lwe_params.n + 1) as f64)
                / params.poly_len as f64)
                .log2()
                .ceil() as usize;

            let out_rows = 1 << (smaller_params.db_dim_2 + params.poly_len_log2);
            assert_eq!(smaller_params.db_dim_1, params.db_dim_2);
            assert!(out_rows as f64 >= (blowup_factor * (lwe_params.n + 1) as f64));
            smaller_params
        };


        Self {
            params,
            smaller_params,
            db_buf_aligned,
            phantom: PhantomData,
            pad_rows,
            ypir_params,
        }
    }

    pub fn db_rows_padded(&self) -> usize {
        if self.pad_rows {
            self.params.db_rows_padded()
        } else {
            1 << (self.params.db_dim_1 + self.params.poly_len_log2)
        }
    }

    pub fn db_cols(&self) -> usize {
        if self.ypir_params.is_simplepir {
            self.params.instances * self.params.poly_len
        } else {
            1 << (self.params.db_dim_2 + self.params.poly_len_log2)
        }
    }

    pub fn multiply_batched_with_db_packed<const K: usize>(
        &self,
        aligned_query_packed: &[u64],
        query_rows: usize,
    ) -> AlignedMemory64 {
        // let db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_rows_padded = self.db_rows_padded();
        let db_cols = self.db_cols();
        assert_eq!(aligned_query_packed.len(), K * query_rows * db_rows_padded);
        assert_eq!(K, 1);
        assert_eq!(query_rows, 1);

        let now = Instant::now();
        let mut result = AlignedMemory64::new(K * db_cols);
        fast_batched_dot_product_avx512::<K, _>(
            self.params,
            result.as_mut_slice(),
            aligned_query_packed,
            db_rows_padded,
            &self.db(),
            db_rows_padded,
            db_cols,
        );
        debug!("Fast dot product in {} us", now.elapsed().as_micros());

        result
    }

    pub fn lwe_multiply_batched_with_db_packed<const K: usize>(
        &self,
        aligned_query_packed: &[u32],
    ) -> Vec<u32> {
        let _db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_cols = self.db_cols();
        let db_rows_padded = self.db_rows_padded();
        assert_eq!(aligned_query_packed.len(), K * db_rows_padded);
        // assert_eq!(aligned_query_packed[db_rows + 1], 0);

        let mut result = vec![0u32; (db_cols + 8) * K];
        let now = Instant::now();
        // let mut result = AlignedMemory64::new(K * db_cols + 8);
        // lwe_fast_batched_dot_product_avx512::<K, _>(
        //     self.params,
        //     result.as_mut_slice(),
        //     aligned_query_packed,
        //     db_rows,
        //     &self.db(),
        //     db_rows,
        //     db_cols,
        // );
        let a_rows = db_cols;
        let a_true_cols = db_rows_padded;
        let a_cols = a_true_cols / 4; // order is inverted on purpose, because db is transposed
        let b_rows = a_true_cols;
        let b_cols = K;
        // this guy just calculates mat A x mat B (vec if K=1) in AVX form
        // if you swap dimensions, then a column major A rows x cols is equivalent to a row major transposed A cols x rows
        matmul_vec_packed(
            result.as_mut_slice(),
            self.db_u32(),
            aligned_query_packed,
            a_rows,
            a_cols,
            b_rows,
            b_cols,
        );
        let t = Instant::now();
        // this op is negligible compared to the matmul_vec_packed, a cost worth it to use SimplePIR's kernel and compute (A x DB) our hint on a column major DB
        let result = transpose_generic(&result, db_cols, K);
        debug!("Transpose in {} us", t.elapsed().as_micros());
        debug!("Fast dot product in {} us", now.elapsed().as_micros());

        result
    }

    pub fn multiply_with_db_ring(
        &self,
        preprocessed_query: &[PolyMatrixNTT],
        col_range: Range<usize>,
        seed_idx: u8,
    ) -> Vec<u64> {
        let db_rows_poly = 1 << (self.params.db_dim_1);
        let db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        assert_eq!(preprocessed_query.len(), db_rows_poly);

        // assert_eq!(db_rows_poly, 1); // temporary restriction

        // let mut preprocessed_query = Vec::new();
        // for query_el in query {
        //     let query_raw = query_el.raw();
        //     let query_raw_transformed =
        //         negacyclic_perm(query_raw.get_poly(0, 0), 0, self.params.modulus);
        //     let mut query_transformed_pol = PolyMatrixRaw::zero(self.params, 1, 1);
        //     query_transformed_pol
        //         .as_mut_slice()
        //         .copy_from_slice(&query_raw_transformed);
        //     preprocessed_query.push(query_transformed_pol.ntt());
        // }

        let mut result = Vec::new();
        let db = self.db();

        let mut prod = PolyMatrixNTT::zero(self.params, 1, 1);
        let mut db_elem_poly = PolyMatrixRaw::zero(self.params, 1, 1);
        let mut db_elem_ntt = PolyMatrixNTT::zero(self.params, 1, 1);

        for col in col_range.clone() {
            let mut sum = PolyMatrixNTT::zero(self.params, 1, 1);

            for row in 0..db_rows_poly {
                for z in 0..self.params.poly_len {
                    db_elem_poly.data[z] =
                        db[col * db_rows + row * self.params.poly_len + z].to_u64();
                }
                to_ntt(&mut db_elem_ntt, &db_elem_poly);

                multiply(&mut prod, &preprocessed_query[row], &db_elem_ntt); // CRT-based modulo multiply

                if row == db_rows_poly - 1 {
                    add_into(&mut sum, &prod); // can take modulo since NTT-friendly
                } else {
                    add_into_no_reduce(&mut sum, &prod);
                }
            }

            let sum_raw = sum.raw();

            // do negacyclic permutation (for first mul only)
            if seed_idx == SEED_0 && !self.ypir_params.is_simplepir {
                // this never happens (negacyclic rules)
                let sum_raw_transformed =
                    negacyclic_perm(sum_raw.get_poly(0, 0), 0, self.params.modulus);
                result.extend(&sum_raw_transformed);
            } else {
                result.extend(sum_raw.as_slice());
            }
        }

        // result
        let now = Instant::now();
        let res = transpose_generic(&result, col_range.len(), self.params.poly_len);
        debug!("transpose in {} us", now.elapsed().as_micros());
        res
    }

    pub fn generate_pseudorandom_query(&self, public_seed_idx: u8) -> Vec<PolyMatrixNTT<'a>> {
        let mut client = Client::init(&self.params);
        client.generate_secret_keys(); // short-secret LWE for automorphisms
        let y_client = YClient::new(&mut client, &self.params);

        // recall - for the query over DB2 (DoublePIR), we used this same call
        // it generates RLWE encryption of a column, which is equivalent to an LWE encryption under the negacyclic matrix
        let query = y_client.generate_query_impl(public_seed_idx, self.params.db_dim_1, true, 0, None, None);

        // this is basically just grabbing the random portion (A2)
        // correct, but not efficient
        let query_mapped = query
            .iter()
            .map(|x| x.submatrix(0, 0, 1, 1))
            .collect::<Vec<_>>();

        let mut preprocessed_query = Vec::new();
        for query_raw in query_mapped {
            // let query_raw_transformed =
            //     negacyclic_perm(query_raw.get_poly(0, 0), 0, self.params.modulus);
            // let query_raw_transformed = query_raw.get_poly(0, 0);
            let query_raw_transformed = if public_seed_idx == SEED_0 {
                negacyclic_perm(query_raw.get_poly(0, 0), 0, self.params.modulus)
                // query_raw.get_poly(0, 0).to_owned()
            } else {
                negacyclic_perm(query_raw.get_poly(0, 0), 0, self.params.modulus)
            };
            let mut query_transformed_pol = PolyMatrixRaw::zero(self.params, 1, 1);
            query_transformed_pol
                .as_mut_slice()
                .copy_from_slice(&query_raw_transformed);
            preprocessed_query.push(query_transformed_pol.ntt());
        }

        preprocessed_query
    }

    pub fn answer_hint_ring(&self, public_seed_idx: u8, cols: usize) -> Vec<u64> {
        let preprocessed_query = self.generate_pseudorandom_query(public_seed_idx);

        let res = self.multiply_with_db_ring(&preprocessed_query, 0..cols, public_seed_idx);

        res
    }

    pub fn generate_hint_0(&self) -> Vec<u64> {
        let _db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_cols = self.db_cols();

        let mut rng_pub = ChaCha20Rng::from_seed(get_seed(SEED_0));
        let lwe_params = LWEParams::default();

        // pseudorandom LWE query is n x db_rows
        let psuedorandom_query =
            generate_matrix_ring(&mut rng_pub, lwe_params.n, lwe_params.n, db_cols);

        // db is db_cols x db_rows (!!!)
        // hint_0 is n x db_cols
        let hint_0 = naive_multiply_matrices(
            &psuedorandom_query,
            lwe_params.n,
            db_cols,
            &self.db(),
            self.db_rows_padded(), // TODO: doesn't quite work
            db_cols,
            true,
        );
        hint_0.iter().map(|&x| x as u64).collect::<Vec<_>>()
    }


    #[cfg(feature = "cuda")]
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

    #[cfg(feature = "cuda")]
    pub fn compute_hint_0_with_context(&self, gpu_ctx: &crate::cuda::OfflineComputeContext) -> Vec<u64> {
        gpu_ctx.compute_hint_0().expect("GPU computation failed")
    }

    /// Initialize Toeplitz GPU context for matrix-based multiplication
    /// Uses coefficient-form polynomials instead of NTT
    #[cfg(all(feature = "cuda", feature = "toeplitz"))]
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

    #[cfg(all(feature = "cuda", feature = "toeplitz"))]
    pub fn compute_hint_0_with_toeplitz(&self, toeplitz_ctx: &crate::cuda::ToeplitzContext) -> Vec<u64> {
        toeplitz_ctx.compute_hint_0().expect("Toeplitz GPU computation failed")
    }

    pub fn generate_hint_0_ring(&self) -> Vec<u64> {
        let lwe_params = LWEParams::default();
        let conv = Convolution::new(lwe_params.n); // wrapper around NTT operations

        let db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_cols = self.db_cols();
        let n = lwe_params.n;

        let mut hint_0 = vec![0u64; n * db_cols];
        let convd_len = conv.params().crt_count * conv.params().poly_len;

        let mut rng_pub = ChaCha20Rng::from_seed(get_seed(SEED_0));
        let mut v_nega_perm_a = Vec::new();
        for _ in 0..db_rows / n {
            let mut a = vec![0u32; n];
            for idx in 0..n {
                a[idx] = rng_pub.sample::<u32, _>(rand::distributions::Standard);
            }
            let nega_perm_a = negacyclic_perm_u32(&a); // re-write a so that an LWE can be interpreted as an RLWE under the same key—yes, negacyclic_matrix_u32 is the Toeplitz analogue of negacyclic_perm_u32
            let nega_perm_a_ntt = conv.ntt(&nega_perm_a);
            v_nega_perm_a.push(nega_perm_a_ntt);
        }

        // this is where we handle the 4.1."Modulus Selection" section
        // q = 2^32 is not an NTT friendly modulus, so we instead work over a much larger group that doesn't overflow
        // in the Toeplitz regime:
        // to avoid overflow, we essentially sum products of Zq, ZN. so the max element is q*N*d (== lwe_params.modulus, lwe_params.pt_modulus, lwe_params.n, respectively)
        // we are working in the coefficient space, so we can just work with uin64_t >> log(q*N*d) and mod 2^32 whenever to avoid overflow on the column
        // recall, per polymut, we are bounded by the q*N*d overflow. but we want to sum m_1 of these (l1/d1) at a time per coefficient, so the potential overflow is
        // l1 * q (2^32) * 2^8. if l1 > 2^22, then we overflow, which is a super large DB (>512 GB if 1bit)
        // the real overflow we saw was with trying to use GEMM32, so we needed to do CRT, etc. etc.
        
        // anyways!!!
        // we are actually doing NTT, so we are solving a different problem here.
        // we are working over the Ring space with modulus ~=2^56 - forget CRT for now, it's a detail
        // if we have coefficients bounded by 2^32 and 2^8, the maximum coefficient for a polynomial multiply will be: (N)(2^8)(2^32)
        // this is denoted as log2_conv_output
        // the idea is that: we are in R_q with q=2^32 right now with d=1024. this is not NTT_friendly (the 2n-th root of unity does not exist in the multiplicative grp)
        // but we can just pretend we are working over the integers as long as we never mod, do the INTT, then mod after.

        // in order to pretend we work in the integers, we work in the larger group
        // Q/A: it's quite neat that we can work in the larger group that exactly works for the RLWE automorphorisms!?
        // so we work in Z_q2[x]/(X^d2) with q2_bits >> log2_conv_output, INTT when we get close to overflowing the group!
        // TADA

        let log2_conv_output =
            log2(lwe_params.modulus) + log2(lwe_params.n as u64) + log2(lwe_params.pt_modulus);
        let log2_modulus = log2(conv.params().modulus); // ~= 2^56
        let log2_max_adds = log2_modulus - log2_conv_output - 1; // -1 so we stay BELOW the 2^56
        let max_adds = 1 << log2_max_adds;

        for col in 0..db_cols {
            let mut tmp_col = vec![0u64; convd_len]; // for each column, we compute one polynomial, stored in CRT format
            for outer_row in 0..db_rows / n {
                let start_idx = col * self.db_rows_padded() + outer_row * n;
                let pt_col = &self.db()[start_idx..start_idx + n];
                let pt_col_u32 = pt_col
                    .iter()
                    .map(|&x| x.to_u64() as u32)
                    .collect::<Vec<_>>();

                let pt_ntt = conv.ntt(&pt_col_u32);
                let convolved_ntt = conv.pointwise_mul(&v_nega_perm_a[outer_row], &pt_ntt); // pointwise mul over CRT-based NTT representation

                for r in 0..convd_len {
                    tmp_col[r] += convolved_ntt[r] as u64;
                }

                // write to hint_0
                if outer_row % max_adds == max_adds - 1 || outer_row == db_rows / n - 1 {
                    let mut col_poly_u32 = vec![0u32; convd_len];

                    // re-mod by CRT moduli
                    for i in 0..conv.params().crt_count {
                        for j in 0..conv.params().poly_len {
                            let val = barrett_coeff_u64(
                                conv.params(),
                                tmp_col[i * conv.params().poly_len + j],
                                i,
                            );
                            col_poly_u32[i * conv.params().poly_len + j] = val as u32;
                        }
                    }
                    
                    let col_poly_raw = conv.raw(&col_poly_u32);

                    // writes one column of the row-major matrix
                    for i in 0..n {
                        hint_0[i * db_cols + col] += col_poly_raw[i] as u64;
                        hint_0[i * db_cols + col] %= 1u64 << 32; // mod to Zq
                    }
                    tmp_col.fill(0);
                }
            }
        }
        hint_0
    }

    pub fn answer_query(&self, aligned_query_packed: &[u64]) -> AlignedMemory64 {
        self.multiply_batched_with_db_packed::<1>(aligned_query_packed, 1)
    }

    pub fn answer_batched_queries<const K: usize>(
        &self,
        aligned_queries_packed: &[u64],
    ) -> AlignedMemory64 {
        self.multiply_batched_with_db_packed::<K>(aligned_queries_packed, 1)
    }

    #[cfg(feature = "cuda")]
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
    #[cfg(feature = "cuda")]
    pub fn compute_hint_0_simplepir_gpu(&self, gpu_ctx: &crate::cuda::SPOfflineContext) -> Vec<u64> {
        gpu_ctx.compute_hint_0().expect("SimplePIR GPU hint_0 computation failed")
    }

    pub fn perform_offline_precomputation_simplepir(
        &self,
        measurement: Option<&mut Measurement>,
    ) -> OfflinePrecomputedValues {
        // Set up some parameters
        let params = self.params;
        assert!(self.ypir_params.is_simplepir);
        let db_cols = params.instances * params.poly_len;
        let num_rlwe_outputs = db_cols / params.poly_len;
        
        // Begin offline precomputation
        let now = Instant::now();
        
        #[cfg(feature = "cuda")]
        let hint_0: Vec<u64> = {
            let init_start = Instant::now();
            let gpu_ctx = self.init_hint_0_simplepir_gpu_context();
            let init_time = init_start.elapsed();
            
            let compute_start = Instant::now();
            let gpu_result = self.compute_hint_0_simplepir_gpu(&gpu_ctx);
            let compute_time = compute_start.elapsed();

            debug!("SimplePIR GPU init: {:?}, compute: {:?}", init_time, compute_time);
            gpu_result
        };
        
        #[cfg(not(feature = "cuda"))]
        let hint_0: Vec<u64> = self.answer_hint_ring(SEED_0, db_cols);
        
        // hint_0 is poly_len x db_cols
        let simplepir_prep_time_ms = now.elapsed().as_millis();
        if let Some(measurement) = measurement {
            measurement.offline.simplepir_prep_time_ms = simplepir_prep_time_ms as usize;
        }

        let now = Instant::now();
        let y_constants = generate_y_constants(&params);

        let combined = [&hint_0[..], &vec![0u64; db_cols]].concat();
        assert_eq!(combined.len(), db_cols * (params.poly_len + 1));
        let prepacked_lwe = prep_pack_many_lwes(&params, &combined, num_rlwe_outputs);

        let fake_pack_pub_params = generate_fake_pack_pub_params(&params);

        let mut precomp: Precomp = Vec::new();
        for i in 0..prepacked_lwe.len() {
            let tup = precompute_pack(
                params,
                params.poly_len_log2,
                &prepacked_lwe[i],
                &fake_pack_pub_params,
                &y_constants,
            );
            precomp.push(tup);
        }
        debug!("Precomp in {} us", now.elapsed().as_micros());

        #[cfg(not(feature = "cuda"))]
        {
            OfflinePrecomputedValues {
                hint_0,
                hint_1: vec![],
                pseudorandom_query_1: vec![],
                y_constants,
                smaller_server: None,
                prepacked_lwe,
                fake_pack_pub_params,
                precomp,
            }
        }
        
        #[cfg(feature = "cuda")]
        {
            // Initialize CUDA online context
            let sp_cuda_context = {
                debug!("Initializing SimplePIR CUDA online context...");
                let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
                
                match crate::cuda::SPOnlineContext::new(
                    self.db_u16(),
                    db_rows,
                    self.db_rows_padded(),
                    db_cols,
                    params.t_exp_left,
                    params.get_q_prime_1(),
                    params.get_q_prime_2(),
                    params,
                ) {
                    Ok(ctx) => {
                        // Flatten packing data
                        let mut y_constants_flat = Vec::new();
                        for m in &y_constants.0 { y_constants_flat.extend_from_slice(m.as_slice()); }
                        for m in &y_constants.1 { y_constants_flat.extend_from_slice(m.as_slice()); }
                        
                        let mut precomp_res_flat = Vec::new();
                        let mut precomp_vals_flat = Vec::new();
                        let mut precomp_tables_flat = Vec::new();
                        
                        for (res, vals, tables) in &precomp {
                            precomp_res_flat.extend_from_slice(res.as_slice());
                            for v in vals {
                                let condensed = condense_matrix(params, v);
                                for row in 0..v.rows {
                                    let poly = condensed.get_poly(row, 0);
                                    precomp_vals_flat.extend_from_slice(&poly[..params.poly_len]);
                                }
                            }
                            for t in tables {
                                for &val in t {
                                    precomp_tables_flat.push(val as u64);
                                }
                            }
                        }
                        
                        ctx.init_packing(
                            &y_constants_flat,
                            &precomp_res_flat,
                            &precomp_vals_flat,
                            &precomp_tables_flat,
                        );
                        
                        debug!("SimplePIR CUDA online context initialized");
                        Some(std::sync::Arc::new(ctx))
                    }
                    Err(e) => {
                        log::warn!("Failed to create SimplePIR CUDA context: {}", e);
                        None
                    }
                }
            };
            
            OfflinePrecomputedValues {
                hint_0,
                hint_1: vec![],
                pseudorandom_query_1: vec![],
                y_constants,
                smaller_server: None,
                prepacked_lwe,
                fake_pack_pub_params,
                precomp,
                cuda_context: None,
                sp_cuda_context
            }
        }
    }

    pub fn perform_offline_precomputation(
        &self,
        measurement: Option<&mut Measurement>,
    ) -> OfflinePrecomputedValues {
        // Set up some parameters

        let params = self.params;
        let lwe_params = LWEParams::default();
        assert!(!self.ypir_params.is_simplepir);

        let db_cols = 1 << (params.db_dim_2 + params.poly_len_log2);

        // LWE reduced moduli
        let lwe_q_prime_bits = lwe_params.q2_bits as usize;
        let lwe_q_prime = lwe_params.get_q_prime_2();

        // The number of bits represented by a plaintext RLWE coefficient
        let pt_bits = (params.pt_modulus as f64).log2().floor() as usize;
        // assert_eq!(pt_bits, 16);

        // The factor by which ciphertext values are bigger than plaintext values
        let blowup_factor = lwe_q_prime_bits as f64 / pt_bits as f64;
        debug!("blowup_factor: {}", blowup_factor);
        // assert!(blowup_factor.ceil() - blowup_factor >= 0.05);

        // The starting index of the final value (the '1' in lwe_params.n + 1)
        // This is rounded to start on a pt_bits boundary
        let special_offs =
            ((lwe_params.n * lwe_q_prime_bits) as f64 / pt_bits as f64).ceil() as usize;
        let special_bit_offs = special_offs * pt_bits;

        // Parameters for the second round (the "DoublePIR" round)
        let mut smaller_params = params.clone();
        smaller_params.db_dim_1 = params.db_dim_2;
        smaller_params.db_dim_2 = ((blowup_factor * (lwe_params.n + 1) as f64)
            / params.poly_len as f64)
            .log2()
            .ceil() as usize;

        let out_rows = 1 << (smaller_params.db_dim_2 + params.poly_len_log2);
        let rho = 1 << smaller_params.db_dim_2;
        assert_eq!(smaller_params.db_dim_1, params.db_dim_2);
        assert!(out_rows as f64 >= (blowup_factor * (lwe_params.n + 1) as f64));

        debug!(
            "the first {} LWE output ciphertexts of the DoublePIR round (out of {} total) are query-indepednent",
            special_offs, out_rows
        );
        debug!(
            "the next {} LWE output ciphertexts are query-dependent",
            blowup_factor.ceil() as usize
        );
        debug!("the rest are zero");

        // Begin offline precomputation

        let simplepir_prep_time_ms: u128;
        #[cfg(feature = "cuda")]
        let hint_0: Vec<u64> = {
            let lwe_params = LWEParams::default();
            let conv = Convolution::new(lwe_params.n);

            #[cfg(feature = "toeplitz")]
            let res = {
                let init_start = Instant::now();
                let toeplitz_ctx = self.init_toeplitz_context(&lwe_params, &conv);
                let init_time = init_start.elapsed();

                let compute_start = Instant::now();
                let gpu_result = self.compute_hint_0_with_toeplitz(&toeplitz_ctx);
                let compute_time = compute_start.elapsed();

                debug!("GPU (Toeplitz) init: {:?}, compute: {:?}", init_time, compute_time);
                (gpu_result, compute_time)
            };

            #[cfg(not(feature = "toeplitz"))]
            let res = {
                let init_start = Instant::now();
                let gpu_ctx = self.init_hint_0_gpu_context(&lwe_params, &conv);
                let init_time = init_start.elapsed();

                let compute_start = Instant::now();
                let gpu_result = self.compute_hint_0_with_context(&gpu_ctx);
                let compute_time = compute_start.elapsed();

                debug!("GPU (NTT) init: {:?}, compute: {:?}", init_time, compute_time);
                (gpu_result, compute_time)
            };

            let (gpu_result, compute_time) = res;
            simplepir_prep_time_ms = compute_time.as_millis();
            gpu_result
            // // Verify against CPU
            // let cpu_start = Instant::now();
            // let cpu_result = self.generate_hint_0_ring();
            // let cpu_time = cpu_start.elapsed();
            
            // debug!("CPU time: {:?}", cpu_time);
            // debug!("Speedup: {:.2}x", cpu_time.as_secs_f64() / compute_time.as_secs_f64());
            
            // let mut matches = true;
            // for i in 0..gpu_result.len() {
            //     if gpu_result[i] != cpu_result[i] {
            //         log::error!("Mismatch at index {}: GPU={}, CPU={}", i, gpu_result[i], cpu_result[i]);
            //         matches = false;
            //         break;
            //     }
            // }
            
            // if matches {
            //     debug!("✓ GPU matches CPU");
            //     gpu_result
            // } else {
            //     log::warn!("✗ GPU/CPU mismatch, using CPU");
            //     cpu_result
            // }
        };

        #[cfg(not(feature = "cuda"))]
        let hint_0: Vec<u64> = {
            let now = Instant::now();
            let result = self.generate_hint_0_ring();
            simplepir_prep_time_ms = now.elapsed().as_millis();
            result
        };
        // hint_0 is n x db_cols
        if let Some(measurement) = measurement {
            measurement.offline.simplepir_prep_time_ms = simplepir_prep_time_ms as usize;
        }
        // The debug message for non-CUDA case is moved inside the cfg block.
        // For CUDA, the timing is already debugged inside the block.

        // compute (most of) the secondary hint
        let intermediate_cts = [&hint_0[..], &vec![0u64; db_cols]].concat(); // concat so we add space for the SimplePIR repsonse in Z_q^l2
        let intermediate_cts_rescaled = intermediate_cts
            .iter()
            .map(|x| rescale(*x, lwe_params.modulus, lwe_q_prime))
            .collect::<Vec<_>>();

        // now we have hint_0, we just modulus shift it down
        // SID: not sure why we're storing it as a Vec<u64> if we already computed the hint, modded by q (2^32), then again modulus switched down to 2^28

        // split and do a second PIR over intermediate_cts
        // split into blowup_factor=q/p instances (so that all values are now mod p)
        // the second PIR is over a database of db_cols x (blowup_factor * (lwe_params.n + 1)) values mod p

        // inp: (lwe_params.n + 1, db_cols)
        // out: (out_rows >= (lwe_params.n + 1) * blowup_factor, db_cols)
        //      we are 'stretching' the columns (and padding)

        debug!("Splitting intermediate cts...");

        // smaller_db: [H1 | T]: Z_p^(~(k*(d1+1)) + ~DB_dim_2) (~ because poly padded)
        // have to expand over the row-space because the column space is fixed
        // write now T is all zeroes per the concat
        // u16 because the plaintext space for the RLWE is 2^15
        let smaller_db = split_alloc(
            &intermediate_cts_rescaled,
            special_bit_offs,
            lwe_params.n + 1, // n represents H1, 1 is for T
            db_cols,
            out_rows,
            lwe_q_prime_bits,
            pt_bits,
        );
        assert_eq!(smaller_db.len(), db_cols * out_rows);

        debug!("Done splitting intermediate cts.");

        // This is the 'intermediate' db after the first pass of PIR and expansion
        // INP TRANSPOSED == TRUE!!!
        // smaller_db is row-major matrix of out_rows x db_cols
        // its stored as row-major as well
        let smaller_server: YServer<u16> = YServer::<u16>::new(
            &self.smaller_params,
            smaller_db.into_iter(),
            false,
            true,
            false,
        );
        debug!("gen'd smaller server.");


        // we just want to calculate H2 = A2 * H1 (recall, we padded H1 num_rows to poly_len)
        // this is an alternate way of generating a hint
        // for hint_0, we called generate_hint_0_ring, which the whole NTT thing
        // in perform_offline_precomputation_simplepir, we use this method to generate hint_0, so they must be functionally equivalent 

        // there is an identity between the encryption of query_row and the computation of hint_0
            // we work in the LWE space
            // query_row was encrypted by sampling a polynomial in q1=2^32 modulus, then getting the negacyclic_matrix and encrypting d1 pt at a time
            // hint_0 was computed using NTTs by sampling the same polynomial, using negacyclic_perm
            // negacyclic isn't really necessary since we don't pack this (it is for YPIR-SP)
        // there is an identify between the encryption of query_col and the computation of hint_1
            // we work in RLWE space
            // query_col was encrypted using the polynomial in q2=2^56 modulus, encrypting d2 pt at a time
            // hint_1 is computed using the same same poylnomial
            // neither was negacyclic, so in order to pack and unpack that has to be done at some point (CDKS 3.2 /JeremyKun)
            // yes-confirmed the random portions for the preprocess are negacyclically transformed in prepack_many_lwes, and the random portion for on the SimplePIR response encryption is negacyclically transformed before packing!!

        // gets A2 in NTT form through obfuscated method
        // then does same NTT multiply as in generate_hint_0_ring, but NTT-friendly so no modulo concerns
        // PolyMatrixRaw/NTT are stored in u64, so there was no overflow concern on adds (mods at the end)
        
        
        // fascinatingly, they pass the rows as the cols, but the DB was stored as row-major, so the column major access will actually be a row-based access
        // we initialized smaller_server with the transpose, so it didn't change it to column major, we also set its params to be swapped just like we pass in here
        // we compute DB2 * A2, iterating each row by poly and multiplying by A2's poly, giving H2 stored row-major: out_rows x poly_len
        // at the end, they transpose, getting a row-major poly_len x out_rows
        let hint_1 = smaller_server.answer_hint_ring(
            SEED_1,
            1 << (smaller_server.params.db_dim_2 + smaller_server.params.poly_len_log2),
        );
        assert_eq!(hint_1.len(), params.poly_len * out_rows);
        // T was 0, so transp(T*A2) will also be 0
        assert_eq!(hint_1[special_offs], 0);
        assert_eq!(hint_1[special_offs + 1], 0);

        // A2 in NTT form (we already generated this in the creation of hint_1)
        let pseudorandom_query_1 = smaller_server.generate_pseudorandom_query(SEED_1);

        // generates the possible rotations needed for the CDKS FFT tree algorithm
        let y_constants = generate_y_constants(&params);

        // now we just add the last row to store the DoublePIR response
        let combined = [&hint_1[..], &vec![0u64; out_rows]].concat(); // stored in row major
        assert_eq!(combined.len(), out_rows * (params.poly_len + 1)); // full DoublePIR response, 0s everywhere besides H2

        // all this does is get the rho many CDKS RLWE squares, drops the b (0), negacylic perms them, then puts them in NTT form. it is a full CT, but the non-random portion is empty
        // TODO: QUESTION: why did we create the hint in NTT form, convert to RAW, then negacyclic then NTT again?
        // couldn't we avoid these NTTs and do negacyclic in the NTT domain?
        let prepacked_lwe = prep_pack_many_lwes(&params, &combined, rho);
        assert_eq!(prepacked_lwe.len(), rho);
        assert_eq!(prepacked_lwe[0].len(), params.poly_len);

        let now = Instant::now();
        // generates automorphism keys, doesn't actually encrypt sk, so only random portion is valid
        let fake_pack_pub_params = generate_fake_pack_pub_params(&params);

        let mut precomp: Precomp = Vec::new();
        for i in 0..prepacked_lwe.len() {
            // for each CDKS square, we compute the final RLWE random portion
            // Algo2 from CDKS, only on the random portion
            // input: d2 many polynomials, log(d2) many automorphism keys, Y constants in NTT for multiplication-based rotations (the value of 1X^(d2/(2^l)) for l in [1..log(d2)])
            // output.0 is the 1 RLWE for the random portion
            // output.1 is the cached NTT for g−1𝑧(𝜏 (𝑐0)) at each level
            // output.2 is the cached lookup tables for the NTT automorphism (reduces to a permutation). tau->{(i, remap_i)}
            let tup: (PolyMatrixNTT<'_>, Vec<PolyMatrixNTT<'_>>, Vec<Vec<usize>>) = precompute_pack(
                params,
                params.poly_len_log2,
                &prepacked_lwe[i],
                &fake_pack_pub_params,
                &y_constants,
            );
            precomp.push(tup);
        }
        debug!("Precomp in {} us", now.elapsed().as_micros());

        #[cfg(not(feature = "cuda"))]
        {
            OfflinePrecomputedValues {
                    hint_0,
                    hint_1,
                    pseudorandom_query_1,
                    y_constants,
                    smaller_server: Some(smaller_server),
                    prepacked_lwe,
                    fake_pack_pub_params,
                    precomp
            }
        }

        // GPU Upload Hook: Prepare CUDA context for online computation
        #[cfg(feature = "cuda")]
        {
            let cuda_context = {
                debug!("Uploading data to GPU for online computation...");
                let upload_start = Instant::now();

                // Upload primary database for Step 1 (SimplePIR)
                // DB is stored column-major: db_cols × db_rows_padded
                // Pack 4 u8 values into each u32 (BASIS=8 bits)
                // Keep column-major layout: db[col][row/4] in packed format

                let db_rows_padded = self.db_rows_padded();
                let db_cols = self.db_cols();
                let db = self.db();
                let packed_rows = db_rows_padded / 4;

                let mut db_u32_packed = vec![0u32; db_cols * packed_rows];

                // Pack while maintaining column-major layout
                for col in 0..db_cols {
                    for packed_row in 0..packed_rows {
                        let mut packed = 0u32;
                        for i in 0..4 {
                            let row = packed_row * 4 + i;
                            // Column-major index: col * db_rows_padded + row
                            let val = db[col * db_rows_padded + row].to_u64() as u32;
                            packed |= val << (i * 8);
                        }
                        // Column-major packed index: col * packed_rows + packed_row
                        db_u32_packed[col * packed_rows + packed_row] = packed;
                    }
                }

                debug!("Packed DB (column-major): {} bytes -> {} u32 values ({} cols × {} packed_rows)",
                    db.len(), db_u32_packed.len(), db_cols, packed_rows);

                // Create dummy A2t for now (Step 2 not implemented yet)
                let flat_query: Vec<u64> = pseudorandom_query_1
                    .iter()
                    .flat_map(|m| m.get_poly(0, 0).iter().copied())
                    .collect();

                // Get smaller_server DB for upload to GPU
                let smaller_db = smaller_server.db();
                let smaller_db_rows = out_rows;

                debug!("Uploading smaller_server DB: {} rows × {} cols = {} u16 values",
                    smaller_db_rows, db_cols, smaller_db.len());

                match crate::cuda::OnlineComputeContext::new(
                    &db_u32_packed,
                    self.db_cols(),           // db_rows in CUDA = logical db_cols
                    self.db_rows_padded(), // db_cols in CUDA = db_rows_padded / 4 (packed)
                    &flat_query,
                    smaller_db,
                    smaller_db_rows,
                ) {
                    Ok(ctx) => {
                        let upload_time = upload_start.elapsed();
                        debug!("GPU upload completed in {:?}", upload_time);
                        debug!("  DB size: {} MB", db_u32_packed.len() * 4 / (1024 * 1024));
                        
                        // Initialize NTT parameters
                        let sp = &smaller_server.params;
                        
                        debug!("Flattening NTT tables...");
                        let mut forward_table = Vec::with_capacity(sp.crt_count * sp.poly_len);
                        let mut forward_prime_table = Vec::with_capacity(sp.crt_count * sp.poly_len);
                        let mut inverse_table = Vec::with_capacity(sp.crt_count * sp.poly_len);
                        let mut inverse_prime_table = Vec::with_capacity(sp.crt_count * sp.poly_len);
                        
                        for i in 0..sp.crt_count {
                            forward_table.extend_from_slice(&sp.ntt_tables[i][0]);
                            forward_prime_table.extend_from_slice(&sp.ntt_tables[i][1]);
                            inverse_table.extend_from_slice(&sp.ntt_tables[i][2]);
                            inverse_prime_table.extend_from_slice(&sp.ntt_tables[i][3]);
                        }
                        debug!("Flattening done. Calling init_ntt...");
                        
                        let pt_bits = (params.pt_modulus as f64).log2().floor() as usize;
                        ctx.init_ntt(
                            sp.poly_len as u32,
                            sp.crt_count as u32,
                            pt_bits,
                            lwe_params.modulus,
                            lwe_q_prime,
                            params.get_q_prime_1(),
                            params.get_q_prime_2(),
                            special_offs,
                            params.t_exp_left,
                            blowup_factor.ceil() as usize,
                            &sp.moduli,
                            &sp.barrett_cr_1,
                            &forward_table,
                            &forward_prime_table,
                            &inverse_table,
                            &inverse_prime_table,
                            sp.mod0_inv_mod1,
                            sp.mod1_inv_mod0,
                            sp.barrett_cr_0_modulus,
                            sp.barrett_cr_1_modulus,
                        );
                        debug!("init_ntt returned.");

                        Some(std::sync::Arc::new(ctx))
                    }
                    Err(e) => {
                        log::warn!("Failed to create CUDA context: {}", e);
                        None
                    }
                }
            };

            if let Some(ref ctx) = cuda_context {
                debug!("Flattening packing data...");
                
                // y_constants
                let mut y_constants_flat = Vec::new();
                for m in &y_constants.0 { y_constants_flat.extend_from_slice(m.as_slice()); }
                for m in &y_constants.1 { y_constants_flat.extend_from_slice(m.as_slice()); }
                
                // prepacked_lwe
                let mut prepacked_lwe_flat = Vec::new();
                for v in &prepacked_lwe {
                    for m in v {
                        prepacked_lwe_flat.extend_from_slice(m.as_slice());
                    }
                }
                
                // precomp
                let mut precomp_res_flat = Vec::new();
                let mut precomp_vals_flat = Vec::new();
                let mut precomp_tables_flat = Vec::new();
                
                for (res, vals, tables) in &precomp {
                    precomp_res_flat.extend_from_slice(res.as_slice());
                    for v in vals {
                        let condensed = condense_matrix(params, v);
                        for row in 0..v.rows {
                            let poly = condensed.get_poly(row, 0);
                            precomp_vals_flat.extend_from_slice(&poly[..params.poly_len]);
                        }
                    }
                    for t in tables {
                        for &val in t {
                            precomp_tables_flat.push(val as u64);
                        }
                    }
                }

                // fake_pack_pub_params
                let mut fake_pack_pub_params_flat = Vec::new();
                for m in &fake_pack_pub_params {
                    fake_pack_pub_params_flat.extend_from_slice(m.as_slice());
                }
                
                debug!("Flattening done. Uploading packing data...");
                ctx.init_packing_data(
                    &y_constants_flat,
                    &prepacked_lwe_flat,
                    &precomp_res_flat,
                    &precomp_vals_flat,
                    &precomp_tables_flat,
                    &fake_pack_pub_params_flat
                );
            }
            OfflinePrecomputedValues {
                hint_0,
                hint_1,
                pseudorandom_query_1,
                y_constants,
                smaller_server: Some(smaller_server),
                prepacked_lwe,
                fake_pack_pub_params,
                precomp,
                cuda_context,
                sp_cuda_context: None
            }
        }
    }

    /// Perform SimplePIR-style YPIR
    #[cfg(not(feature = "cuda"))]
    pub fn perform_online_computation_simplepir(
        &self,
        first_dim_queries_packed: &[&[u64]],
        offline_vals: &OfflinePrecomputedValues<'a>,
        pack_pub_params_row_1s: &[&[PolyMatrixNTT<'a>]],
        mut measurement: Option<&mut Measurement>,
    ) -> Vec<Vec<Vec<u8>>> {
        assert!(self.ypir_params.is_simplepir);

        // Set up some parameters

        let params = self.params;

        let y_constants = &offline_vals.y_constants;
        let prepacked_lwe = &offline_vals.prepacked_lwe;
        let precomp = &offline_vals.precomp;

        // RLWE reduced moduli
        let rlwe_q_prime_1 = params.get_q_prime_1();
        let rlwe_q_prime_2 = params.get_q_prime_2();

        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = params.instances * params.poly_len;

        assert_eq!(first_dim_queries_packed[0].len(), params.db_rows_padded());

        // Begin online computation

        let first_pass = Instant::now();
        debug!("Performing mul...");
        let mut intermediate = AlignedMemory64::new(db_cols);
        fast_batched_dot_product_avx512::<1, T>(
            &params,
            intermediate.as_mut_slice(),
            first_dim_queries_packed[0],
            db_rows,
            self.db(),
            db_rows,
            db_cols,
        );
        debug!("Done w mul...");
        let first_pass_time_ms = first_pass.elapsed().as_millis();
        if let Some(ref mut m) = measurement {
            m.online.first_pass_time_ms = first_pass_time_ms as usize;
        }

        let ring_packing = Instant::now();
        let num_rlwe_outputs = db_cols / params.poly_len;
        let packed = pack_many_lwes(
            &params,
            &prepacked_lwe,
            &precomp,
            intermediate.as_slice(),
            num_rlwe_outputs,
            &pack_pub_params_row_1s[0],
            &y_constants,
        );
        debug!("Packed...");
        if let Some(m) = measurement {
            m.online.ring_packing_time_ms = ring_packing.elapsed().as_millis() as usize;
        }

        let mut packed_mod_switched = Vec::with_capacity(packed.len());
        for ct in packed.iter() {
            let res = ct.raw();
            let res_switched = res.switch(rlwe_q_prime_1, rlwe_q_prime_2);
            packed_mod_switched.push(res_switched);
        }

        vec![packed_mod_switched]
    }

    #[cfg(feature = "cuda")]
    pub fn perform_online_computation_simplepir(
        &self,
        first_dim_queries_packed: &[&[u64]],
        offline_vals: &OfflinePrecomputedValues<'a>,
        pack_pub_params_row_1s: &[&[PolyMatrixNTT<'a>]],
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
        
        // Flatten pack_pub_params_row_1s: batch_size * ell * t_exp_left * poly_len
        let mut pub_params_flat: Vec<u64> = Vec::new();
        for keys in pack_pub_params_row_1s {
            for key in keys.iter() {
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

    #[cfg(not(feature = "cuda"))]
    pub fn perform_online_computation<const K: usize>(
        &self,
        offline_vals: &mut OfflinePrecomputedValues<'a>,
        first_dim_queries_packed: &[u32],
        second_dim_queries: &[(&[u64], &[PolyMatrixNTT<'a>])],
        mut measurement: Option<&mut Measurement>,
    ) -> Vec<Vec<Vec<u8>>> {
        // Set up some parameters

        let params = self.params;
        let lwe_params = LWEParams::default();

        let db_cols = self.db_cols();

        // RLWE reduced moduli
        let rlwe_q_prime_1 = params.get_q_prime_1();
        let rlwe_q_prime_2 = params.get_q_prime_2();

        // LWE reduced moduli
        let lwe_q_prime_bits = lwe_params.q2_bits as usize;
        let lwe_q_prime = lwe_params.get_q_prime_2();

        // The number of bits represented by a plaintext RLWE coefficient
        let pt_bits = (params.pt_modulus as f64).log2().floor() as usize;
        // assert_eq!(pt_bits, 16);

        // The factor by which ciphertext values are bigger than plaintext values  // SID: calling this blowup_factor is misleading because it's comparing Zp to Zq' instead of Zn to Zq'
        let blowup_factor = lwe_q_prime_bits as f64 / pt_bits as f64;
        debug!("blowup_factor: {}", blowup_factor);
        // assert!(blowup_factor.ceil() - blowup_factor >= 0.05);

        // The starting index of the final value (the '1' in lwe_params.n + 1)
        // This is rounded to start on a pt_bits boundary
        // used in DoublePIR packing step to differentiate the H2/c2*H1 from the T*A2/c2*T
        let special_offs =
            ((lwe_params.n * lwe_q_prime_bits) as f64 / pt_bits as f64).ceil() as usize;

        // Parameters for the second round (the "DoublePIR" round)
        let mut smaller_params = params.clone();
        smaller_params.db_dim_1 = params.db_dim_2;
        // realistically, this is (d1+1)*K
        // we always divide by poly_len so that we can pad to a power of 2 multiple of poly_len 
        // so why is K the correct decomposition?
        // well, we defintely are operating on p (2^15) as the plaintext space so we have to decompose our 2^32 vals into that
        // before we decompose, we modulus switch down, so this is accurate
        smaller_params.db_dim_2 = ((blowup_factor * (lwe_params.n + 1) as f64)
            / params.poly_len as f64)
            .log2()
            .ceil() as usize;

        let out_rows = 1 << (smaller_params.db_dim_2 + params.poly_len_log2);
        // number of blocks of RLWE that we pack
        let rho = 1 << smaller_params.db_dim_2;
        assert_eq!(smaller_params.db_dim_1, params.db_dim_2);
        assert!(out_rows as f64 >= (blowup_factor * (lwe_params.n + 1) as f64));

        // Load offline precomputed values
        let hint_1_combined = &mut offline_vals.hint_1;
        let pseudorandom_query_1 = &offline_vals.pseudorandom_query_1;
        let y_constants = &offline_vals.y_constants;
        let smaller_server = offline_vals.smaller_server.as_mut().unwrap();
        let prepacked_lwe = &offline_vals.prepacked_lwe;
        let fake_pack_pub_params = &offline_vals.fake_pack_pub_params;
        let precomp = &offline_vals.precomp;

        // Begin online computation

        let online_phase = Instant::now();
        let first_pass = Instant::now();
        // memory bandwidth right here! but this op is 60% of the time, so we have to optimize E2E
        // ex: (    "firstPassTimeMs": 148, "secondPassTimeMs": 31,"ringPackingTimeMs": 63,)
        let intermediate = self.lwe_multiply_batched_with_db_packed::<K>(first_dim_queries_packed);
        let simplepir_resp_bytes = intermediate.len() / K * (lwe_q_prime_bits as usize) / 8;
        debug!("simplepir_resp_bytes {} bytes", simplepir_resp_bytes);
        let first_pass_time_ms = first_pass.elapsed().as_millis();
        debug!("First pass took {} us", first_pass.elapsed().as_micros());

        if let Some(ref mut m) = measurement {
            m.online.first_pass_time_ms = first_pass_time_ms as usize;
            m.online.simplepir_resp_bytes = simplepir_resp_bytes;
        }

        debug!("intermediate.len(): {}", intermediate.len());
        let mut second_pass_time_ms = 0;
        let mut ring_packing_time_ms = 0;
        let mut responses = Vec::new();
        for (intermediate_chunk, (packed_query_col, pack_pub_params_row_1s)) in intermediate
            .as_slice()
            .chunks(db_cols)
            .zip(second_dim_queries.iter())
        {
            let second_pass = Instant::now();
            let intermediate_cts_rescaled = intermediate_chunk
                .iter()
                .map(|x| rescale(*x as u64, lwe_params.modulus, lwe_q_prime))
                .collect::<Vec<_>>();
            assert_eq!(intermediate_cts_rescaled.len(), db_cols);
            debug!(
                "intermediate_cts_rescaled[0] = {}",
                intermediate_cts_rescaled[0]
            );

            let now = Instant::now();
            // modify the smaller_server db to include the intermediate values
            // let mut smaller_server_clone = smaller_server.clone();
            {
                // remember, this is stored in 'transposed' form
                // so it is out_cols x db_cols
                let smaller_db_mut: &mut [u16] = smaller_server.db_mut();
                for j in 0..db_cols {
                    // new value to write into the db
                    let val = intermediate_cts_rescaled[j];

                    for m in 0..blowup_factor.ceil() as usize {
                        // index in the transposed db
                        let out_idx = (special_offs + m) * db_cols + j;

                        // part of the value to write into the db
                        let val_part = ((val >> (m * pt_bits)) & ((1 << pt_bits) - 1)) as u16;

                        // assert_eq!(smaller_db_mut[out_idx], DoubleType::default());
                        smaller_db_mut[out_idx] = val_part;
                    }
                }
            }
            debug!("load secondary hint {} us", now.elapsed().as_micros());

            let now = Instant::now();
            {
                let blowup_factor_ceil = blowup_factor.ceil() as usize;

                let phase = Instant::now();
                // analogue to the answer_hint_ring call that generated H2 originally, now we generated A2 * T
                let secondary_hint = smaller_server.multiply_with_db_ring(
                    &pseudorandom_query_1,
                    special_offs..special_offs + blowup_factor_ceil,
                    SEED_1,
                );
                debug!(
                    "multiply_with_db_ring took: {} us",
                    phase.elapsed().as_micros()
                );
                assert_eq!(secondary_hint.len(), params.poly_len * blowup_factor_ceil);

                // now we write all the 
                for i in 0..params.poly_len {
                    for j in 0..blowup_factor_ceil {
                        let inp_idx = i * blowup_factor_ceil + j;//agree
                        let out_idx = i * out_rows + special_offs + j;//agree

                        // assert_eq!(hint_1_combined[out_idx], 0); // we no longer clone for each query, just overwrite
                        hint_1_combined[out_idx] = secondary_hint[inp_idx];
                    }
                }
            }
        debug!("compute secondary hint in {} us", now.elapsed().as_micros());

            assert_eq!(hint_1_combined.len(), params.poly_len * out_rows);

            // computing c2* [H1.extend(T)]
            let response: AlignedMemory64 = smaller_server.answer_query(packed_query_col);
            // the result is in Zq2 (2^56) space (CRT composed), raw form, just a constant polynomial
            
            second_pass_time_ms += second_pass.elapsed().as_millis();
            let ring_packing = Instant::now();
            let now = Instant::now();
            assert_eq!(response.len(), 1 * out_rows);

            // put all the new "random" RLWEs into NTT form (A2*decomp(T)) ~ 2
            let mut excess_cts = Vec::with_capacity(blowup_factor.ceil() as usize);
            for j in special_offs..special_offs + blowup_factor.ceil() as usize {
                let mut rlwe_ct = PolyMatrixRaw::zero(&params, 2, 1);

                // 'a' vector
                // put this in negacyclic order
                let mut poly = Vec::new();
                for k in 0..params.poly_len {
                    poly.push(hint_1_combined[k * out_rows + j]);
                }
                let nega = negacyclic_perm(&poly, 0, params.modulus);

                rlwe_ct.get_poly_mut(0, 0).copy_from_slice(&nega);

                excess_cts.push(rlwe_ct.ntt());
            }
            debug!("in between: {} us", now.elapsed().as_micros());

            // assert_eq!(pack_pub_params_row_1s[0].rows, 1);

            // this will run the full packing over the rho many squares, now we can get all the real responses
            // but we don't have the precomputed for the A2*decomp(T) RLWEs, so we just work with their b's and add that part of the RLWE in later (other_packed)
            let mut packed = pack_many_lwes(
                &params,
                &prepacked_lwe,
                &precomp,
                response.as_slice(),
                rho,
                &pack_pub_params_row_1s,
                &y_constants,
            );

            let now = Instant::now();
            let mut pack_pub_params = fake_pack_pub_params.clone();
            for i in 0..pack_pub_params.len() {
                let uncondensed = uncondense_matrix(params, &pack_pub_params_row_1s[i]);
                pack_pub_params[i].copy_into(&uncondensed, 1, 0);
            }
            debug!("uncondense pub params: {} us", now.elapsed().as_micros());
            let now = Instant::now();
            let other_packed =
                pack_using_single_with_offset(&params, &pack_pub_params, &excess_cts, special_offs);
            
            // wouldn't it be packed[-1], and then modulo special_offs % poly_len? 
            // well it turns out with our parameter choices in YPIR, we always pack into a single RLWE!!
            // the other_packed just rotated to that slot
            add_into(&mut packed[0], &other_packed);

            debug!(
                "pack_using_single_with_offset: {} us",
                now.elapsed().as_micros()
            );

            let now = Instant::now();
            let mut packed_mod_switched = Vec::with_capacity(packed.len());
            for (i, ct) in packed.iter().enumerate() {
                let res = ct.raw();

                let res_switched = res.switch(rlwe_q_prime_1, rlwe_q_prime_2);
                packed_mod_switched.push(res_switched);
            }
            debug!("switching: {} us", now.elapsed().as_micros());
            // debug!("Preprocessing pack in {} us", now.elapsed().as_micros());
            // debug!("");
            ring_packing_time_ms += ring_packing.elapsed().as_millis();

            // packed is blowup_factor ring ct's
            // these encode, contiguously [poly_len + 1, blowup_factor]
            // (and some padding)
            assert_eq!(packed.len(), rho);

            responses.push(packed_mod_switched);
        }
        debug!(
            "Total online time: {} us",
            online_phase.elapsed().as_micros()
        );
        debug!("");

        if let Some(ref mut m) = measurement {
            m.online.second_pass_time_ms = second_pass_time_ms as usize;
            m.online.ring_packing_time_ms = ring_packing_time_ms as usize;
        }

        responses
    }

    /// CUDA-accelerated online computation
    #[cfg(feature = "cuda")]
    pub fn perform_online_computation<const K: usize>(
        &self,
        offline_vals: &mut OfflinePrecomputedValues<'a>,
        first_dim_queries_packed: &[u32],
        second_dim_queries: &[(&[u64], &[PolyMatrixNTT<'a>])],
        mut measurement: Option<&mut Measurement>,
    ) -> Vec<Vec<Vec<u8>>> {
        debug!("CUDA-accelerated online computation");
        // Set up parameters (same as CPU version)
        let params = self.params;
        let db_cols = self.db_cols();

        // RLWE reduced moduli
        let rlwe_q_prime_1 = params.get_q_prime_1();
        let rlwe_q_prime_2 = params.get_q_prime_2();


        let mut query_q2_batch = Vec::with_capacity(second_dim_queries.len() * db_cols);
        let mut pack_pub_params_row_1s_batch: Vec<u64> = Vec::with_capacity(second_dim_queries.len() * params.poly_len_log2 * params.t_exp_left);

        for (packed_query_col, automorph_keys) in second_dim_queries.iter() {
            query_q2_batch.extend_from_slice(packed_query_col);
            assert_eq!(automorph_keys.len(), params.poly_len_log2);
            for key in *automorph_keys {
                let slc = key.as_slice();
                for col in 0..params.t_exp_left {
                    let start = col * 2 * params.poly_len;
                    pack_pub_params_row_1s_batch.extend_from_slice(&slc[start..start + params.poly_len]);
                }
            }
        }

        let batch_size = second_dim_queries.len();
        
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

    // generic function that returns a u8 or u16:
    pub fn db(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(
                self.db_buf_aligned.as_ptr() as *const T,
                self.db_buf_aligned.len() * 8 / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn db_mut(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.db_buf_aligned.as_ptr() as *mut T,
                self.db_buf_aligned.len() * 8 / std::mem::size_of::<T>(),
            )
        }
    }

    pub fn db_u16(&self) -> &[u16] {
        unsafe {
            std::slice::from_raw_parts(
                self.db_buf_aligned.as_ptr() as *const u16,
                self.db_buf_aligned.len() * 8 / std::mem::size_of::<u16>(),
            )
        }
    }

    pub fn db_u32(&self) -> &[u32] {
        unsafe {
            std::slice::from_raw_parts(
                self.db_buf_aligned.as_ptr() as *const u32,
                self.db_buf_aligned.len() * 8 / std::mem::size_of::<u32>(),
            )
        }
    }

    pub fn get_elem(&self, row: usize, col: usize) -> T {
        self.db()[col * self.db_rows_padded() + row] // stored transposed
    }

    pub fn get_row(&self, row: usize) -> Vec<T> {
        let db_cols = self.db_cols();
        let mut res = Vec::with_capacity(db_cols);
        for col in 0..db_cols {
            res.push(self.get_elem(row, col));
        }
        res
        // // convert to u8 contiguously
        // let mut res_u8 = Vec::with_capacity(db_cols * std::mem::size_of::<T>());
        // for &x in res.iter() {
        //     res_u8.extend_from_slice(&x.to_u64().to_le_bytes()[..std::mem::size_of::<T>()]);
        // }
        // res_u8
    }
}

#[cfg(not(target_feature = "avx2"))]
#[allow(non_camel_case_types)]
type __m512i = u64;

pub trait ToM512 {
    fn to_m512(self) -> __m512i;
}

#[cfg(target_feature = "avx512f")]
mod m512_impl {
    use super::*;

    impl ToM512 for *const u8 {
        #[inline(always)]
        fn to_m512(self) -> __m512i {
            unsafe { _mm512_cvtepu8_epi64(_mm_loadl_epi64(self as *const _)) }
        }
    }

    impl ToM512 for *const u16 {
        #[inline(always)]
        fn to_m512(self) -> __m512i {
            unsafe { _mm512_cvtepu16_epi64(_mm_load_si128(self as *const _)) }
        }
    }

    impl ToM512 for *const u32 {
        #[inline(always)]
        fn to_m512(self) -> __m512i {
            unsafe { _mm512_cvtepu32_epi64(_mm256_load_si256(self as *const _)) }
        }
    }
}

#[cfg(not(target_feature = "avx512f"))]
mod m512_impl {
    use super::*;

    impl ToM512 for *const u8 {
        #[inline(always)]
        fn to_m512(self) -> __m512i {
            self as __m512i
        }
    }

    impl ToM512 for *const u16 {
        #[inline(always)]
        fn to_m512(self) -> __m512i {
            self as __m512i
        }
    }

    impl ToM512 for *const u32 {
        #[inline(always)]
        fn to_m512(self) -> __m512i {
            self as __m512i
        }
    }
}

pub trait ToU64 {
    fn to_u64(self) -> u64;
}

impl ToU64 for u8 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl ToU64 for u16 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl ToU64 for u32 {
    fn to_u64(self) -> u64 {
        self as u64
    }
}

impl ToU64 for u64 {
    fn to_u64(self) -> u64 {
        self
    }
}
