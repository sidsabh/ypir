#[cfg(target_feature = "avx2")]
use std::arch::x86_64::*;
use std::{marker::PhantomData, ops::Range, time::Instant};

use log::debug;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use spiral_rs::aligned_memory::AlignedMemory64;
use spiral_rs::{arith::*, client::*, number_theory::invert_uint_mod, params::*, poly::*};

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
    // InspiRING packing fields
    pub packing_type: PackingType,
    pub packing_params: Option<PackParams<'a>>,
    pub precomp_inspir_vec: Option<Vec<PrecompInsPIR<'a>>>,
    pub offline_packing_keys: Option<OfflinePackingKeys<'a>>,
    #[cfg(feature = "cuda")]
    pub cuda_context: Option<std::sync::Arc<crate::cuda::OnlineComputeContext>>,
    #[cfg(feature = "cuda")]
    pub sp_cuda_context: Option<std::sync::Arc<crate::cuda::SPOnlineContext>>,
    #[cfg(feature = "cuda")]
    pub word_cuda_context: Option<std::sync::Arc<crate::cuda::WordOnlineContext>>,
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
    T: Sized + Copy + ToU64 + Default + Sync,
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

        // UNCOMMENT - JUST FOR SKIPPING FILLING
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
        let query = y_client.generate_query_impl(public_seed_idx, self.params.db_dim_1, PackingType::CDKS, 0, None, None);

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

    /// Word-based matrix-vector product for a single query.
    /// Returns the intermediate result (db_cols entries) after matmul in Z_{2^64},
    /// mod-switched to Z_Q, and CRT-packed into u64s.
    pub fn answer_query_word(&self, query: &[u64]) -> Vec<u64> {
        let params = self.params;
        let q = params.modulus;
        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_rows_padded = self.db_rows_padded();
        let db_cols = self.db_cols();
        let db = self.db();

        assert_eq!(query.len(), db_rows);

        let mut intermediate = vec![0u64; db_cols];
        for col in 0..db_cols {
            let mut acc: u64 = 0;
            for row in 0..db_rows {
                acc = acc.wrapping_add(
                    query[row].wrapping_mul(db[col * db_rows_padded + row].to_u64()),
                );
            }
            let val_q = Self::modswitch_word_to_q(acc, q);
            let crt0 = val_q % params.moduli[0];
            let crt1 = val_q % params.moduli[1];
            intermediate[col] = crt0 | (crt1 << 32);
        }

        intermediate
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
        packing: PackingType,
    ) -> OfflinePrecomputedValues {
        // Set up some parameters
        let params = self.params;
        assert!(self.ypir_params.is_simplepir);
        let db_cols = params.instances * params.poly_len;
        let num_rlwe_outputs = db_cols / params.poly_len;
        
        // Begin offline precomputation
        let simplepir_prep_time_ms: u128;

        #[cfg(feature = "cuda")]
        let hint_0: Vec<u64> = {
            let init_start = Instant::now();
            let gpu_ctx = self.init_hint_0_simplepir_gpu_context();
            let init_time = init_start.elapsed();

            let compute_start = Instant::now();
            let gpu_result = self.compute_hint_0_simplepir_gpu(&gpu_ctx);
            let compute_time = compute_start.elapsed();

            debug!("SimplePIR GPU init: {:?}, compute: {:?}", init_time, compute_time);
            simplepir_prep_time_ms = compute_time.as_millis();
            gpu_result
        };

        #[cfg(not(feature = "cuda"))]
        let hint_0: Vec<u64> = {
            let now = Instant::now();
            let result = self.answer_hint_ring(SEED_0, db_cols);
            simplepir_prep_time_ms = now.elapsed().as_millis();
            result
        };

        // hint_0 is poly_len x db_cols
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
        debug!("size of cached_ntt: {}", precomp.len() * precomp[0].1.len() * precomp[0].1[0].data.len() * 8 );
        debug!("Precomp in {} us", now.elapsed().as_micros());

        // InspiRING offline precomputation
        let gamma = params.poly_len; // always full packing for InspiRING
        let (packing_params, precomp_inspir_vec) = if packing == PackingType::InspiRING {
            let packing_params = PackParams::new(&params, gamma);
            let offline_packing_keys = OfflinePackingKeys::init_full(&packing_params, crate::scheme::W_SEED, crate::scheme::V_SEED);

            let now_precomp = Instant::now();
            let mut precomp_vec = Vec::with_capacity(num_rlwe_outputs);
            for i in 0..num_rlwe_outputs {
                let mut a_ct_tilde = Vec::new();
                for j in 0..gamma {
                    if j < prepacked_lwe[i].len() {
                        a_ct_tilde.push(prepacked_lwe[i][j].submatrix(0, 0, 1, 1));
                    }
                }

                let w_all = offline_packing_keys.w_all.as_ref().unwrap();
                let w_bar_all = offline_packing_keys.w_bar_all.as_ref().unwrap();
                let v_mask = offline_packing_keys.v_mask.as_ref().unwrap();

                let precomp_i = crate::packing::full_packing_with_preprocessing_offline(
                    &packing_params,
                    w_all,
                    w_bar_all,
                    v_mask,
                    &a_ct_tilde,
                );
                precomp_vec.push(precomp_i);
            }
            debug!("InspiRING precomp in {} us", now_precomp.elapsed().as_micros());
            (Some(packing_params), Some(precomp_vec))
        } else {
            (None, None)
        };

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
                packing_type: packing,
                packing_params,
                precomp_inspir_vec,
                offline_packing_keys: None,
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
                    256, // max_batch_size
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
                            // Extract CDKS tables in the order CUDA expects:
                            // table[0] → t=(1<<ell)+1, table[1] → t=(1<<(ell-1))+1, ..., table[ell-1] → t=3
                            for cur_ell in (1..=params.poly_len_log2).rev() {
                                let t = (1 << cur_ell) + 1;
                                let full_table_idx = (t - 1) / 2;
                                for &val in &tables[full_table_idx] {
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
                prepacked_lwe: vec![],
                fake_pack_pub_params: vec![],
                precomp: vec![],
                packing_type: packing,
                packing_params,
                precomp_inspir_vec,
                offline_packing_keys: None,
                cuda_context: None,
                sp_cuda_context,
                word_cuda_context: None,
            }
        }
    }

    /// Modswitch a single value from Z_{2^64} to Z_Q with rounding.
    fn modswitch_word_to_q(x: u64, q: u64) -> u64 {
        ((x as u128 * q as u128 + (1u128 << 63)) >> 64) as u64
    }

    /// Compute hint_0 using plain word-based matmul in Z_{2^64}, then modswitch to Z_Q.
    /// A is poly_len × db_rows, DB is column-major db_cols × db_rows_padded.
    /// Output: poly_len × db_cols values in Z_Q, stored row-major.
    pub fn compute_hint_0_word(&self) -> Vec<u64> {
        let poly_len = self.params.poly_len;
        let db_rows = 1 << (self.params.db_dim_1 + self.params.poly_len_log2);
        let db_rows_padded = self.db_rows_padded();
        let db_cols = self.db_cols();
        let q = self.params.modulus;

        debug!(
            "compute_hint_0_word: poly_len={}, db_rows={}, db_cols={}, total_muls={}",
            poly_len, db_rows, db_cols, poly_len as u64 * db_rows as u64 * db_cols as u64
        );

        let now = Instant::now();
        let a = generate_pseudorandom_matrix_word(SEED_0, poly_len, db_rows);
        debug!("  A matrix generated in {} ms", now.elapsed().as_millis());

        let now = Instant::now();
        let mut hint_0 = vec![0u64; poly_len * db_cols];
        let db = self.db();

        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let chunk_size = (db_cols + num_threads - 1) / num_threads;

        // Parallelize over columns: each thread computes a contiguous range of columns.
        // Use col-major scratch buffer so each thread writes to its own contiguous chunk.
        let mut hint_0_col_major = vec![0u64; db_cols * poly_len];

        std::thread::scope(|s| {
            let chunks: Vec<&mut [u64]> = hint_0_col_major.chunks_mut(chunk_size * poly_len).collect();
            let mut handles = Vec::new();
            for (chunk_idx, chunk) in chunks.into_iter().enumerate() {
                let col_start = chunk_idx * chunk_size;
                let col_end = (col_start + chunk_size).min(db_cols);
                let a_ref = &a;
                let db_ref = db;
                handles.push(s.spawn(move || {
                    for col in col_start..col_end {
                        let local_col = col - col_start;
                        for i in 0..poly_len {
                            let mut acc: u64 = 0;
                            for j in 0..db_rows {
                                acc = acc.wrapping_add(
                                    a_ref[i * db_rows + j]
                                        .wrapping_mul(db_ref[col * db_rows_padded + j].to_u64()),
                                );
                            }
                            chunk[local_col * poly_len + i] =
                                Self::modswitch_word_to_q(acc, q);
                        }
                    }
                }));
            }
            for h in handles {
                h.join().unwrap();
            }
        });

        // Transpose col-major (db_cols × poly_len) to row-major (poly_len × db_cols)
        for col in 0..db_cols {
            for i in 0..poly_len {
                hint_0[i * db_cols + col] = hint_0_col_major[col * poly_len + i];
            }
        }

        debug!(
            "  Matmul ({} threads) + modswitch in {} ms",
            num_threads,
            now.elapsed().as_millis()
        );

        hint_0
    }

    /// Offline precomputation for word-based SimplePIR.
    /// Same structure as perform_offline_precomputation_simplepir but uses plain word matmul.
    pub fn perform_offline_precomputation_simplepir_word(
        &self,
        measurement: Option<&mut Measurement>,
        packing: PackingType,
    ) -> OfflinePrecomputedValues {
        let params = self.params;
        assert!(self.ypir_params.is_simplepir);
        let db_cols = params.instances * params.poly_len;
        let num_rlwe_outputs = db_cols / params.poly_len;

        let simplepir_prep_time_ms: u128;

        // GPU path: hint_0 is computed on GPU (already CRT-packed)
        #[cfg(feature = "cuda")]
        let hint_0_packed: Vec<u64> = {
            let init_start = Instant::now();
            let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
            let db_rows_padded = self.db_rows_padded();
            let a = generate_pseudorandom_matrix_word(SEED_0, params.poly_len, db_rows);
            let gpu_ctx = crate::cuda::WordOfflineContext::new(
                self.db_u16(),
                &a,
                db_rows as u32,
                db_rows_padded as u32,
                db_cols as u32,
                params.poly_len as u32,
                params.modulus,
                params.moduli[0],
                params.moduli[1],
                packing != PackingType::InspiRING,
            ).expect("Failed to initialize Word offline GPU context");
            let init_time = init_start.elapsed();

            let compute_start = Instant::now();
            let result = gpu_ctx.compute_hint_0().expect("Word GPU hint_0 failed");
            let compute_time = compute_start.elapsed();

            debug!("Word offline GPU init: {:?}, compute: {:?}", init_time, compute_time);
            simplepir_prep_time_ms = compute_time.as_millis();
            result
        };

        // CPU path: compute hint_0 then CRT-pack
        #[cfg(not(feature = "cuda"))]
        let hint_0_packed: Vec<u64> = {
            let now = Instant::now();
            let mut hint_0 = self.compute_hint_0_word();
            // CDKS packing multiplies a-part by N; pre-multiply by inv_N to cancel.
            // InspiRING normalization happens via mod_inv_poly during offline precomp, so skip.
            if packing != PackingType::InspiRING {
                let inv_n = invert_uint_mod(params.poly_len as u64, params.modulus).unwrap();
                for val in hint_0.iter_mut() {
                    *val = multiply_uint_mod(*val, inv_n, params.modulus);
                }
            }
            simplepir_prep_time_ms = now.elapsed().as_millis();
            hint_0
        };

        if let Some(measurement) = measurement {
            measurement.offline.simplepir_prep_time_ms = simplepir_prep_time_ms as usize;
        }

        // With apply_inv_n=false for InspiRING, the GPU hint already has the correct scaling.

        let now = Instant::now();
        let y_constants = generate_y_constants(&params);

        let combined = [&hint_0_packed[..], &vec![0u64; db_cols]].concat();
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

        // InspiRING offline precomputation
        let gamma = params.poly_len;
        let (inspir_packing_params, inspir_precomp_vec) = if packing == PackingType::InspiRING {
            let packing_params = PackParams::new(&params, gamma);
            let offline_packing_keys = OfflinePackingKeys::init_full(&packing_params, crate::scheme::W_SEED, crate::scheme::V_SEED);

            let now_precomp = Instant::now();
            let mut precomp_vec = Vec::with_capacity(num_rlwe_outputs);
            for i in 0..num_rlwe_outputs {
                let mut a_ct_tilde = Vec::new();
                for j in 0..gamma {
                    if j < prepacked_lwe[i].len() {
                        a_ct_tilde.push(prepacked_lwe[i][j].submatrix(0, 0, 1, 1));
                    }
                }

                let w_all = offline_packing_keys.w_all.as_ref().unwrap();
                let w_bar_all = offline_packing_keys.w_bar_all.as_ref().unwrap();
                let v_mask = offline_packing_keys.v_mask.as_ref().unwrap();

                let precomp_i = crate::packing::full_packing_with_preprocessing_offline(
                    &packing_params,
                    w_all,
                    w_bar_all,
                    v_mask,
                    &a_ct_tilde,
                );
                precomp_vec.push(precomp_i);
            }
            debug!("InspiRING precomp in {} us", now_precomp.elapsed().as_micros());
            (Some(packing_params), Some(precomp_vec))
        } else {
            (None, None)
        };

        #[cfg(not(feature = "cuda"))]
        {
            OfflinePrecomputedValues {
                hint_0: hint_0_packed,
                hint_1: vec![],
                pseudorandom_query_1: vec![],
                y_constants,
                smaller_server: None,
                prepacked_lwe,
                fake_pack_pub_params,
                precomp,
                packing_type: packing,
                packing_params: inspir_packing_params,
                precomp_inspir_vec: inspir_precomp_vec,
                offline_packing_keys: None,
            }
        }

        #[cfg(feature = "cuda")]
        {
            let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
            let db_rows_padded = self.db_rows_padded();

            // Initialize Word CUDA online context
            let word_cuda_context = {
                debug!("Initializing Word CUDA online context...");
                match crate::cuda::WordOnlineContext::new(
                    self.db_u16(),
                    db_rows,
                    db_rows_padded,
                    db_cols,
                    params.t_exp_left,
                    params.get_q_prime_1(),
                    params.get_q_prime_2(),
                    params,
                    256, // max_batch_size
                    packing != PackingType::InspiRING,
                ) {
                    Ok(ctx) => {
                        if packing == PackingType::InspiRING {
                            // InspiRING: upload bold_t, bold_t_bar, bold_t_hat, a_hat
                            let inspir_vec = inspir_precomp_vec.as_ref().unwrap();
                            let mut bold_t_flat = Vec::new();
                            let mut bold_t_bar_flat = Vec::new();
                            let mut bold_t_hat_flat = Vec::new();
                            let mut a_hat_flat = Vec::new();

                            for precomp_i in inspir_vec {
                                for row in 0..precomp_i.bold_t_condensed.rows {
                                    for col in 0..precomp_i.bold_t_condensed.cols {
                                        let poly = precomp_i.bold_t_condensed.get_poly(row, col);
                                        bold_t_flat.extend_from_slice(&poly[..params.poly_len]);
                                    }
                                }
                                for row in 0..precomp_i.bold_t_bar_condensed.rows {
                                    for col in 0..precomp_i.bold_t_bar_condensed.cols {
                                        let poly = precomp_i.bold_t_bar_condensed.get_poly(row, col);
                                        bold_t_bar_flat.extend_from_slice(&poly[..params.poly_len]);
                                    }
                                }
                                for row in 0..precomp_i.bold_t_hat_condensed.rows {
                                    for col in 0..precomp_i.bold_t_hat_condensed.cols {
                                        let poly = precomp_i.bold_t_hat_condensed.get_poly(row, col);
                                        bold_t_hat_flat.extend_from_slice(&poly[..params.poly_len]);
                                    }
                                }
                                a_hat_flat.extend_from_slice(precomp_i.a_hat.get_poly(0, 0));
                            }

                            let num_iter = params.poly_len / 2 - 1;

                            // Flatten permutation tables and gen_pows for GPU expand
                            let pp = inspir_packing_params.as_ref().unwrap();
                            let tables_flat: Vec<u32> = pp.tables.iter()
                                .flat_map(|t| t.iter().map(|&v| v as u32))
                                .collect();
                            let num_tables = pp.tables.len();
                            let gen_pows_flat: Vec<u32> = pp.gen_pows[..num_iter]
                                .iter()
                                .map(|&v| v as u32)
                                .collect();

                            ctx.init_packing_inspir(
                                &bold_t_flat,
                                &bold_t_bar_flat,
                                &bold_t_hat_flat,
                                &a_hat_flat,
                                num_iter,
                                &tables_flat,
                                num_tables,
                                &gen_pows_flat,
                            );
                        } else {
                            // CDKS: flatten packing data (same as SP path)
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
                                // Extract CDKS tables in the order CUDA expects:
                                // table[0] → t=(1<<ell)+1, table[1] → t=(1<<(ell-1))+1, ..., table[ell-1] → t=3
                                for cur_ell in (1..=params.poly_len_log2).rev() {
                                    let t = (1 << cur_ell) + 1;
                                    let full_table_idx = (t - 1) / 2;
                                    for &val in &tables[full_table_idx] {
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
                        }

                        debug!("Word CUDA online context initialized");
                        Some(std::sync::Arc::new(ctx))
                    }
                    Err(e) => {
                        log::warn!("Failed to create Word CUDA context: {}", e);
                        None
                    }
                }
            };

            OfflinePrecomputedValues {
                hint_0: hint_0_packed,
                hint_1: vec![],
                pseudorandom_query_1: vec![],
                y_constants,
                smaller_server: None,
                prepacked_lwe: vec![],
                fake_pack_pub_params: vec![],
                precomp: vec![],
                packing_type: packing,
                packing_params: inspir_packing_params,
                precomp_inspir_vec: inspir_precomp_vec,
                offline_packing_keys: None,
                cuda_context: None,
                sp_cuda_context: None,
                word_cuda_context,
            }
        }
    }

    pub fn perform_offline_precomputation(
        &self,
        measurement: Option<&mut Measurement>,
        packing: PackingType,
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

        // now we just add the last row to store the DoublePIR response
        let combined = [&hint_1[..], &vec![0u64; out_rows]].concat(); // stored in row major
        assert_eq!(combined.len(), out_rows * (params.poly_len + 1)); // full DoublePIR response, 0s everywhere besides H2

        // get the rho many RLWE squares, negacyclic perms them, NTT form
        let prepacked_lwe = prep_pack_many_lwes(&params, &combined, rho);
        assert_eq!(prepacked_lwe.len(), rho);
        assert_eq!(prepacked_lwe[0].len(), params.poly_len);

        let gamma = params.poly_len;
        let now = Instant::now();
        let mut y_constants = (Vec::new(), Vec::new());
        let mut fake_pack_pub_params = Vec::new();
        let mut precomp: Precomp = Vec::new();
        let mut packing_params_opt: Option<PackParams> = None;
        let mut precomp_inspir_vec_opt: Option<Vec<PrecompInsPIR>> = None;
        let mut offline_packing_keys_opt: Option<OfflinePackingKeys> = None;

        match packing {
            PackingType::InspiRING => {
                let packing_params = PackParams::new(&params, gamma);
                let offline_packing_keys = OfflinePackingKeys::init_full(&packing_params, crate::scheme::W_SEED, crate::scheme::V_SEED);

                let mut precomp_vec = Vec::with_capacity(rho);
                for i in 0..rho {
                    let mut a_ct_tilde = Vec::new();
                    for j in 0..gamma {
                        if j < prepacked_lwe[i].len() {
                            a_ct_tilde.push(prepacked_lwe[i][j].submatrix(0, 0, 1, 1));
                        }
                    }

                    let w_all = offline_packing_keys.w_all.as_ref().unwrap();
                    let w_bar_all = offline_packing_keys.w_bar_all.as_ref().unwrap();
                    let v_mask = offline_packing_keys.v_mask.as_ref().unwrap();

                    let precomp_i = crate::packing::full_packing_with_preprocessing_offline_without_rotations(
                        &packing_params, w_all, w_bar_all, v_mask, &a_ct_tilde,
                    );
                    precomp_vec.push(precomp_i);
                }
                debug!("InspiRING DoublePIR precomp in {} us", now.elapsed().as_micros());
                packing_params_opt = Some(packing_params);
                precomp_inspir_vec_opt = Some(precomp_vec);
                offline_packing_keys_opt = Some(offline_packing_keys);
            },
            _ => {
                y_constants = generate_y_constants(&params);
                fake_pack_pub_params = generate_fake_pack_pub_params(&params);
                for i in 0..prepacked_lwe.len() {
                    let tup: (PolyMatrixNTT<'_>, Vec<PolyMatrixNTT<'_>>, Vec<Vec<usize>>) = precompute_pack(
                        params,
                        params.poly_len_log2,
                        &prepacked_lwe[i],
                        &fake_pack_pub_params,
                        &y_constants,
                    );
                    precomp.push(tup);
                }
                debug!("CDKS Precomp in {} us", now.elapsed().as_micros());
            },
        }

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
                    precomp,
                    packing_type: packing,
                    packing_params: packing_params_opt,
                    precomp_inspir_vec: precomp_inspir_vec_opt,
                    offline_packing_keys: offline_packing_keys_opt,
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
                    256,                       // max_batch_size
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
                    // Extract CDKS tables in the order CUDA expects
                    for cur_ell in (1..=params.poly_len_log2).rev() {
                        let t = (1 << cur_ell) + 1;
                        let full_table_idx = (t - 1) / 2;
                        for &val in &tables[full_table_idx] {
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
                packing_type: packing,
                packing_params: None,
                precomp_inspir_vec: None,
                offline_packing_keys: None,
                cuda_context,
                sp_cuda_context: None,
                word_cuda_context: None,
            }
        }
    }

    /// Perform SimplePIR-style YPIR (CPU path, supports batching and InspiRING/CDKS dispatch)
    #[cfg(not(feature = "cuda"))]
    pub fn perform_online_computation_simplepir(
        &self,
        first_dim_queries_packed: &[&[u64]],
        offline_vals: &OfflinePrecomputedValues<'a>,
        packing_keys: &mut [PackingKeys<'a>],
        mut measurement: Option<&mut Measurement>,
    ) -> Vec<Vec<Vec<u8>>> {
        assert!(self.ypir_params.is_simplepir);

        let params = self.params;
        let y_constants = &offline_vals.y_constants;
        let prepacked_lwe = &offline_vals.prepacked_lwe;
        let precomp = &offline_vals.precomp;

        let rlwe_q_prime_1 = params.get_q_prime_1();
        let rlwe_q_prime_2 = params.get_q_prime_2();

        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_cols = params.instances * params.poly_len;
        let num_rlwe_outputs = db_cols / params.poly_len;
        let gamma = params.poly_len;

        let batch_size = first_dim_queries_packed.len();
        assert_eq!(first_dim_queries_packed[0].len(), params.db_rows_padded());

        // Step 1: Matmul for all queries
        let first_pass = Instant::now();
        debug!("Performing mul ({} queries)...", batch_size);
        let mut all_intermediates = vec![vec![0u64; db_cols]; batch_size];
        for (batch, query) in first_dim_queries_packed.iter().enumerate() {
            let mut intermediate = AlignedMemory64::new(db_cols);
            fast_batched_dot_product_avx512::<1, T>(
                &params,
                intermediate.as_mut_slice(),
                query,
                db_rows,
                self.db(),
                db_rows,
                db_cols,
            );
            all_intermediates[batch] = intermediate.as_slice().to_vec();
        }
        debug!("Done w mul...");
        let first_pass_time_ms = first_pass.elapsed().as_millis();
        if let Some(ref mut m) = measurement {
            m.online.first_pass_time_ms = first_pass_time_ms as usize;
        }

        // Step 2: Packing dispatch per client
        let ring_packing = Instant::now();
        let mut all_responses = Vec::with_capacity(batch_size);
        for (batch, intermediate) in all_intermediates.iter().enumerate() {
            let pk = &mut packing_keys[batch];

            let packed_mod_switched = match offline_vals.packing_type {
                PackingType::InspiRING => {
                    let packing_params = offline_vals.packing_params.as_ref().unwrap();
                    let precomp_inspir_vec = offline_vals.precomp_inspir_vec.as_ref().unwrap();

                    pk.expand(packing_params);
                    let packed = pack_many_lwes_inspir(
                        packing_params,
                        precomp_inspir_vec,
                        intermediate,
                        pk,
                        gamma,
                    );

                    let mut switched = Vec::with_capacity(packed.len());
                    for p in packed.iter() {
                        switched.push(p.switch_and_keep(rlwe_q_prime_1, rlwe_q_prime_2, gamma));
                    }
                    switched
                },
                _ => {
                    let packed = pack_many_lwes(
                        &params,
                        &prepacked_lwe,
                        &precomp,
                        intermediate,
                        num_rlwe_outputs,
                        &pk.pack_pub_params_row_1s,
                        &y_constants,
                    );

                    let mut switched = Vec::with_capacity(packed.len());
                    for ct in packed.iter() {
                        let res = ct.raw();
                        switched.push(res.switch(rlwe_q_prime_1, rlwe_q_prime_2));
                    }
                    switched
                },
            };
            all_responses.push(packed_mod_switched);
        }
        if let Some(m) = measurement {
            m.online.ring_packing_time_ms = ring_packing.elapsed().as_millis() as usize;
        }

        all_responses
    }

    #[cfg(feature = "cuda")]
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
    #[cfg(feature = "cuda")]
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

    /// Online computation for word-based SimplePIR (CPU path).
    /// Takes raw u64 queries (NOT CRT-packed), does plain matmul in Z_{2^64},
    /// modswitches to Z_Q, CRT-packs, and feeds into existing packing.
    #[cfg(not(feature = "cuda"))]
    pub fn perform_online_computation_simplepir_word(
        &self,
        word_queries: &[&[u64]],
        offline_vals: &OfflinePrecomputedValues<'a>,
        packing_keys: &mut [PackingKeys<'a>],
        mut measurement: Option<&mut Measurement>,
    ) -> Vec<Vec<Vec<u8>>> {
        assert!(self.ypir_params.is_simplepir);

        let params = self.params;
        let q = params.modulus;
        let y_constants = &offline_vals.y_constants;
        let prepacked_lwe = &offline_vals.prepacked_lwe;
        let precomp = &offline_vals.precomp;

        let rlwe_q_prime_1 = params.get_q_prime_1();
        let rlwe_q_prime_2 = params.get_q_prime_2();

        let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
        let db_rows_padded = self.db_rows_padded();
        let db_cols = params.instances * params.poly_len;
        let num_rlwe_outputs = db_cols / params.poly_len;
        let gamma = params.poly_len;

        let batch_size = word_queries.len();
        assert_eq!(word_queries[0].len(), db_rows);

        let first_pass = Instant::now();
        debug!("Performing word matmul ({} queries)...", batch_size);

        let db = self.db();

        // Compute intermediate per batch item: matmul in Z_{2^64}, modswitch to Z_Q
        let mut all_intermediates = vec![vec![0u64; db_cols]; batch_size];
        for (batch, query) in word_queries.iter().enumerate() {
            let intermediate = &mut all_intermediates[batch];
            for col in 0..db_cols {
                let mut acc: u64 = 0;
                for row in 0..db_rows {
                    acc = acc.wrapping_add(
                        query[row].wrapping_mul(db[col * db_rows_padded + row].to_u64()),
                    );
                }
                intermediate[col] = Self::modswitch_word_to_q(acc, q);
            }
        }

        // Pre-multiply all intermediate values by inv_N mod Q.
        // CDKS packing multiplies b-values by N (lines 594-601 of packing.rs).
        // N * inv_N = 1 mod Q, so the query noise is NOT amplified by N.
        // This mirrors the ring path's pre-division by N in generate_query_impl.
        // InspiRING does NOT pre-divide by N (its normalization happens via mod_inv_poly
        // on the mask side during offline precomp), so skip this for InspiRING.
        if offline_vals.packing_type != PackingType::InspiRING {
            let inv_n = invert_uint_mod(params.poly_len as u64, q).unwrap();
            for intermediate in all_intermediates.iter_mut() {
                for val in intermediate.iter_mut() {
                    *val = multiply_uint_mod(*val, inv_n, q);
                }
            }
        }

        debug!("Done w word matmul...");
        let first_pass_time_ms = first_pass.elapsed().as_millis();
        if let Some(ref mut m) = measurement {
            m.online.first_pass_time_ms = first_pass_time_ms as usize;
        }

        let ring_packing = Instant::now();
        let mut all_responses = Vec::with_capacity(batch_size);
        for (batch, intermediate) in all_intermediates.iter().enumerate() {
            let pk = &mut packing_keys[batch];

            let packed_mod_switched = match offline_vals.packing_type {
                PackingType::InspiRING => {
                    let packing_params = offline_vals.packing_params.as_ref().unwrap();
                    let precomp_inspir_vec = offline_vals.precomp_inspir_vec.as_ref().unwrap();

                    pk.expand(packing_params);
                    let packed = pack_many_lwes_inspir(
                        packing_params,
                        precomp_inspir_vec,
                        intermediate,
                        pk,
                        gamma,
                    );

                    let mut switched = Vec::with_capacity(packed.len());
                    for p in packed.iter() {
                        switched.push(p.switch_and_keep(rlwe_q_prime_1, rlwe_q_prime_2, gamma));
                    }
                    switched
                },
                _ => {
                    let packed = pack_many_lwes(
                        &params,
                        &prepacked_lwe,
                        &precomp,
                        intermediate,
                        num_rlwe_outputs,
                        &pk.pack_pub_params_row_1s,
                        &y_constants,
                    );
                    debug!("Packed batch {}...", batch);

                    let mut switched = Vec::with_capacity(packed.len());
                    for ct in packed.iter() {
                        let res = ct.raw();
                        switched.push(res.switch(rlwe_q_prime_1, rlwe_q_prime_2));
                    }
                    switched
                },
            };
            all_responses.push(packed_mod_switched);
        }
        if let Some(m) = measurement {
            m.online.ring_packing_time_ms = ring_packing.elapsed().as_millis() as usize;
        }

        all_responses
    }

    #[cfg(not(feature = "cuda"))]
    pub fn perform_online_computation<const K: usize>(
        &self,
        offline_vals: &mut OfflinePrecomputedValues<'a>,
        first_dim_queries_packed: &[u32],
        second_dim_query_cols: &[&[u64]],
        packing_keys: &mut [PackingKeys<'a>],
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

        let blowup_factor = lwe_q_prime_bits as f64 / pt_bits as f64;
        debug!("blowup_factor: {}", blowup_factor);

        // The starting index of the final value (the '1' in lwe_params.n + 1)
        let special_offs =
            ((lwe_params.n * lwe_q_prime_bits) as f64 / pt_bits as f64).ceil() as usize;

        // Parameters for the second round (the "DoublePIR" round)
        let mut smaller_params = params.clone();
        smaller_params.db_dim_1 = params.db_dim_2;
        smaller_params.db_dim_2 = ((blowup_factor * (lwe_params.n + 1) as f64)
            / params.poly_len as f64)
            .log2()
            .ceil() as usize;

        let out_rows = 1 << (smaller_params.db_dim_2 + params.poly_len_log2);
        let rho = 1 << smaller_params.db_dim_2;
        let gamma = params.poly_len;
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
        let packing_type = offline_vals.packing_type;

        // Begin online computation

        let online_phase = Instant::now();
        let first_pass = Instant::now();
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
        for (batch, intermediate_chunk) in intermediate
            .as_slice()
            .chunks(db_cols)
            .enumerate()
        {
            let packed_query_col = second_dim_query_cols[batch];
            let pk = &mut packing_keys[batch];

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
            {
                let smaller_db_mut: &mut [u16] = smaller_server.db_mut();
                for j in 0..db_cols {
                    let val = intermediate_cts_rescaled[j];
                    for m in 0..blowup_factor.ceil() as usize {
                        let out_idx = (special_offs + m) * db_cols + j;
                        let val_part = ((val >> (m * pt_bits)) & ((1 << pt_bits) - 1)) as u16;
                        smaller_db_mut[out_idx] = val_part;
                    }
                }
            }
            debug!("load secondary hint {} us", now.elapsed().as_micros());

            let now = Instant::now();
            {
                let blowup_factor_ceil = blowup_factor.ceil() as usize;

                let phase = Instant::now();
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

                for i in 0..params.poly_len {
                    for j in 0..blowup_factor_ceil {
                        let inp_idx = i * blowup_factor_ceil + j;
                        let out_idx = i * out_rows + special_offs + j;
                        hint_1_combined[out_idx] = secondary_hint[inp_idx];
                    }
                }
            }
            debug!("compute secondary hint in {} us", now.elapsed().as_micros());

            assert_eq!(hint_1_combined.len(), params.poly_len * out_rows);

            let response: AlignedMemory64 = smaller_server.answer_query(packed_query_col);

            second_pass_time_ms += second_pass.elapsed().as_millis();
            let ring_packing = Instant::now();
            let now = Instant::now();
            assert_eq!(response.len(), 1 * out_rows);

            let blowup_factor_ceil = blowup_factor.ceil() as usize;

            // Build excess CTs (body: query-dependent a-vectors at special_offs)
            let mut excess_cts_raw = Vec::with_capacity(blowup_factor_ceil);
            for j in special_offs..special_offs + blowup_factor_ceil {
                let mut rlwe_ct = PolyMatrixRaw::zero(&params, 2, 1);

                let mut poly = Vec::new();
                for k in 0..params.poly_len {
                    poly.push(hint_1_combined[k * out_rows + j]);
                }
                let nega = negacyclic_perm(&poly, 0, params.modulus);
                rlwe_ct.get_poly_mut(0, 0).copy_from_slice(&nega);
                // For InspiRING NoPacking body, we need the b-value in the CT
                if packing_type == PackingType::InspiRING {
                    rlwe_ct.get_poly_mut(1, 0)[0] = response[j];
                }
                // For CDKS, row 1 stays zero — b-values are handled via pack_many_lwes on the full response

                excess_cts_raw.push(rlwe_ct);
            }
            debug!("excess cts: {} us", now.elapsed().as_micros());

            // Dispatch: mask packing + body packing + modulus switching based on packing type
            let packed_mod_switched = match packing_type {
                PackingType::InspiRING => {
                    let packing_params = offline_vals.packing_params.as_ref().unwrap();
                    let precomp_inspir_vec = offline_vals.precomp_inspir_vec.as_ref().unwrap();

                    // Mask packing: only the first special_offs b-values (offline-known a-vectors)
                    let packed = pack_many_lwes_inspir_without_rotations(
                        packing_params,
                        precomp_inspir_vec,
                        &response.as_slice()[..special_offs],
                        pk,
                        gamma,
                    );
                    let num_mask_cts = packed.len();

                    // Modulus switch mask CTs with switch_and_keep (InspiRING packing)
                    let now = Instant::now();
                    let mut packed_mod_switched = Vec::with_capacity(num_mask_cts + excess_cts_raw.len());
                    for ct in packed.iter() {
                        packed_mod_switched.push(ct.switch_and_keep(rlwe_q_prime_1, rlwe_q_prime_2, gamma));
                    }
                    // Body: NoPacking — excess CTs use plain switch
                    for ct in &excess_cts_raw {
                        packed_mod_switched.push(ct.switch(rlwe_q_prime_1, rlwe_q_prime_2));
                    }
                    debug!("switching: {} us", now.elapsed().as_micros());

                    packed_mod_switched
                },
                _ => {
                    // CDKS path: pack all response values, then add body via automorphisms
                    let excess_cts_ntt: Vec<PolyMatrixNTT> = excess_cts_raw.iter()
                        .map(|ct| ct.ntt())
                        .collect();

                    let mut packed = pack_many_lwes(
                        &params,
                        &prepacked_lwe,
                        &precomp,
                        response.as_slice(),
                        rho,
                        &pk.pack_pub_params_row_1s,
                        &y_constants,
                    );

                    let now = Instant::now();
                    let mut pack_pub_params = fake_pack_pub_params.clone();
                    for i in 0..pack_pub_params.len() {
                        let uncondensed = uncondense_matrix(params, &pk.pack_pub_params_row_1s[i]);
                        pack_pub_params[i].copy_into(&uncondensed, 1, 0);
                    }
                    debug!("uncondense pub params: {} us", now.elapsed().as_micros());
                    let other_packed =
                        pack_using_single_with_offset(&params, &pack_pub_params, &excess_cts_ntt, special_offs);

                    add_into(&mut packed[0], &other_packed);
                    debug!(
                        "pack_using_single_with_offset: {} us",
                        now.elapsed().as_micros()
                    );

                    let now = Instant::now();
                    let packed_mod_switched = packed.into_iter()
                        .map(|p| p.raw().switch(rlwe_q_prime_1, rlwe_q_prime_2))
                        .collect::<Vec<_>>();
                    debug!("switching: {} us", now.elapsed().as_micros());

                    packed_mod_switched
                },
            };
            ring_packing_time_ms += ring_packing.elapsed().as_millis();

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
        second_dim_query_cols: &[&[u64]],
        packing_keys: &mut [PackingKeys<'a>],
        mut measurement: Option<&mut Measurement>,
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

#[cfg(not(target_feature = "avx512f"))]
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
