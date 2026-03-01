use std::time::Instant;

use log::debug;
use rand::{thread_rng, Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;

use spiral_rs::aligned_memory::AlignedMemory64;
use spiral_rs::arith::rescale;
use spiral_rs::poly::{PolyMatrix, PolyMatrixRaw};
use spiral_rs::{client::*, params::*};

use crate::bits::{read_bits, u64s_to_contiguous_bytes};
use crate::modulus_switch::ModulusSwitch;
use crate::noise_analysis::YPIRSchemeParams;
use crate::packing::{
    condense_matrix, PackingKeys, PackingType,
};

use super::{client::*, lwe::LWEParams, measurement::*, params::*, server::*};

pub const STATIC_PUBLIC_SEED: [u8; 32] = [0u8; 32];
pub const SEED_0: u8 = 0;
pub const SEED_1: u8 = 1;

pub const STATIC_SEED_2: [u8; 32] = [
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

pub const W_SEED: [u8; 32] = [
    7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
];
pub const V_SEED: [u8; 32] = [
    8, 8, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7,
];


macro_rules! dispatch_const {
    ($val:expr, [$($n:literal),+], |$name:ident| $body:expr) => {
        match $val {
            $($n => { const $name: usize = $n; $body },)+
            _ => panic!("Unsupported value: {}", $val),
        }
    };
}

pub fn run_ypir_batched(
    num_items: usize,
    item_size_bits: usize,
    num_clients: usize,
    is_simplepir: bool,
    word: bool,
    trials: usize,
    packing: PackingType,
) -> Measurement {
    #[cfg(feature = "cuda")]
    assert!(
        packing != PackingType::InspiRING || word,
        "GPU InspiRING packing only supported for word SimplePIR. Use CDKS packing or run on CPU."
    );

    let params = if is_simplepir || word {
        params_for_scenario_simplepir(num_items, item_size_bits)
    } else {
        assert!(item_size_bits <= 8, "Standard YPIR supports item sizes of 1-8 bits.");
        params_for_scenario(num_items, item_size_bits)
    };

    // InspiRING batching now supported: offline precomp shared across clients,
    // PackingKeys per-client, server loops over clients.

    // Q: why is clients as a generic instead of a parameter?
    // A: CPU YPIR runs the DB-Q1 product using AVX. the supported number of clients: 1, 2, 4, or 8
    #[cfg(feature = "cuda")]
    let measurement = dispatch_const!(num_clients, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 128, 256], |N| {
        run_ypir_on_params::<N>(params, is_simplepir, word, trials, packing)
    });

    #[cfg(not(feature = "cuda"))]
    let measurement = dispatch_const!(num_clients, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], |N| {
        run_ypir_on_params::<N>(params, is_simplepir, word, trials, packing)
    });

    debug!("{:#?}", measurement);
    let db_size_bytes = (num_items * item_size_bits + 7) / 8;
    println!(
        "Hint Prep. Throughput {:.2} GB/sec",
        (db_size_bytes as f64) / ((measurement.offline.simplepir_prep_time_ms) as f64 / 1000.0) / (1 << 30) as f64
    );
    println!(
        "Throughput: {:.2} GB/sec",
        (num_clients * db_size_bytes) as f64 / ((measurement.online.server_time_ms) as f64 / 1000.0) / (1 << 30) as f64
    );
    measurement
}

pub trait Sample {
    fn sample() -> Self;
}

impl Sample for u8 {
    fn sample() -> Self {
        fastrand::u8(..)
    }
}

impl Sample for u16 {
    fn sample() -> Self {
        fastrand::u16(..)
    }
}

pub fn run_simple_ypir_on_params<const K: usize>(params: Params, word: bool, trials: usize, packing: PackingType) -> Measurement {

    let is_simplepir = true;
    let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
    let db_rows_padded = params.db_rows_padded();
    let db_cols = params.instances * params.poly_len;

    let mut rng = thread_rng();

    // RLWE reduced moduli
    let rlwe_q_prime_1 = params.get_q_prime_1();
    let rlwe_q_prime_2 = params.get_q_prime_2();

    let num_rlwe_outputs = db_cols / params.poly_len;
    let gamma = params.poly_len; // always full packing for InspiRING

    // --

    let now = Instant::now();
    type T = u16;
    let pt_iter = std::iter::repeat_with(|| (T::sample() as u64 % params.pt_modulus) as T);
    let y_server = YServer::<T>::new(&params, pt_iter, is_simplepir, false, true);
    debug!("Created server in {} us", now.elapsed().as_micros());
    debug!(
        "Database of {} bytes",
        y_server.db().len() * (params.pt_modulus as f64).log2().ceil() as usize / 8
    );
    assert_eq!(y_server.db().len(), db_rows_padded * db_cols);

    // ================================================================
    // OFFLINE PHASE (InspiRING precomp now lives inside the server)
    // ================================================================
    let mut measurements = vec![Measurement::default(); trials + 1];

    let start_offline_comp = Instant::now();

    let offline_values = if word {
        y_server.perform_offline_precomputation_simplepir_word(Some(&mut measurements[0]), packing)
    } else {
        y_server.perform_offline_precomputation_simplepir(Some(&mut measurements[0]), packing)
    };

    let offline_server_time_ms = start_offline_comp.elapsed().as_millis();

    let packed_query_row_sz = params.db_rows_padded();

    for trial in 0..trials + 1 {
        debug!("trial: {}", trial);
        let mut measurement = &mut measurements[trial];
        measurement.offline.server_time_ms = offline_server_time_ms as usize;

        // ================================================================
        // QUERY GENERATION PHASE
        // ================================================================
        let mut online_upload_bytes = 0;

        // Unified: both packing types produce (YClient, target_idx, PackingKeys)
        let mut query_meta: Vec<(YClient, usize, PackingKeys)> = Vec::new();
        let mut ring_queries: Vec<AlignedMemory64> = Vec::new();
        let mut word_queries_storage: Vec<Vec<u64>> = Vec::new();

        let mut clients = (0..K).map(|_| Client::init(&params)).collect::<Vec<_>>();

        for (_batch, client) in (0..K).zip(clients.iter_mut()) {
            let target_idx: usize = rng.gen::<usize>() % (db_rows * db_cols);
            let target_row = target_idx / db_cols;
            let target_col = target_idx % db_cols;
            debug!(
                "Target item: {} ({}, {})",
                target_idx, target_row, target_col
            );

            let start = Instant::now();
            client.generate_secret_keys();
            let sk_reg = &client.get_sk_reg();

            let (pub_params_size, pk) = match packing {
                PackingType::InspiRING => {
                    let pp = offline_values.packing_params.as_ref().unwrap();
                    let packing_keys = PackingKeys::init_full(pp, sk_reg, W_SEED, V_SEED);
                    let size = packing_keys.get_size_bytes();
                    debug!("InspiRING packing key size: {} bytes", size);
                    (size, packing_keys)
                },
                _ => {
                    let pack_pub_params = raw_generate_expansion_params(
                        &params,
                        &sk_reg,
                        params.poly_len_log2,
                        params.t_exp_left,
                        &mut ChaCha20Rng::from_entropy(),
                        &mut ChaCha20Rng::from_seed(STATIC_SEED_2),
                    );
                    let mut pack_pub_params_row_1s = pack_pub_params.to_vec();
                    for i in 0..pack_pub_params.len() {
                        pack_pub_params_row_1s[i] =
                            pack_pub_params[i].submatrix(1, 0, 1, pack_pub_params[i].cols);
                        pack_pub_params_row_1s[i] = condense_matrix(&params, &pack_pub_params_row_1s[i]);
                    }
                    let size = get_vec_pm_size_bytes(&pack_pub_params_row_1s);
                    debug!("pub params size: {} bytes", size);
                    let packing_keys = PackingKeys::init_cdks_from_keys(params.clone(), pack_pub_params_row_1s);
                    (size, packing_keys)
                },
            };

            let y_client = YClient::new(client, &params);

            if word {
                let wq = y_client.generate_query_word(SEED_0, params.db_dim_1, target_row);
                assert_eq!(wq.len(), db_rows);
                let query_size = wq.len() * 8;
                online_upload_bytes = query_size + pub_params_size;
                word_queries_storage.push(wq);
            } else {
                let query_row = y_client.generate_query(SEED_0, params.db_dim_1, packing, target_row, None, None);
                let query_row_last_row: &[u64] = &query_row;
                assert_eq!(query_row_last_row.len(), db_rows);
                let packed_query_row = pack_query(&params, query_row_last_row);
                let query_size = ((packed_query_row.len() as f64 * params.modulus_log2 as f64) / 8.0)
                    .ceil() as usize;
                online_upload_bytes = query_size + pub_params_size;
                ring_queries.push(packed_query_row);
            }

            query_meta.push((y_client, target_idx, pk));

            measurement.online.client_query_gen_time_ms = start.elapsed().as_millis() as usize;
            debug!("Generated query in {} us", start.elapsed().as_micros());
            debug!("Query size: {} bytes", online_upload_bytes);
            debug!("Client {} query generated", _batch);
        }

        let offline_values = offline_values.clone();

        // ================================================================
        // ONLINE PHASE (unified: server handles both packing types)
        // ================================================================

        let mut packing_keys: Vec<PackingKeys> = query_meta.iter().map(|(_, _, pk)| pk.clone()).collect();

        let start_online_comp = Instant::now();

        let responses: Vec<Vec<Vec<u8>>> = if word {
            let query_slices: Vec<&[u64]> = word_queries_storage.iter().map(|q| q.as_slice()).collect();
            y_server.perform_online_computation_simplepir_word(
                &query_slices,
                &offline_values,
                &mut packing_keys,
                Some(&mut measurement),
            )
        } else {
            let mut all_queries_packed = AlignedMemory64::new(K * packed_query_row_sz);
            for (i, chunk_mut) in all_queries_packed
                .as_mut_slice()
                .chunks_mut(packed_query_row_sz)
                .enumerate()
            {
                (&mut chunk_mut[..db_rows]).copy_from_slice(ring_queries[i].as_slice());
            }
            let query_slices: Vec<&[u64]> = all_queries_packed
                .as_slice()
                .chunks(packed_query_row_sz)
                .collect();
            y_server.perform_online_computation_simplepir(
                &query_slices,
                &offline_values,
                &mut packing_keys,
                Some(&mut measurement),
            )
        };

        let online_server_time_ms = start_online_comp.elapsed().as_millis();
        let online_download_bytes = get_size_bytes(&responses);

        // check correctness
        for (response_switched, (y_client, target_idx, _)) in
            responses.iter().zip(query_meta.iter())
        {
            let (target_row, _target_col) = (target_idx / db_cols, target_idx % db_cols);
            let corr_result = y_server
                .get_row(target_row)
                .iter()
                .map(|x| x.to_u64())
                .collect::<Vec<_>>();

            let start_decode = Instant::now();

            debug!("rescaling response...");
            let mut response = Vec::new();
            for ct_bytes in response_switched.iter() {
                let ct = PolyMatrixRaw::recover(&params, rlwe_q_prime_1, rlwe_q_prime_2, ct_bytes);
                response.push(ct);
            }

            debug!("decrypting outer cts...");
            let outer_ct: Vec<u64> = response
                .iter()
                .flat_map(|ct| {
                    decrypt_ct_reg_measured(
                        y_client.client(),
                        &params,
                        &ct.ntt(),
                        params.poly_len,
                    )
                    .as_slice()
                    .to_vec()
                })
                .collect();
            assert_eq!(outer_ct.len(), num_rlwe_outputs * params.poly_len);
            let final_result = outer_ct.as_slice();
            measurement.online.client_decode_time_ms = start_decode.elapsed().as_millis() as usize;

            if final_result != corr_result {
                let mismatches: Vec<usize> = final_result.iter().zip(corr_result.iter())
                    .enumerate()
                    .filter(|(_, (a, b))| a != b)
                    .map(|(i, _)| i)
                    .collect();
                eprintln!("MISMATCH: {} / {} values differ", mismatches.len(), final_result.len());
                for &i in mismatches.iter().take(10) {
                    eprintln!("  [{i}] got={}, expected={}", final_result[i], corr_result[i]);
                }
            }
            assert_eq!(final_result, corr_result);
        }

        measurement.online.upload_bytes = online_upload_bytes;
        measurement.online.download_bytes = online_download_bytes;
        measurement.online.server_time_ms = online_server_time_ms as usize;
    }

    // discard the first measurement (if there were multiple trials)
    // copy offline values from the first measurement to the second measurement
    if trials > 1 {
        measurements[1].offline = measurements[0].offline.clone();
        measurements.remove(0);
    }

    let mut final_measurement = measurements[0].clone();
    final_measurement.online.server_time_ms = mean(
        &measurements
            .iter()
            .map(|m| m.online.server_time_ms)
            .collect::<Vec<_>>(),
    )
    .round() as usize;
    final_measurement.online.all_server_times_ms = measurements
        .iter()
        .map(|m| m.online.server_time_ms)
        .collect::<Vec<_>>();
    final_measurement.online.std_dev_server_time_ms =
        std_dev(&final_measurement.online.all_server_times_ms);

    final_measurement
}

pub fn run_ypir_on_params<const K: usize>(
    params: Params,
    is_simplepir: bool,
    word: bool,
    trials: usize,
    packing: PackingType,
) -> Measurement {
    if is_simplepir || word {
        return run_simple_ypir_on_params::<K>(params, word, trials, packing);
    }
    let lwe_params = LWEParams::default();

    let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
    let db_rows_padded = params.db_rows_padded(); // == db_rows as defined above
    let db_cols = 1 << (params.db_dim_2 + params.poly_len_log2);

    let sqrt_n_bytes = db_cols * (lwe_params.pt_modulus as f64).log2().floor() as usize / 8;

    let mut rng = thread_rng();

    // RLWE reduced moduli
    let rlwe_q_prime_1 = params.get_q_prime_1(); // (2^28)
    let rlwe_q_prime_2 = params.get_q_prime_2(); // (2^20)

    // LWE reduced moduli (2^28)
    let lwe_q_prime_bits = lwe_params.q2_bits as usize;

    // The number of bits represented by a plaintext RLWE coefficient
    let pt_bits = (params.pt_modulus as f64).log2().floor() as usize;
    // assert_eq!(pt_bits, 16);

    // The factor by which ciphertext values are bigger than plaintext values
    let blowup_factor = lwe_q_prime_bits as f64 / pt_bits as f64;
    debug!("blowup_factor: {}", blowup_factor);

    let mut smaller_params = params.clone();
    smaller_params.db_dim_1 = params.db_dim_2;
    smaller_params.db_dim_2 = ((blowup_factor * (lwe_params.n + 1) as f64) / params.poly_len as f64)
        .log2()
        .ceil() as usize;

    let out_rows = 1 << (smaller_params.db_dim_2 + params.poly_len_log2);
    let rho = 1 << smaller_params.db_dim_2; // rho

    debug!("rho: {}", rho);

    assert_eq!(smaller_params.db_dim_1, params.db_dim_2);
    assert!(out_rows as f64 >= (blowup_factor * (lwe_params.n + 1) as f64));

    // --

    let lwe_q_bits = (lwe_params.modulus as f64).log2().ceil() as usize;

    let rlwe_q_prime_1_bits = (rlwe_q_prime_1 as f64).log2().ceil() as usize;
    let rlwe_q_prime_2_bits = (rlwe_q_prime_2 as f64).log2().ceil() as usize;
    let simplepir_hint_bytes = (lwe_params.n * db_cols * lwe_q_prime_bits) / 8;
    let doublepir_hint_bytes = (params.poly_len * out_rows * rlwe_q_prime_2_bits) / 8;
    let simplepir_query_bytes = db_rows * lwe_q_bits / 8;
    let doublepir_query_bytes = db_cols * params.modulus_log2 as usize / 8;
    let simplepir_resp_bytes = (db_cols * lwe_q_prime_bits) / 8;
    let doublepir_resp_bytes = ((rho * params.poly_len) * rlwe_q_prime_2_bits
        + (rho * params.poly_len) * rlwe_q_prime_1_bits)
        / 8;
    debug!(
        "          \"simplepirHintBytes\": {},",
        simplepir_hint_bytes
    );
    debug!("          \"doublepirHintBytes\": {}", doublepir_hint_bytes);
    debug!(
        "          \"simplepirQueryBytes\": {},",
        simplepir_query_bytes
    );
    debug!(
        "          \"doublepirQueryBytes\": {},",
        doublepir_query_bytes
    );
    debug!(
        "          \"simplepirRespBytes\": {},",
        simplepir_resp_bytes
    );
    debug!(
        "          \"doublepirRespBytes\": {},",
        doublepir_resp_bytes
    );

    // --

    let now = Instant::now();
    let pt_iter = std::iter::repeat_with(|| u8::sample()); // TODO: can substitute the correct database here.
    let y_server = YServer::<u8>::new(&params, pt_iter, is_simplepir, false, true);
    debug!("Created server in {} us", now.elapsed().as_micros());
    debug!(
        "Database of {} bytes",
        y_server.db().len() * std::mem::size_of::<u8>()
    );
    let db_pt_modulus = lwe_params.pt_modulus;
    
    assert_eq!(
        y_server.db().len() * std::mem::size_of::<u8>(),
        db_rows_padded * db_cols * (db_pt_modulus as f64).log2().ceil() as usize / 8
    );

    // ================================================================
    // OFFLINE PHASE
    // ================================================================
    let mut measurements = vec![Measurement::default(); trials + 1];

    let start_offline_comp = Instant::now();
    // server precomputed state (silent preprocessing)
    let offline_values = y_server.perform_offline_precomputation(Some(&mut measurements[0]), packing);
    let offline_server_time_ms = start_offline_comp.elapsed().as_millis();

    let packed_query_row_sz = params.db_rows_padded();
    // let mut all_queries_packed = AlignedMemory64::new(K * packed_query_row_sz);

    for trial in 0..trials + 1 {
        let mut measurement = &mut measurements[trial];
        measurement.offline.server_time_ms = offline_server_time_ms as usize;
        measurement.offline.simplepir_hint_bytes = simplepir_hint_bytes;
        measurement.offline.doublepir_hint_bytes = doublepir_hint_bytes;
        measurement.online.simplepir_query_bytes = simplepir_query_bytes;
        measurement.online.doublepir_query_bytes = doublepir_query_bytes;
        measurement.online.simplepir_resp_bytes = simplepir_resp_bytes;
        measurement.online.doublepir_resp_bytes = doublepir_resp_bytes;

        // ================================================================
        // QUERY GENERATION PHASE
        // ================================================================
        let mut online_upload_bytes = 0;
        // Unified: both packing types produce (YClient, target_idx, PackingKeys)
        let mut query_meta: Vec<(YClient, usize)> = Vec::new();
        let mut packed_query_rows: Vec<Vec<u32>> = Vec::new();
        let mut packed_query_cols: Vec<AlignedMemory64> = Vec::new();
        let mut packing_keys_vec: Vec<PackingKeys> = Vec::new();

        let mut clients = (0..K).map(|_| Client::init(&params)).collect::<Vec<_>>();

        for (_batch, client) in (0..K).zip(clients.iter_mut()) {
            let target_idx: usize = rng.gen::<usize>() % (db_rows * db_cols);
            let target_row = target_idx / db_cols;
            let target_col = target_idx % db_cols;
            debug!(
                "Target item: {} ({}, {})",
                target_idx, target_row, target_col
            );

            let start = Instant::now();
            client.generate_secret_keys();
            let sk_reg = &client.get_sk_reg();

            let (pub_params_size, pk) = match packing {
                PackingType::InspiRING => {
                    let pp = offline_values.packing_params.as_ref().unwrap();
                    let packing_keys = PackingKeys::init_full(pp, sk_reg, W_SEED, V_SEED);
                    let size = packing_keys.get_size_bytes();
                    debug!("InspiRING packing key size: {} bytes", size);
                    (size, packing_keys)
                },
                _ => {
                    let pack_pub_params = raw_generate_expansion_params(
                        &params,
                        &sk_reg,
                        params.poly_len_log2,
                        params.t_exp_left,
                        &mut ChaCha20Rng::from_entropy(),
                        &mut ChaCha20Rng::from_seed(STATIC_SEED_2),
                    );
                    let mut pack_pub_params_row_1s = pack_pub_params.to_vec();
                    for i in 0..pack_pub_params.len() {
                        pack_pub_params_row_1s[i] =
                            pack_pub_params[i].submatrix(1, 0, 1, pack_pub_params[i].cols);
                        pack_pub_params_row_1s[i] = condense_matrix(&params, &pack_pub_params_row_1s[i]);
                    }
                    let size = get_vec_pm_size_bytes(&pack_pub_params_row_1s);
                    debug!("pub params size: {} bytes", size);
                    let packing_keys = PackingKeys::init_cdks_from_keys(params.clone(), pack_pub_params_row_1s);
                    (size, packing_keys)
                },
            };

            let y_client = YClient::new(client, &params);
            let query_row = y_client.generate_query(SEED_0, params.db_dim_1, PackingType::NoPacking, target_row, None, None);
            let query_row_last_row: &[u64] = &query_row[lwe_params.n * db_rows..];

            let packed_query_row_u32 = query_row_last_row
                .iter()
                .map(|x| *x as u32)
                .collect::<Vec<_>>();

            let query_col = y_client.generate_query(SEED_1, params.db_dim_2, packing, target_col, None, None);
            let packed_query_col = pack_query(&params, &query_col);

            let query_size = query_row_last_row.len() * 4 + query_col.len() * 8;

            measurement.online.client_query_gen_time_ms = start.elapsed().as_millis() as usize;
            debug!("Generated query in {} us", start.elapsed().as_micros());

            online_upload_bytes = query_size + pub_params_size;
            debug!("Query size: {} bytes", online_upload_bytes);
            debug!("Client {} query generated", _batch);

            query_meta.push((y_client, target_idx));
            packed_query_rows.push(packed_query_row_u32);
            packed_query_cols.push(packed_query_col);
            packing_keys_vec.push(pk);
        }

        let mut all_queries_packed = vec![0u32; K * packed_query_row_sz];
        for (i, chunk_mut) in all_queries_packed
            .as_mut_slice()
            .chunks_mut(packed_query_row_sz)
            .enumerate()
        {
            (&mut chunk_mut[..db_rows]).copy_from_slice(packed_query_rows[i].as_slice());
        }

        let mut offline_values = offline_values.clone();

        // ================================================================
        // ONLINE PHASE
        // ================================================================

        let start_online_comp = Instant::now();

        let query_col_slices: Vec<&[u64]> = packed_query_cols.iter().map(|q| q.as_slice()).collect();

        let responses = y_server.perform_online_computation::<K>(
            &mut offline_values,
            &all_queries_packed,
            &query_col_slices,
            &mut packing_keys_vec,
            Some(&mut measurement),
        );

        assert_eq!(responses.len(), K);

        let online_server_time_ms = start_online_comp.elapsed().as_millis();
        let online_download_bytes = get_size_bytes(&responses); // TODO: this is not quite right for multiple clients

        // check correctness
        for (response_switched, (y_client, target_idx)) in
            responses.iter().zip(query_meta.iter())
        {
            let (target_row, target_col) = (target_idx / db_cols, target_idx % db_cols);
            let corr_result = y_server.get_elem(target_row, target_col).to_u64();

            let scheme_params = YPIRSchemeParams::from_params(&params, &lwe_params);
            let log2_corr_err = scheme_params.delta().log2();
            let log2_expected_outer_noise = scheme_params.expected_outer_noise().log2();
            debug!("log2_correctness_err: {}", log2_corr_err); // 2^-41.751921014932854
            debug!("log2_expected_outer_noise: {}", log2_expected_outer_noise);

            let start_decode = Instant::now();

            debug!("rescaling response...");
            let mut response = Vec::new();
            for ct_bytes in response_switched.iter() {
                let ct = PolyMatrixRaw::recover(&params, rlwe_q_prime_1, rlwe_q_prime_2, ct_bytes);
                response.push(ct);
            }
            // response now contains a full RLWE in q2 space

            debug!("decrypting outer cts...");
            let outer_ct = response
                .iter()
                .flat_map(|ct| {
                    decrypt_ct_reg_measured(y_client.client(), &params, &ct.ntt(), params.poly_len)
                        .as_slice()
                        .to_vec()
                })
                .collect::<Vec<_>>();

            let mut inner_ct = PolyMatrixRaw::zero(&params, 2, 1);

            let lwe_q_prime = lwe_params.get_q_prime_2();
            let special_offs =
                ((lwe_params.n * lwe_q_prime_bits) as f64 / pt_bits as f64).ceil() as usize;
            let gamma = params.poly_len;

            // first n values were packed tightly in the mask CTs
            let outer_ct_t_u8 = u64s_to_contiguous_bytes(&outer_ct, pt_bits);
            let mut bit_offs = 0;
            for z in 0..lwe_params.n {
                let val = read_bits(&outer_ct_t_u8, bit_offs, lwe_q_prime_bits);
                bit_offs += lwe_q_prime_bits;
                assert!(
                    val < lwe_q_prime,
                    "val: {}, lwe_q_prime: {}",
                    val,
                    lwe_q_prime
                );
                inner_ct.data[z] = rescale(val, lwe_q_prime, lwe_params.modulus);
            }

            // b_val extraction depends on packing type
            let val = match packing {
                PackingType::InspiRING => {
                    // NoPacking body: excess CTs are raw after the mask CTs
                    // b_val is reconstructed from coefficient 0 of each decrypted excess CT
                    let num_rlwes_for_mask = (special_offs as f64 / gamma as f64).ceil() as usize;
                    let mut val = 0u64;
                    for i in 0..blowup_factor.ceil() as usize {
                        val |= (outer_ct[num_rlwes_for_mask * gamma + i * gamma] % params.pt_modulus)
                            << (i * pt_bits);
                    }
                    val
                },
                _ => {
                    // CDKS: b_val packed at special_offs within the packed CTs
                    bit_offs = special_offs * pt_bits;
                    read_bits(&outer_ct_t_u8, bit_offs, lwe_q_prime_bits)
                },
            };
            assert!(
                val < lwe_q_prime,
                "val: {}, lwe_q_prime: {}",
                val,
                lwe_q_prime
            );
            debug!("got b_val of: {}", val);
            inner_ct.data[lwe_params.n] = rescale(val, lwe_q_prime, lwe_params.modulus);

            debug!("decrypting inner ct...");
            // let plaintext = decrypt_ct_reg_measured(y_client.client(), &params, &inner_ct.ntt(), 1);
            // let final_result = plaintext.data[0];
            let inner_ct_as_u32 = inner_ct
                .as_slice()
                .iter()
                .take(lwe_params.n + 1)
                .map(|x| *x as u32)
                .collect::<Vec<_>>();
            let decrypted = y_client.lwe_client().decrypt(&inner_ct_as_u32);
            let final_result = rescale(decrypted as u64, lwe_params.modulus, lwe_params.pt_modulus);

            measurement.online.client_decode_time_ms = start_decode.elapsed().as_millis() as usize;

            debug!("got {}, expected {}", final_result, corr_result);
            // debug!("was correct? {}", final_result == corr_result);
            assert_eq!(final_result, corr_result);
        }

        measurement.online.upload_bytes = online_upload_bytes;
        measurement.online.download_bytes = online_download_bytes;
        measurement.online.server_time_ms = online_server_time_ms as usize;
        measurement.online.sqrt_n_bytes = sqrt_n_bytes;
    }

    // discard the first measurement (if there were multiple trials)
    // copy offline values from the first measurement to the second measurement
    if trials > 1 {
        measurements[1].offline = measurements[0].offline.clone();
        measurements.remove(0);
    }

    let mut final_measurement = measurements[0].clone();
    final_measurement.online.server_time_ms = mean(
        &measurements
            .iter()
            .map(|m| m.online.server_time_ms)
            .collect::<Vec<_>>(),
    )
    .round() as usize;
    final_measurement.online.all_server_times_ms = measurements
        .iter()
        .map(|m| m.online.server_time_ms)
        .collect::<Vec<_>>();
    final_measurement.online.std_dev_server_time_ms =
        std_dev(&final_measurement.online.all_server_times_ms);

    final_measurement
}



fn mean(xs: &[usize]) -> f64 {
    xs.iter().map(|x| *x as f64).sum::<f64>() / xs.len() as f64
}

fn std_dev(xs: &[usize]) -> f64 {
    let mean = mean(xs);
    let mut variance = 0.;
    for x in xs {
        variance += (*x as f64 - mean).powi(2);
    }
    (variance / xs.len() as f64).sqrt()
}

// e2e tests!
#[cfg(test)]
mod test {
    use super::*;
    use crate::packing::PackingType;
    use test_log::test;

    #[test]
    fn test_ypir_basic() {
        run_ypir_batched(1 << 30, 1, 1, false, false, 1, PackingType::CDKS);
    }

    #[test]
    fn test_ypir_simplepir_basic() {
        run_ypir_batched(1 << 15, 2048 * 15, 1, true, false, 1, PackingType::CDKS);
    }

    #[test]
    fn test_ypir_sp_word_basic() {
        run_ypir_batched(1 << 10, 65536 * 8, 1, true, true, 1, PackingType::CDKS);
    }

    #[test]
    fn test_ypir_simplepir_rectangle() {
        run_ypir_batched(1 << 16, 16384 * 8, 1, true, false, 1, PackingType::CDKS);
    }

    #[test]
    fn test_ypir_simplepir_rectangle_8gb() {
        run_ypir_batched(1 << 17, 65536 * 8, 1, true, false, 1, PackingType::CDKS);
    }

    #[test]
    fn test_ypir_many_clients() {
        run_ypir_batched(1 << 30, 1, 2, false, false, 1, PackingType::CDKS);
    }

    #[test]
    fn test_ypir_many_clients_and_trials() {
        run_ypir_batched(1 << 30, 1, 2, false, false, 5, PackingType::CDKS);
    }

    #[test]
    #[ignore]
    fn test_ypir_small() {
        run_ypir_batched(1 << 20, 8, 1, false, false, 5, PackingType::CDKS);
    }

    #[test]
    #[ignore]
    fn test_ypir_1gb() {
        run_ypir_batched(1 << 33, 1, 2, false, false, 1, PackingType::CDKS);
    }

    #[test]
    #[ignore]
    // add a test for Toeplitz matrix (8 GB GPU RAM limit)
    fn test_ypir_toeplitz() {
        run_ypir_batched(1 << 32, 1, 1, false, false, 5, PackingType::CDKS);
    }

    #[test]
    #[ignore]
    fn test_ypir_2gb() {
        run_ypir_batched(1 << 34, 1, 1, false, false, 5, PackingType::CDKS);
    }

    #[test]
    #[ignore]
    fn test_ypir_4gb() {
        run_ypir_batched(1 << 35, 1, 1, false, false, 5, PackingType::CDKS);
    }

    #[test]
    #[ignore]
    fn test_ypir_8gb() {
        run_ypir_batched(1 << 36, 1, 1, false, false, 5, PackingType::CDKS);
    }

    #[test]
    #[ignore]
    fn test_ypir_16gb() {
        run_ypir_batched(1 << 37, 1, 1, false, false, 5, PackingType::CDKS);
    }

    #[test]
    #[ignore]
    fn test_ypir_32gb() {
        run_ypir_batched(1 << 38, 1, 1, false, false, 5, PackingType::CDKS);
    }

    #[test]
    #[ignore]
    fn test_batched_4_ypir() {
        run_ypir_batched(1 << 30, 1, 4, false, false, 5, PackingType::CDKS);
    }

    #[test]
    fn test_ypir_sp_word_inspiring() {
        run_ypir_batched(1 << 10, 65536 * 8, 1, true, true, 1, PackingType::InspiRING);
    }

    #[test]
    fn test_ypir_simplepir_inspiring() {
        run_ypir_batched(1 << 10, 65536 * 8, 1, true, false, 1, PackingType::InspiRING);
    }

    #[test]
    fn test_ypir_sp_word_inspiring_batched() {
        run_ypir_batched(1 << 10, 65536 * 8, 2, true, true, 1, PackingType::InspiRING);
    }

    #[test]
    fn test_ypir_simplepir_batched() {
        run_ypir_batched(1 << 10, 65536 * 8, 2, true, false, 1, PackingType::CDKS);
    }

    #[test]
    fn test_ypir_doublepir_inspiring() {
        run_ypir_batched(1 << 30, 1, 1, false, false, 1, PackingType::InspiRING);
    }

    #[test]
    fn test_ypir_doublepir_inspiring_batched() {
        run_ypir_batched(1 << 30, 1, 2, false, false, 1, PackingType::InspiRING);
    }
}
