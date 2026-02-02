use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::net::TcpStream;
use std::time::Instant;
use clap::Parser;
use log::{debug, info};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use spiral_rs::client::*;
use spiral_rs::poly::{PolyMatrix, PolyMatrixRaw};

use ypir::api::sp_messages::{SetupParams, SimplePIRQuery, SimplePIRQueryBatch, SimplePIRResponseBatch};
use ypir::client::{decrypt_ct_reg_measured, pack_query, raw_generate_expansion_params, YClient};
use ypir::modulus_switch::ModulusSwitch;
use ypir::packing::condense_matrix;
use ypir::params::{params_for_scenario_simplepir, GetQPrime};
use ypir::scheme::{SEED_0, STATIC_SEED_2};

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:9000")]
    server: String,
    #[arg(long, default_value = "results.txt")]
    output: String,
    /// Row indices to query
    #[arg(num_args = 0..)]
    indices: Vec<usize>,
}

fn main() -> std::io::Result<()> {
    env_logger::init();
    let args = Args::parse();

    info!("Connecting to server at {}...", args.server);
    let mut stream = TcpStream::connect(&args.server)?;
    let mut reader = BufReader::new(stream.try_clone()?);

    let mut setup_line = String::new();
    reader.read_line(&mut setup_line)?;
    let setup: SetupParams = serde_json::from_str(setup_line.trim())?;
    info!("Connected. DB: {} rows x {} cols", setup.db_rows, setup.db_cols);

    let params = params_for_scenario_simplepir(setup.num_items, setup.item_size_bits);

    let mut output_file = File::create(&args.output)?;

    if !args.indices.is_empty() {
        let results = process_batch(&args.indices, &mut stream, &mut reader, &setup, &params)?;
        for (idx, row) in results {
            let line = format!("{}: {:?}", idx, row);
            println!("{}", line);
            writeln!(output_file, "{}", line)?;
        }
    } else {
        loop {
            print!("Row indices (space-separated, or 'q')> ");
            std::io::stdout().flush()?;
            let mut input = String::new();
            if std::io::stdin().read_line(&mut input)? == 0 { break; }
            let input = input.trim();
            if input == "q" { break; }

            let indices: Vec<usize> = input
                .split_whitespace()
                .filter_map(|s| s.parse().ok())
                .collect();

            if indices.is_empty() { continue; }

            match process_batch(&indices, &mut stream, &mut reader, &setup, &params) {
                Ok(results) => {
                    for (idx, row) in results {
                        let line = format!("{}: {:?}", idx, row);
                        writeln!(output_file, "{}", line)?;
                    }
                    output_file.flush()?;
                }
                Err(e) => println!("Error: {}", e),
            }
        }
    }
    Ok(())
}

fn process_batch(
    indices: &[usize],
    stream: &mut TcpStream,
    reader: &mut BufReader<TcpStream>,
    setup: &SetupParams,
    params: &spiral_rs::params::Params,
) -> std::io::Result<Vec<(usize, Vec<u64>)>> {
    let k = indices.len();
    info!("Processing batch of {} queries", k);
    let start = Instant::now();

    // 1. Initialize Clients
    // 'clients' owns the Client structs.
    let mut clients: Vec<Client> = (0..k).map(|_| Client::init(params)).collect();

    // store: (client index, row index, packed_query, packed_pub_params)
    let mut queries_data: Vec<(usize, usize, Vec<u64>, Vec<Vec<u64>>)> = Vec::with_capacity(k);

    for (i, &row_index) in indices.iter().enumerate() {
        if row_index >= setup.num_items {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("Index {}", row_index),
            ));
        }

        let client = &mut clients[i];
        client.generate_secret_keys();

        let packed_pub_params = {
            let sk_reg = client.get_sk_reg();
            let pack_pub_params = raw_generate_expansion_params(
                params,
                &sk_reg,
                params.poly_len_log2,
                params.t_exp_left,
                &mut ChaCha20Rng::from_entropy(),
                &mut ChaCha20Rng::from_seed(STATIC_SEED_2),
            );
            pack_pub_params
                .iter()
                .map(|p| {
                    let row1 = p.submatrix(1, 0, 1, p.cols);
                    let condensed = condense_matrix(params, &row1);
                    condensed.as_slice().to_vec()
                })
                .collect::<Vec<Vec<u64>>>()
        };

        let packed_query = {
            let y_client = YClient::new(client, params);
            let query_row = y_client.generate_query(SEED_0, params.db_dim_1, true, row_index);
            pack_query(params, &query_row).as_slice().to_vec()
        }; // y_client dropped here

        queries_data.push((i, row_index, packed_query, packed_pub_params));
    }

    debug!("Query generation: {:?}", start.elapsed());

    // 3. Send Batch
    let batch_query = SimplePIRQueryBatch {
        queries: queries_data
            .iter()
            .map(|(_, row, q, pp)| SimplePIRQuery {
                target_row: *row,
                target_col: 0,
                packed_query: q.clone(),
                packed_pub_params: pp.clone(),
            })
            .collect(),
    };

    let t_e2e = Instant::now();

    writeln!(stream, "{}", serde_json::to_string(&batch_query)?)?;
    stream.flush()?;

    // 4. Receive batch
    let mut resp_line = String::new();
    reader.read_line(&mut resp_line)?;
    let response: SimplePIRResponseBatch = serde_json::from_str(resp_line.trim())?;

    let e2e_ms = t_e2e.elapsed().as_secs_f64() * 1e3;
    info!(
        "YPIR-SP & serialize latency: {:.2} ms ({} queries, {:.2} ms/query)",
        e2e_ms,
        k,
        e2e_ms / k as f64
    );

    debug!("Server response received: {:?}", start.elapsed());

    // 5. Decrypt
    // We iterate over 'queries_data' to get the 'y_client' we stored.
    let rlwe_q_prime_1 = params.get_q_prime_1();
    let rlwe_q_prime_2 = params.get_q_prime_2();
    let mut results = Vec::with_capacity(k);

    for (resp_i, resp) in response.responses.iter().enumerate() {
        let (client_i, original_idx, _q, _pp) = &queries_data[resp_i];
        let client = &clients[*client_i];

        let mut row_data: Vec<u64> = Vec::new();
        for ct_bytes in resp.data.iter() {
            let ct = PolyMatrixRaw::recover(params, rlwe_q_prime_1, rlwe_q_prime_2, ct_bytes);
            let decrypted = decrypt_ct_reg_measured(client, params, &ct.ntt(), params.poly_len);
            row_data.extend_from_slice(decrypted.as_slice());
        }

        row_data.truncate(setup.db_cols);
        results.push((*original_idx, row_data));
    }

    info!("Batch complete: {} queries in {:?}", k, start.elapsed());
    Ok(results)
}