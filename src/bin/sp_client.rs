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

#[derive(Debug, Clone)]
struct QuerySpec {
    index: usize,
    weights: Vec<u64>,
}

/// Parse a single query spec in the form:
///   "idx:w1,w2,w3"
/// Examples:
///   "10:1,2,3,4"
///   "42:5,6,7,8"
fn parse_query_spec(s: &str) -> Result<QuerySpec, String> {
    let s = s.trim();
    if s.is_empty() {
        return Err("empty query token".to_string());
    }

    // Accept either "idx" or "idx:w1,w2,..."
    let (idx_str, w_opt) = match s.split_once(':') {
        Some((a, b)) => (a, Some(b)),
        None => (s, None),
    };

    let index = idx_str
        .parse::<usize>()
        .map_err(|_| format!("bad idx: {}", idx_str))?;

    let weights = match w_opt {
        None => vec![], // means "use default weights later"
        Some(w_str) => {
            let w_str = w_str.trim();
            if w_str.is_empty() {
                vec![] // "idx:" -> also default later
            } else {
                w_str
                    .split(',')
                    .map(|x| x.trim())
                    .filter(|x| !x.is_empty())
                    .map(|x| x.parse::<u64>().map_err(|_| format!("bad weight: {}", x)))
                    .collect::<Result<Vec<_>, _>>()?
            }
        }
    };

    Ok(QuerySpec { index, weights })
}

/// Parse a line containing multiple query specs separated by whitespace:
///   "10:1,2,3,4 42:5,6,7,8"
fn parse_query_line(line: &str) -> Result<Vec<QuerySpec>, String> {
    let mut out = Vec::new();
    for tok in line.split_whitespace() {
        out.push(parse_query_spec(tok)?);
    }
    if out.is_empty() {
        return Err("no queries found".to_string());
    }
    Ok(out)
}

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:9000")]
    server: String,

    #[arg(long, default_value = "results.txt")]
    output: String,

    #[arg(long, short, default_value_t = 1)]
    embedding_width: usize,

    /// Repeatable: --query 10:1,2,3,4 --query 42:5,6,7,8
    #[arg(long, value_parser = parse_query_spec)]
    query: Vec<QuerySpec>,
}

// A helper to handle modular decoding and find the Top K
fn get_top_k(scores: &[u64], k: usize, pt_modulus: u64) -> Vec<(usize, i64)> {
    let mut decoded: Vec<(usize, i64)> = scores
        .iter()
        .enumerate()
        .map(|(i, &val)| {
            // Modular Decoding: Map Z_p to signed i64
            let signed_val = if val > (pt_modulus / 2) {
                (val as i64) - (pt_modulus as i64)
            } else {
                val as i64
            };
            (i, signed_val)
        })
        .collect();

    // Sort descending by score
    decoded.sort_by(|a, b| b.1.cmp(&a.1));
    decoded.into_iter().take(k).collect()
}

fn main() -> std::io::Result<()> {
    env_logger::init();
    let args = Args::parse();

    info!("Connecting to server at {}...", args.server);
    let mut stream = TcpStream::connect(&args.server)?;
    let mut reader = BufReader::new(stream.try_clone()?);

    // Setup line from server
    let mut setup_line = String::new();
    reader.read_line(&mut setup_line)?;
    let setup: SetupParams = serde_json::from_str(setup_line.trim())?;

    info!("Connected. DB: {} rows x {} cols", setup.db_rows, setup.db_cols);

    // embedding width sanity
    assert!(setup.db_rows % args.embedding_width == 0);
    info!("Num embeddings: {}", setup.db_rows / args.embedding_width);

    // Local params constructed from setup
    let params = params_for_scenario_simplepir(setup.num_items, setup.item_size_bits);

    let mut output_file = File::create(&args.output)?;

    let handle_results = |results: Vec<(usize, Vec<u64>)>, output_file: &mut File, pt_modulus: u64| -> std::io::Result<()> {
        for (cluster_idx, row_scores) in results {
            let top_10 = get_top_k(&row_scores, 10, pt_modulus);
            
            println!("\n--- Top 10 for Cluster {} ---", cluster_idx);
            for (rank, (local_idx, score)) in top_10.iter().enumerate() {
                let res_str = format!("Rank {}: Local Index {}, Score {}", rank + 1, local_idx, score);
                println!("{}", res_str);
                writeln!(output_file, "{}: {}", cluster_idx, res_str)?;
            }
        }
        output_file.flush()?;
        Ok(())
    };

    // Non-interactive mode: use --query flags
    if !args.query.is_empty() {
        let results = process_batch(
            &args.query,
            &mut stream,
            &mut reader,
            &setup,
            &params,
            args.embedding_width,
        )?;

        for (idx, row) in &results {
            let line = format!("{}: {:?}", idx, row);
            writeln!(output_file, "{}", line)?;
        }
        handle_results(results, &mut output_file, setup.pt_modulus)?;
        output_file.flush()?;
        return Ok(());
    }

    // Interactive mode
    loop {
        print!("Queries: idx or idx:w1,w2,... (e.g. 10 or 10:1,2,3,4) or 'q'> ");
        std::io::stdout().flush()?;

        let mut input = String::new();
        if std::io::stdin().read_line(&mut input)? == 0 {
            break;
        }
        let input = input.trim();
        if input == "q" || input == "quit" {
            break;
        }
        if input.is_empty() {
            continue;
        }

        let batch = match parse_query_line(input) {
            Ok(b) => b,
            Err(e) => {
                println!("Parse error: {}", e);
                continue;
            }
        };

        match process_batch(
            &batch,
            &mut stream,
            &mut reader,
            &setup,
            &params,
            args.embedding_width,
        ) {
            Ok(results) => {
                for (idx, row) in &results {
                    let line = format!("{}: {:?}", idx, row);
                    writeln!(output_file, "{}", line)?;
                }
                handle_results(results, &mut output_file, setup.pt_modulus)?;
                output_file.flush()?;
            }
            Err(e) => println!("Error: {}", e),
        }
    }

    Ok(())
}

fn process_batch(
    queries: &[QuerySpec],
    stream: &mut TcpStream,
    reader: &mut BufReader<TcpStream>,
    setup: &SetupParams,
    params: &spiral_rs::params::Params,
    embedding_width: usize,
) -> std::io::Result<Vec<(usize, Vec<u64>)>> {
    let k = queries.len();
    info!("Processing batch of {} queries", k);

    // Validate each query and normalize weights requirements
    let num_embeddings = setup.num_items / embedding_width;
    for q in queries {
        if q.index >= num_embeddings {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!("index {} out of range [0..{})", q.index, num_embeddings),
            ));
        }
    }

    // 1) Initialize Clients
    let mut clients: Vec<Client> = (0..k).map(|_| Client::init(params)).collect();

    // store: (client_i, embedding_idx, packed_query, packed_pub_params)
    let mut queries_data: Vec<(usize, usize, Vec<u64>, Vec<Vec<u64>>)> = Vec::with_capacity(k);

    for (i, q) in queries.iter().enumerate() {
        let row_index = q.index;

        let client = &mut clients[i];
        client.generate_secret_keys();

        // 2a) Pub params
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

        // Normalize weights
        let weights: Vec<u64> = if q.weights.is_empty() {
            vec![1u64; embedding_width]
        } else {
            q.weights.clone()
        };

        // If user supplied weights, they must match embedding_width
        if weights.len() != embedding_width {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                format!(
                    "index {} has {} weights, expected {}",
                    q.index,
                    weights.len(),
                    embedding_width
                ),
            ));
        }
        for weight in &weights {
            assert!(*   weight < params.pt_modulus);
        }

        // 2b) Query
        let packed_query = {
            let y_client = YClient::new(client, params);
            let query_row = y_client.generate_query(
                SEED_0,
                params.db_dim_1,
                true,
                row_index,
                Some(embedding_width),
                Some(weights.as_slice()),
            );
            pack_query(params, &query_row).as_slice().to_vec()
        }; // y_client dropped here

        queries_data.push((i, row_index, packed_query, packed_pub_params));
    }

    debug!("Query generation complete.");

    // 3) Send Batch
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

    // One E2E timer: send -> recv
    let t_e2e = Instant::now();

    writeln!(stream, "{}", serde_json::to_string(&batch_query)?)?;
    stream.flush()?;

    // 4) Receive batch
    let mut resp_line = String::new();
    reader.read_line(&mut resp_line)?;
    let response: SimplePIRResponseBatch = serde_json::from_str(resp_line.trim())?;

    let e2e_ms = t_e2e.elapsed().as_secs_f64() * 1e3;
    info!(
        "E2E send->recv: {:.2} ms ({} queries, {:.2} ms/query)",
        e2e_ms,
        k,
        e2e_ms / k as f64
    );

    // 5) Decrypt
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

    Ok(results)
}
