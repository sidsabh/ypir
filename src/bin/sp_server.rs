use std::fs::File;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use clap::Parser;
use log::{debug, error, info};
use spiral_rs::params::Params;
use spiral_rs::poly::{PolyMatrix, PolyMatrixNTT};
use ypir::api::sp_messages::{
    SetupParams,
    SimplePIRQueryBatch,
    SimplePIRResponse,
    SimplePIRResponseBatch,
};
use ypir::params::params_for_scenario_simplepir;
use ypir::server::YServer;
use ypir::scheme::Sample;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(long, default_value = "127.0.0.1:9000")]
    bind: String,
    #[arg(long, default_value_t = 1 << 17)]
    num_items: usize,
    #[arg(long, default_value_t = 184_320)]
    item_size_bits: usize,
    #[arg(long)]
    db_path: Option<PathBuf>,
}

struct U16LeFileIter {
    reader: BufReader<File>,
    buf: [u8; 2],
    remaining: usize,
}

impl U16LeFileIter {
    fn new(path: &Path, count_u16: usize) -> std::io::Result<Self> {
        Ok(Self {
            reader: BufReader::new(File::open(path)?),
            buf: [0u8; 2],
            remaining: count_u16,
        })
    }
}

impl Iterator for U16LeFileIter {
    type Item = u16;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        if self.reader.read_exact(&mut self.buf).is_err() {
            return None;
        }
        self.remaining -= 1;
        Some(u16::from_le_bytes(self.buf))
    }
}

fn main() -> std::io::Result<()> {
    env_logger::init();
    let args = Args::parse();

    let params_obj = params_for_scenario_simplepir(args.num_items, args.item_size_bits);
    let params: &'static Params = Box::leak(Box::new(params_obj));

    let db_rows = 1 << (params.db_dim_1 + params.poly_len_log2);
    let db_cols = params.instances * params.poly_len;
    let pt_modulus = params.pt_modulus;

    info!("Initializing Server...");
    info!("Config: num_items={}, item_size_bits={}", args.num_items, args.item_size_bits);
    info!("DB dims: {} rows x {} cols (entries = {})", db_rows, db_cols, db_rows * db_cols);
    info!("PT modulus: {}", pt_modulus);

    let expected_cols_from_bits = args.item_size_bits / 15;
    info!("item_size_bits/15 = {}", expected_cols_from_bits);
    if expected_cols_from_bits != db_cols {
        error!(
            "Mismatch: db_cols (instances*poly_len) = {} but item_size_bits/15 = {}.\n",
            db_cols, expected_cols_from_bits
        );
        std::process::exit(1);
    }

    let expected_u16 = db_rows * db_cols;

    let db_iter: Box<dyn Iterator<Item = u16> + Send> = if let Some(ref path) = args.db_path {
        let sz = std::fs::metadata(path)?.len() as usize;
        let expected_bytes = expected_u16 * 2;

        info!("Loading DB from {:?}", path);
        info!("DB file size: {} bytes, expected at least {} bytes", sz, expected_bytes);

        if sz < expected_bytes {
            error!("DB file too small.");
            std::process::exit(1);
        }

        // Streaming iterator; we modulus-reduce in YServer by providing already-reduced values here
        let it = U16LeFileIter::new(path.as_path(), expected_u16)?
            .map(move |v| (v as u64 % pt_modulus) as u16);

        Box::new(it)
    } else {
        info!("No DB path provided; generating random data.");
        Box::new((0..expected_u16).map(move |_| (u16::sample() as u64 % pt_modulus) as u16))
    };

    // Build server
    let y_server_obj = YServer::<u16>::new(params, db_iter, true, false, true);
    let y_server: &'static YServer<u16> = Box::leak(Box::new(y_server_obj));

    // Quick sanity check: print first few entries of DB row 1
    {
        let r0 = y_server.get_row(1);
        let n = r0.len().min(8);
        info!("DB[1][0..{}] = {:?}", n, &r0[..n]);
    }

    info!("Offline precomputation...");
    let offline_obj = y_server.perform_offline_precomputation_simplepir(None);
    let offline_values: &'static _ = Box::leak(Box::new(offline_obj));

    info!("Server ready.");
    let listener = TcpListener::bind(&args.bind)?;
    info!("Listening on {}", args.bind);

    let setup = SetupParams {
        num_items: args.num_items,
        item_size_bits: args.item_size_bits,
        db_rows,
        db_cols,
        pt_modulus,
    };

    for stream in listener.incoming() {
        match stream {
            Ok(stream) => {
                let s_server = y_server;
                let s_offline = offline_values;
                let s_params = params;
                let s_setup = setup.clone();

                std::thread::spawn(move || {
                    if let Err(e) = handle_client(stream, s_server, s_offline, s_params, s_setup) {
                        error!("Client handler error: {}", e);
                    }
                });
            }
            Err(e) => error!("Connection failed: {}", e),
        }
    }

    Ok(())
}

fn handle_client(
    mut stream: TcpStream,
    server: &YServer<u16>,
    offline: &ypir::server::OfflinePrecomputedValues,
    params: &Params,
    setup: SetupParams,
) -> std::io::Result<()> {
    let peer = stream.peer_addr()?;
    info!("Client connected: {}", peer);

    let mut reader = BufReader::new(stream.try_clone()?);

    // Send setup
    let setup_json = serde_json::to_string(&setup)?;
    writeln!(stream, "{}", setup_json)?;
    stream.flush()?;

    loop {
        let mut line = String::new();
        let bytes_read = reader.read_line(&mut line)?;
        if bytes_read == 0 {
            break;
        }

        let batch: SimplePIRQueryBatch = match serde_json::from_str(line.trim()) {
            Ok(b) => b,
            Err(e) => {
                error!(
                    "Invalid JSON from {}: {}. First 200 chars: {}",
                    peer,
                    e,
                    &line[..line.len().min(200)]
                );
                continue;
            }
        };

        let k = batch.queries.len();
        debug!("Processing batch of {} queries from {}", k, peer);

        // Reconstruct pub params for each query
        let pub_params_all: Vec<Vec<PolyMatrixNTT>> = batch
            .queries
            .iter()
            .map(|q| {
                q.packed_pub_params
                    .iter()
                    .map(|data| {
                        let mut pm = PolyMatrixNTT::zero(params, 1, params.t_exp_left);
                        pm.as_mut_slice().copy_from_slice(data);
                        pm
                    })
                    .collect()
            })
            .collect();

        let query_slices: Vec<&[u64]> = batch
            .queries
            .iter()
            .map(|q| q.packed_query.as_slice())
            .collect();

        let pub_param_refs: Vec<&[PolyMatrixNTT]> =
            pub_params_all.iter().map(|v| v.as_slice()).collect();

        let responses = server.perform_online_computation_simplepir(
            &query_slices,
            offline,
            &pub_param_refs,
            None,
        );

        // Build batch response
        let resp_batch = SimplePIRResponseBatch {
            responses: responses
                .into_iter()
                .map(|data| SimplePIRResponse { data })
                .collect(),
        };

        let resp_json = serde_json::to_string(&resp_batch)?;
        writeln!(stream, "{}", resp_json)?;
        stream.flush()?;
    }

    info!("Client disconnected: {}", peer);
    Ok(())
}
