use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct SetupParams {
    pub num_items: usize,
    pub item_size_bits: usize,
    pub db_rows: usize,
    pub db_cols: usize,
    pub pt_modulus: u64,
}

#[derive(Serialize, Deserialize)]
pub struct SimplePIRQuery {
    pub target_row: usize,
    pub target_col: usize,
    pub packed_query: Vec<u64>,
    pub packed_pub_params: Vec<Vec<u64>>,
}

#[derive(Serialize, Deserialize)]
pub struct SimplePIRQueryBatch {
    pub queries: Vec<SimplePIRQuery>,
}

#[derive(Serialize, Deserialize)]
pub struct SimplePIRResponse {
    pub data: Vec<Vec<u8>>,
}

#[derive(Serialize, Deserialize)]
pub struct SimplePIRResponseBatch {
    pub responses: Vec<SimplePIRResponse>,
}