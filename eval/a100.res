c301-001.ls6(59)$ ./build/pir_bench --mode all --batches 1,8,64,128,256   --warmup 1 --iters 2

=== uint32 DB x uint32 query ===
DB: 32768 x 32768 uint32 (4.0 GiB)
Query/Output type: uint32
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1         21.17            47             202.8             202.8
       8         21.11           379            1627.3             203.5
      64         21.25          3012           12934.8             202.9
     128         18.45          6936           29789.8             234.6
     256         30.94          8275           35542.6             141.0
------------------------------------------------------------------------
Eff Tput = (DB_size * batch) / time  [amortized]
HW BW   = (DB + queries + output) / time

=== uint32 DB x uint64 query ===
DB: 32768 x 32768 uint32 (4.0 GiB)
Query/Output type: uint64
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1         71.72            14              59.9              59.9
       8         71.74           112             478.9              59.9
      64         71.95           889            3820.2              60.2
     128         73.42          1743            7487.9              59.4
     256        122.29          2093            8991.2              36.2
------------------------------------------------------------------------
Eff Tput = (DB_size * batch) / time  [amortized]
HW BW   = (DB + queries + output) / time

=== uint16 DB x uint32 query ===
DB: 32768 x 32768 uint16 (2.0 GiB)
Query/Output type: uint32
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1         18.99            53             113.1             113.1
       8         18.98           421             905.0             113.2
      64         19.01          3367            7230.4             113.9
     128         19.02          6731           14454.6             114.7
     256         31.87          8034           17252.2              69.5
------------------------------------------------------------------------
Eff Tput = (DB_size * batch) / time  [amortized]
HW BW   = (DB + queries + output) / time

=== uint16 DB x uint64 query ===
DB: 32768 x 32768 uint16 (2.0 GiB)
Query/Output type: uint64
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1         74.27            13              28.9              28.9
       8         74.27           108             231.3              29.0
      64         74.28           862            1850.2              29.4
     128         74.32          1722            3698.6              29.8
     256        124.71          2053            4408.4              18.3
------------------------------------------------------------------------
Eff Tput = (DB_size * batch) / time  [amortized]
HW BW   = (DB + queries + output) / time

=== CRT: 2x (uint16 DB x uint32 query) ===
DB: 32768 x 32768 uint16 (2.0 GiB)
Each GEMM: uint16 x uint32 -> uint32, combined time for both
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1         37.98            26              56.5             113.1
       8         37.97           211             452.4             113.2
      64         38.01          1684            3615.7             113.9
     128         38.04          3365            7225.8             114.7
     256         63.70          4019            8630.0              69.5
------------------------------------------------------------------------
Eff Tput = (DB_size * batch) / time  [amortized]
HW BW   = (2*DB + 2*queries + 2*output) / time  [both GEMMs]

=== CRT-i64: 2x (int32 DB x int32 query -> int64 accum) ===
DB: 32768 x 32768 int32 (widened from uint16, 4.0 GiB)
Each GEMM: int32 x int32 -> int64, combined time for both
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1         41.63            24              51.6             206.4
       8         41.63           192             412.6             206.5
      64         41.69          1535            3296.3             207.2
     128         69.10          1852            3978.2             125.8
     256        130.99          1954            4196.8              67.1
------------------------------------------------------------------------
Eff Tput = (DB_uint16_size * batch) / time  [amortized]
HW BW   = (2*DB_i32 + 2*queries_i32 + 2*output_i64) / time


=== cuBLAS Tensor Core: 4x int8 GEMM (DoublePIR Step 1) ===
DB: 32768 x 32768 int8 (1.0 GiB)
4 GEMMs with alpha/beta folding: int8 x int8 -> int32 (tensor cores via cuBLAS)
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1          8.55           117             125.6             502.5
       2          7.66           261             280.4             561.1
       4          7.35           544             584.2             584.7
       8          6.79          1178            1265.0             633.7
      16         13.59          1177            1264.2             317.3
      32          3.26          9806           10528.9            1326.4
      64          3.40         18805           20192.2            1281.7
     128          3.87         33034           35469.8            1143.1
     256          7.59         33749           36237.4             601.6
     512         15.66         32691           35102.2             308.5
    1024         31.89         32108           34475.4             168.3
    2048         69.18         29603           31786.0              93.1
------------------------------------------------------------------------

=== uint8 DB x uint32 query -> uint32 (DoublePIR Step 1) ===
DB: 32768 x 32768 uint8 (1.0 GiB)
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1         12.81            78              83.8              83.9
       2          9.66           207             222.4             111.2
       4          9.66           414             444.4             111.2
       8          9.67           827             888.0             111.2
      16          9.71          1648            1769.7             111.0
      32          9.71          3294            3536.9             111.4
      64          9.76          6560            7044.1             111.8
     128         18.57          6891            7399.4              59.6
     256         34.58          7404            7949.9              33.0
     512         68.52          7472            8023.5              17.6
    1024        136.26          7515            8069.3               9.9
    2048        271.49          7543            8099.7               5.9
------------------------------------------------------------------------
Eff Tput = (DB_uint8_size * batch) / time  [amortized]

=== cuBLAS Tensor Core: 15x int8 GEMM (uint16 DB x uint64 query) ===
DB: 32768 x 32768 uint16 (2.0 GiB)
15 GEMMs (2 byte DB x 8 byte query, skip power 8): int8 x int8 -> int32 (tensor cores)
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1         38.58            26              55.7             417.6
       2         39.60            51             108.5             407.0
       4         40.44            99             212.4             398.7
       8         37.83           212             454.2             426.7
      16         74.22           216             463.0             217.9
      32         12.66          2527            5426.1            1282.2
      64         14.91          4293            9218.5            1098.2
     128         21.96          5829           12518.7             757.8
     256         36.88          6942           14907.7             465.6
     512         71.45          7166           15387.9             255.2
    1024        142.90          7166           15388.6             142.5
    2048        287.83          7115           15280.2              85.6
------------------------------------------------------------------------


RUST_LOG=debug cargo run --features toeplitz --release -- 4294967296 8 64 0
# tensor cores
Hint Prep. Throughput 16.53 GB/sec
Throughput: 1361.70 GB/sec
Step1 17.021 ms, Steps2-5 (parallel, 128 streams) 148.512 ms
Measurement completed. See the README for details on what the following fields mean.
Result:
{
  "offline": {
    "uploadBytes": 0,
    "downloadBytes": 0,
    "serverTimeMs": 9687,
    "clientTimeMs": 0,
    "simplepirPrepTimeMs": 242,
    "simplepirHintBytes": 234881024,
    "doublepirHintBytes": 14680064
  },
  "online": {
    "uploadBytes": 1259520,
    "downloadBytes": 786432,
    "simplepirQueryBytes": 262144,
    "doublepirQueryBytes": 458752,
    "simplepirRespBytes": 229376,
    "doublepirRespBytes": 12288,
    "serverTimeMs": 188,
    "clientQueryGenTimeMs": 960,
    "clientDecodeTimeMs": 0,
    "firstPassTimeMs": 0,
    "secondPassTimeMs": 0,
    "ringPackingTimeMs": 0,
    "sqrtNBytes": 65536,
    "allServerTimesMs": [
      188
    ],
    "stdDevServerTimeMs": 0.0
  }
}
# no tensor cores for online
Hint Prep. Throughput 20.30 GB/sec
Throughput: 1032.26 GB/sec
Step1 100.582 ms, Steps2-5 (parallel, 128 streams) 125.543 ms
Measurement completed. See the README for details on what the following fields mean.
Result:
{
  "offline": {
    "uploadBytes": 0,
    "downloadBytes": 0,
    "serverTimeMs": 9658,
    "clientTimeMs": 0,
    "simplepirPrepTimeMs": 197,
    "simplepirHintBytes": 234881024,
    "doublepirHintBytes": 14680064
  },
  "online": {
    "uploadBytes": 1259520,
    "downloadBytes": 786432,
    "simplepirQueryBytes": 262144,
    "doublepirQueryBytes": 458752,
    "simplepirRespBytes": 229376,
    "doublepirRespBytes": 12288,
    "serverTimeMs": 248,
    "clientQueryGenTimeMs": 1000,
    "clientDecodeTimeMs": 0,
    "firstPassTimeMs": 0,
    "secondPassTimeMs": 0,
    "ringPackingTimeMs": 0,
    "sqrtNBytes": 65536,
    "allServerTimesMs": [
      248
    ],
    "stdDevServerTimeMs": 0.0
  }
}



# ypir-sp 
RUST_LOG=debug cargo run --release -- 131072 368640 64 0 -i

# NTT-based hint computation
Hint Prep. Throughput 3.23 GB/sec
Throughput: 366.60 GB/sec
Measurement completed. See the README for details on what the following fields mean.
Result:
{
  "offline": {
    "uploadBytes": 0,
    "downloadBytes": 0,
    "serverTimeMs": 13865,
    "clientTimeMs": 0,
    "simplepirPrepTimeMs": 1739,
    "simplepirHintBytes": 0,
    "doublepirHintBytes": 0
  },
  "online": {
    "uploadBytes": 1390592,
    "downloadBytes": 9437184,
    "simplepirQueryBytes": 0,
    "doublepirQueryBytes": 0,
    "simplepirRespBytes": 0,
    "doublepirRespBytes": 0,
    "serverTimeMs": 982,
    "clientQueryGenTimeMs": 100,
    "clientDecodeTimeMs": 6,
    "firstPassTimeMs": 970,
    "secondPassTimeMs": 0,
    "ringPackingTimeMs": 0,
    "sqrtNBytes": 0,
    "allServerTimesMs": [
      982
    ],
    "stdDevServerTimeMs": 0.0
  }
}

# ypir-sp-word, no tensor cores
Word Step1(SIMT) 173.771 ms, Step2 (149 streams, 1 chunks) 213.502 ms
Hint Prep. Throughput 1.94 GB/sec
Throughput: 895.52 GB/sec
Measurement completed. See the README for details on what the following fields mean.
Result:
{
  "offline": {
    "uploadBytes": 0,
    "downloadBytes": 0,
    "serverTimeMs": 16986,
    "clientTimeMs": 0,
    "simplepirPrepTimeMs": 5242,
    "simplepirHintBytes": 0,
    "doublepirHintBytes": 0
  },
  "online": {
    "uploadBytes": 1521664,
    "downloadBytes": 9437184,
    "simplepirQueryBytes": 0,
    "doublepirQueryBytes": 0,
    "simplepirRespBytes": 0,
    "doublepirRespBytes": 0,
    "serverTimeMs": 402,
    "clientQueryGenTimeMs": 3224,
    "clientDecodeTimeMs": 6,
    "firstPassTimeMs": 402,
    "secondPassTimeMs": 0,
    "ringPackingTimeMs": 0,
    "sqrtNBytes": 0,
    "allServerTimesMs": [
      402
    ],
    "stdDevServerTimeMs": 0.0
  }
}
# tensor cores
Hint Prep. Throughput 4.82 GB/sec
Throughput: 1121.50 GB/sec
Word Step1(TC) 77.981 ms, Step2 (149 streams, 1 chunks) 225.546 ms
Measurement completed. See the README for details on what the following fields mean.
Result:
{
  "offline": {
    "uploadBytes": 0,
    "downloadBytes": 0,
    "serverTimeMs": 15654,
    "clientTimeMs": 0,
    "simplepirPrepTimeMs": 3753,
    "simplepirHintBytes": 0,
    "doublepirHintBytes": 0
  },
  "online": {
    "uploadBytes": 1521664,
    "downloadBytes": 9437184,
    "simplepirQueryBytes": 0,
    "doublepirQueryBytes": 0,
    "simplepirRespBytes": 0,
    "doublepirRespBytes": 0,
    "serverTimeMs": 321,
    "clientQueryGenTimeMs": 3253,
    "clientDecodeTimeMs": 6,
    "firstPassTimeMs": 321,
    "secondPassTimeMs": 0,
    "ringPackingTimeMs": 0,
    "sqrtNBytes": 0,
    "allServerTimesMs": [
      321
    ],
    "stdDevServerTimeMs": 0.0
  }
}
