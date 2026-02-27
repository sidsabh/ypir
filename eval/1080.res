> ./build/pir_bench --mode all --batches 1,8,64,128,256   --warmup 1 --iters 2

=== uint32 DB x uint32 query ===
DB: 32768 x 32768 uint32 (4.0 GiB)
Query/Output type: uint32
GPU memory: 7.9 GiB total, 7.7 GiB free
GPU: NVIDIA GeForce GTX 1080
HBM BW (theoretical): ~320.3 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1        108.58             9              39.6              39.6
       8        110.45            72             311.1              38.9
      64        132.60           483            2073.0              32.5
     128        157.14           815            3498.6              27.5
     256        342.91           747            3206.4              12.7
------------------------------------------------------------------------
Eff Tput = (DB_size * batch) / time  [amortized]
HW BW   = (DB + queries + output) / time

=== uint32 DB x uint64 query ===
DB: 32768 x 32768 uint32 (4.0 GiB)
Query/Output type: uint64
GPU memory: 7.9 GiB total, 7.7 GiB free
GPU: NVIDIA GeForce GTX 1080
HBM BW (theoretical): ~320.3 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1       1595.17             1               2.7               2.7
       8       1595.26             5              21.5               2.7
      64       1595.45            40             172.3               2.7
     128       1595.82            80             344.5               2.7
     256       3185.90            80             345.1               1.4
------------------------------------------------------------------------
Eff Tput = (DB_size * batch) / time  [amortized]
HW BW   = (DB + queries + output) / time

=== uint16 DB x uint32 query ===
DB: 32768 x 32768 uint16 (2.0 GiB)
Query/Output type: uint32
GPU memory: 7.9 GiB total, 7.7 GiB free
GPU: NVIDIA GeForce GTX 1080
HBM BW (theoretical): ~320.3 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1        103.43            10              20.8              20.8
       8        103.87            77             165.4              20.7
      64        103.84           616            1323.6              20.8
     128        104.02          1231            2642.6              21.0
     256        194.50          1316            2826.5              11.4
------------------------------------------------------------------------
Eff Tput = (DB_size * batch) / time  [amortized]
HW BW   = (DB + queries + output) / time

=== uint16 DB x uint64 query ===
DB: 32768 x 32768 uint16 (2.0 GiB)
Query/Output type: uint64
GPU memory: 7.9 GiB total, 7.7 GiB free
GPU: NVIDIA GeForce GTX 1080
HBM BW (theoretical): ~320.3 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1       1612.11             1               1.3               1.3
       8       1612.33             5              10.7               1.3
      64       1612.56            40              85.2               1.4
     128       1612.74            79             170.4               1.4
     256       3218.83            80             170.8               0.7
------------------------------------------------------------------------
Eff Tput = (DB_size * batch) / time  [amortized]
HW BW   = (DB + queries + output) / time

=== CRT: 2x (uint16 DB x uint32 query) ===
DB: 32768 x 32768 uint16 (2.0 GiB)
Each GEMM: uint16 x uint32 -> uint32, combined time for both
GPU memory: 7.9 GiB total, 7.7 GiB free
GPU: NVIDIA GeForce GTX 1080
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1        207.61             5              10.3              20.7
       8        210.02            38              81.8              20.5
      64        210.47           304             653.0              20.6
     128        210.75           607            1304.3              20.7
     256        393.97           650            1395.4              11.2
------------------------------------------------------------------------
Eff Tput = (DB_size * batch) / time  [amortized]
HW BW   = (2*DB + 2*queries + 2*output) / time  [both GEMMs]

=== CRT-i64: 2x (int32 DB x int32 query -> int64 accum) ===
DB: 32768 x 32768 int32 (widened from uint16, 4.0 GiB)
Each GEMM: int32 x int32 -> int64, combined time for both
GPU memory: 7.9 GiB total, 7.7 GiB free
GPU: NVIDIA GeForce GTX 1080
HBM BW (theoretical): ~320.3 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1        386.12             3               5.6              22.2
       8        386.09            21              44.5              22.3
      64        386.66           166             355.4              22.3
     128        733.55           174             374.7              11.8
     256       1471.60           174             373.6               6.0
------------------------------------------------------------------------


=== uint8 DB x uint32 query -> uint32 (DoublePIR Step 1) ===
DB: 32768 x 32768 uint8 (1.0 GiB)
GPU memory: 7.9 GiB total, 7.7 GiB free
GPU: NVIDIA GeForce GTX 1080
HBM BW (theoretical): ~320.3 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1         55.36            18              19.4              19.4
       2         55.52            36              38.7              19.3
       4         55.87            72              76.9              19.2
       8         55.53           144             154.7              19.4
      16         55.82           287             307.8              19.3
      32         55.62           575             617.7              19.5
      64         55.41          1155            1240.2              19.7
     128        106.84          1198            1286.4              10.4
     256        350.00           731             785.4               3.3
     512        835.55           613             658.0               1.4
    1024       1768.40           579             621.8               0.8
    2048       3595.98           570             611.5               0.4
------------------------------------------------------------------------

Eff Tput = (DB_uint16_size * batch) / time  [amortized]
HW BW   = (2*DB_i32 + 2*queries_i32 + 2*output_i64) / time


RUST_LOG=debug cargo run --release -- 4294967296 8 64 0
Hint Prep. Throughput 2.08 GB/sec
Throughput: 445.99 GB/sec
Step1 269.025 ms, Steps2-5 (parallel, 128 streams) 253.227 ms
Measurement completed. See the README for details on what the following fields mean.
Result:
{
  "offline": {
    "uploadBytes": 0,
    "downloadBytes": 0,
    "serverTimeMs": 21494,
    "clientTimeMs": 0,
    "simplepirPrepTimeMs": 1922,
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
    "serverTimeMs": 574,
    "clientQueryGenTimeMs": 1576,
    "clientDecodeTimeMs": 1,
    "firstPassTimeMs": 0,
    "secondPassTimeMs": 0,
    "ringPackingTimeMs": 0,
    "sqrtNBytes": 65536,
    "allServerTimesMs": [
      574
    ],
    "stdDevServerTimeMs": 0.0
  }
}

> RUST_LOG=debug cargo run --release -- 262144 184320 64 0 -i
# 6 GB database
Hint Prep. Throughput 1.76 GB/sec
Throughput: 90.73 GB/sec
Measurement completed. See the README for details on what the following fields mean.
Result:
{
  "offline": {
    "uploadBytes": 0,
    "downloadBytes": 0,
    "serverTimeMs": 13199,
    "clientTimeMs": 0,
    "simplepirPrepTimeMs": 3193,
    "simplepirHintBytes": 0,
    "doublepirHintBytes": 0
  },
  "online": {
    "uploadBytes": 2308096,
    "downloadBytes": 4718592,
    "simplepirQueryBytes": 0,
    "doublepirQueryBytes": 0,
    "simplepirRespBytes": 0,
    "doublepirRespBytes": 0,
    "serverTimeMs": 3968,
    "clientQueryGenTimeMs": 191,
    "clientDecodeTimeMs": 6,
    "firstPassTimeMs": 3941,
    "secondPassTimeMs": 0,
    "ringPackingTimeMs": 0,
    "sqrtNBytes": 0,
    "allServerTimesMs": [
      3968
    ],
    "stdDevServerTimeMs": 0.0
  }
}