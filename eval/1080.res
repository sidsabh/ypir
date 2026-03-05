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


=== uint32 DB x uint32 query ===
DB: 32768 x 32768 uint32 (4.0 GiB)
Query/Output type: uint32
GPU memory: 7.9 GiB total, 7.7 GiB free
GPU: NVIDIA GeForce GTX 1080
HBM BW (theoretical): ~320.3 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1      111.67      107.33           9              40.0              38.5
       2      107.02      106.82          19              80.4              40.1
       4      107.36      107.11          37             160.4              40.0
       8      108.69      109.05          73             315.1              39.5
      16      111.16      111.81         143             614.6              38.7
      32      116.74      119.17         269            1153.3              36.9
      64      129.74      133.34         480            2061.5              33.2
     128      155.40      163.41         783            3364.4              27.9
     256      329.09      346.55         739            3172.8              13.3
     512      688.42      718.49         713            3060.6               6.4
    1024     1404.53     1484.28         690            2963.1               3.2
    2048     2850.63     3007.09         681            2925.1               1.7
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_size * batch) / E2E_time  [amortized]
HW BW   = (DB + queries + output) / comp_time

=== uint32 DB x uint64 query ===
DB: 32768 x 32768 uint32 (4.0 GiB)
Query/Output type: uint64
GPU memory: 7.9 GiB total, 7.7 GiB free
GPU: NVIDIA GeForce GTX 1080
HBM BW (theoretical): ~320.3 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1     1597.04     1597.02           1               2.7               2.7
       2     1596.82     1597.09           1               5.4               2.7
       4     1596.77     1597.27           3              10.8               2.7
       8     1596.95     1597.62           5              21.5               2.7
      16     1596.83     1598.94          10              43.0               2.7
      32     1596.93     1601.04          20              85.8               2.7
      64     1597.17     1605.12          40             171.3               2.7
     128     1597.35     1613.35          79             340.8               2.7
     256     3188.33     3220.64          79             341.4               1.4
     512     6430.02     6493.30          79             338.7               0.7
    1024    12735.66    12852.93          80             342.2               0.4
    2048    25469.99    25724.42          80             341.9               0.2
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_size * batch) / E2E_time  [amortized]
HW BW   = (DB + queries + output) / comp_time

=== uint16 DB x uint32 query ===
DB: 32768 x 32768 uint16 (2.0 GiB)
Query/Output type: uint32
GPU memory: 7.9 GiB total, 7.7 GiB free
GPU: NVIDIA GeForce GTX 1080
HBM BW (theoretical): ~320.3 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1      104.55      104.62          10              20.5              20.5
       2      104.55      104.71          19              41.0              20.5
       4      104.54      104.82          38              81.9              20.6
       8      104.55      105.08          76             163.5              20.6
      16      104.55      105.62         151             325.3              20.6
      32      104.59      106.62         300             644.5              20.6
      64      105.04      109.03         587            1260.6              20.6
     128      105.19      113.29        1130            2426.4              20.7
     256      197.02      212.96        1202            2581.5              11.2
     512      393.34      424.36        1207            2591.0               5.8
    1024      787.20      847.48        1208            2594.8               3.1
    2048     1557.13     1687.61        1214            2606.1               1.7
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_size * batch) / E2E_time  [amortized]
HW BW   = (DB + queries + output) / comp_time

=== uint16 DB x uint64 query ===
DB: 32768 x 32768 uint16 (2.0 GiB)
Query/Output type: uint64
GPU memory: 7.9 GiB total, 7.7 GiB free
GPU: NVIDIA GeForce GTX 1080
HBM BW (theoretical): ~320.3 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1     1628.55     1628.54           1               1.3               1.3
       2     1611.76     1611.92           1               2.7               1.3
       4     1611.49     1612.17           2               5.3               1.3
       8     1611.57     1612.59           5              10.7               1.3
      16     1611.99     1614.01          10              21.3               1.3
      32     1611.96     1616.33          20              42.5               1.3
      64     1611.83     1620.10          40              84.8               1.4
     128     1612.79     1628.61          79             168.8               1.4
     256     3217.75     3249.94          79             169.2               0.7
     512     6421.79     6485.33          79             169.5               0.4
    1024    12718.37    12855.92          80             171.1               0.2
    2048    25417.73    25676.45          80             171.3               0.1
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_size * batch) / E2E_time  [amortized]
HW BW   = (DB + queries + output) / comp_time

=== CRT: 2x (uint16 DB x uint32 query) ===
DB: 32768 x 32768 uint16 (2.0 GiB)
Each GEMM: uint16 x uint32 -> uint32, combined time for both
GPU memory: 7.9 GiB total, 7.7 GiB free
GPU: NVIDIA GeForce GTX 1080
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1      207.61      207.83           5              10.3              20.7
       2      210.24      210.59           9              20.4              20.4
       4      210.24      210.88          19              40.7              20.4
       8      210.25      211.40          38              81.3              20.4
      16      210.23      212.34          75             161.8              20.5
      32      210.24      214.43         149             320.5              20.5
      64      210.29      218.62         293             628.7              20.6
     128      210.55      226.61         565            1213.0              20.7
     256      393.27      425.51         602            1292.0              11.3
     512      785.97      849.92         602            1293.7               5.8
    1024     1567.95     1696.30         604            1296.4               3.1
    2048     3124.88     3374.69         607            1303.2               1.7
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_size * batch) / E2E_time  [amortized]
HW BW   = (2*DB + 2*queries + 2*output) / comp_time  [both GEMMs]

=== uint8 DB x uint32 query -> uint32 (DoublePIR Step 1) ===
DB: 32768 x 32768 uint8 (1.0 GiB)
GPU memory: 7.9 GiB total, 7.7 GiB free
GPU: NVIDIA GeForce GTX 1080
HBM BW (theoretical): ~320.3 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1       55.90       56.01          18              19.2              19.2
       2       56.97       55.75          36              38.5              18.9
       4       56.96       55.78          72              77.0              18.9
       8       55.91       56.45         142             152.2              19.2
      16       55.93       56.68         282             303.1              19.3
      32       56.02       58.04         551             592.0              19.3
      64       55.62       60.17        1064            1142.1              19.6
     128      105.26      113.27        1130            1213.4              10.5
     256      316.57      324.04         790             848.3               3.6
     512      823.70      861.46         594             638.2               1.5
    1024     1771.56     1821.25         562             603.7               0.8
    2048     3592.31     3719.46         551             591.2               0.4
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_uint8_size * batch) / E2E_time  [amortized]
HW BW   = (DB + queries + output) / comp_time

=== CRT-i64: 2x (int32 DB x int32 query -> int64 accum) ===
DB: 32768 x 32768 int32 (widened from uint16, 4.0 GiB)
Each GEMM: int32 x int32 -> int64, combined time for both
GPU memory: 7.9 GiB total, 7.7 GiB free
GPU: NVIDIA GeForce GTX 1080
HBM BW (theoretical): ~320.3 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1      387.40      387.74           3               5.5              22.2
       2      387.93      389.22           5              11.0              22.1
       4      387.85      388.42          10              22.1              22.2
       8      389.10      388.91          21              44.2              22.1
      16      388.03      390.70          41              87.9              22.2
      32      388.06      393.84          81             174.5              22.2
      64      388.48      400.71         160             343.0              22.2
     128      741.93      765.94         167             358.9              11.7
     256     1484.55     1533.71         167             358.4               5.9
     512     2959.10     3055.51         168             359.8               3.0
    1024     5904.92     6098.37         168             360.6               1.6
    2048    11821.32    12191.70         168             360.7               0.9
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_uint16_size * batch) / E2E_time  [amortized]
HW BW   = (2*DB_i32 + 2*queries_i32 + 2*output_i64) / comp_time

=== cuBLAS Tensor Core: 4x int8 GEMM (DoublePIR Step 1) ===
DB: 32768 x 32768 int8 (1.0 GiB)
4 GEMMs with alpha/beta folding: int8 x int8 -> int32 (tensor cores via cuBLAS)
GPU memory: 7.9 GiB total, 7.7 GiB free
GPU: NVIDIA GeForce GTX 1080
HBM BW (theoretical): ~320.3 GB/s
SKIPPED: Tensor cores require SM75+ (Turing or later)

=== cuBLAS Tensor Core: 15x int8 GEMM (uint16 DB x uint64 query) ===
DB: 32768 x 32768 uint16 (2.0 GiB)
15 GEMMs (2 byte DB x 8 byte query, skip power 8): int8 x int8 -> int32 (tensor cores)
GPU memory: 7.9 GiB total, 7.7 GiB free
GPU: NVIDIA GeForce GTX 1080
HBM BW (theoretical): ~320.3 GB/s
SKIPPED: Tensor cores require SM75+ (Turing or later)

=== CUTLASS uint8 TC: 1x wide GEMM (uint8 DB x uint32 query) ===
DB: 32768 x 32768 uint8 (1.0 GiB)
1 GEMM: uint8 × uint8 → int32, M×K × K×(4*batch)
DB read exactly 1x
GPU memory: 7.9 GiB total, 7.7 GiB free
GPU: NVIDIA GeForce GTX 1080
HBM BW (theoretical): ~320.3 GB/s
SKIPPED: CUTLASS uint8 tensor cores require SM75+ (Turing or later)

=== CUTLASS uint8 TC: 2x wide GEMM (uint16 DB x uint64 query) ===
DB: 32768 x 32768 uint16 (2.0 GiB)
2 GEMMs: uint8 × uint8 → int32, each M×K × K×(8*batch)
DB read exactly 2x (1x per byte slice)
GPU memory: 7.9 GiB total, 7.7 GiB free
GPU: NVIDIA GeForce GTX 1080
HBM BW (theoretical): ~320.3 GB/s
SKIPPED: CUTLASS uint8 tensor cores require SM75+ (Turing or later)


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