=== uint32 DB x uint32 query ===
DB: 32768 x 32768 uint32 (4.0 GiB)
Query/Output type: uint32
GPU memory: 14.6 GiB total, 14.5 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1         47.70            21              90.0              90.0
       2         46.66            43             184.1              92.1
       4         47.67            84             360.4              90.1
       8         48.03           167             715.3              89.5
      16         48.71           328            1410.7              88.3
      32         48.30           662            2845.3              89.1
      64         49.16          1302            5591.2              87.7
     128         49.70          2575           11061.4              87.1
     256         96.98          2640           11337.6              45.0
     512        193.31          2649           11375.6              22.9
    1024        387.88          2640           11338.8              11.8
    2048        790.93          2589           11121.3               6.1
------------------------------------------------------------------------
Eff Tput = (DB_size * batch) / time  [amortized]
HW BW   = (DB + queries + output) / time

=== uint32 DB x uint64 query ===
DB: 32768 x 32768 uint32 (4.0 GiB)
Query/Output type: uint64
GPU memory: 14.6 GiB total, 14.4 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1        161.63             6              26.6              26.6
       2        162.10            12              53.0              26.5
       4        162.43            25             105.8              26.5
       8        161.57            50             212.7              26.6
      16        161.28            99             426.1              26.7
      32        162.32           197             846.7              26.6
      64        162.35           394            1693.1              26.7
     128        164.16           780            3349.0              26.6
     256        310.03           826            3546.4              14.3
     512        632.36           810            3477.5               7.2
    1024       1272.02           805            3457.5               3.8
    2048       2543.78           805            3457.9               2.1
------------------------------------------------------------------------
Eff Tput = (DB_size * batch) / time  [amortized]
HW BW   = (DB + queries + output) / time

=== uint16 DB x uint32 query ===
DB: 32768 x 32768 uint16 (2.0 GiB)
Query/Output type: uint32
GPU memory: 14.6 GiB total, 14.4 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1         44.42            23              48.3              48.4
       2         46.10            43              93.2              46.6
       4         44.75            89             192.0              48.0
       8         45.70           175             375.9              47.0
      16         45.82           349             749.9              47.0
      32         45.83           698            1499.6              47.0
      64         45.43          1409            3025.0              47.6
     128         48.59          2634            5657.4              44.9
     256         93.12          2749            5903.4              23.8
     512        184.89          2769            5946.7              12.3
    1024        368.81          2777            5962.5               6.6
    2048        752.90          2720            5841.5               3.6
------------------------------------------------------------------------
Eff Tput = (DB_size * batch) / time  [amortized]
HW BW   = (DB + queries + output) / time

=== uint16 DB x uint64 query ===
DB: 32768 x 32768 uint16 (2.0 GiB)
Query/Output type: uint64
GPU memory: 14.6 GiB total, 14.4 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1        158.61             6              13.5              13.5
       2        158.64            13              27.1              13.5
       4        158.41            25              54.2              13.6
       8        159.68            50             107.6              13.5
      16        159.35           100             215.6              13.5
      32        159.88           200             429.8              13.5
      64        160.24           399             857.7              13.6
     128        161.37           793            1703.4              13.7
     256        304.21           842            1807.2               7.5
     512        616.65           830            1783.1               3.9
    1024       1226.86           835            1792.4               2.2
    2048       2438.65           840            1803.5               1.3
------------------------------------------------------------------------
Eff Tput = (DB_size * batch) / time  [amortized]
HW BW   = (DB + queries + output) / time

=== CRT: 2x (uint16 DB x uint32 query) ===
DB: 32768 x 32768 uint16 (2.0 GiB)
Each GEMM: uint16 x uint32 -> uint32, combined time for both
GPU memory: 14.6 GiB total, 14.4 GiB free
GPU: Tesla T4
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1         91.11            11              23.6              47.1
       2         90.98            22              47.2              47.2
       4         90.93            44              94.5              47.3
       8         90.30            89             190.3              47.6
      16         92.09           174             373.1              46.7
      32         92.87           345             740.0              46.4
      64         94.13           680            1460.2              46.0
     128         95.92          1334            2865.7              45.5
     256        184.66          1386            2977.1              24.0
     512        367.10          1395            2995.1              12.4
    1024        746.97          1371            2943.9               6.5
    2048       1506.76          1359            2918.9               3.6
------------------------------------------------------------------------
Eff Tput = (DB_size * batch) / time  [amortized]
HW BW   = (2*DB + 2*queries + 2*output) / time  [both GEMMs]

=== uint8 DB x uint32 query -> uint32 (DoublePIR Step 1) ===
DB: 32768 x 32768 uint8 (1.0 GiB)
GPU memory: 14.6 GiB total, 14.4 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1         23.97            42              44.8              44.8
       2         29.39            68              73.1              36.6
       4         28.26           142             152.0              38.0
       8         28.28           283             303.7              38.0
      16         27.98           572             613.9              38.5
      32         27.91          1147            1231.3              38.8
      64         29.63          2160            2318.9              36.8
     128         66.24          1932            2074.9              16.7
     256        136.18          1880            2018.5               8.4
     512        284.69          1798            1931.0               4.2
    1024        598.11          1712            1838.3               2.2
    2048       1181.14          1734            1861.8               1.4
------------------------------------------------------------------------
Eff Tput = (DB_uint8_size * batch) / time  [amortized]
HW BW   = (DB + queries + output) / time

=== CRT-i64: 2x (int32 DB x int32 query -> int64 accum) ===
DB: 32768 x 32768 int32 (widened from uint16, 4.0 GiB)
Each GEMM: int32 x int32 -> int64, combined time for both
GPU memory: 14.6 GiB total, 14.4 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1        100.22            10              21.4              85.7
       2         99.44            20              43.2              86.4
       4         99.47            40              86.4              86.4
       8        100.17            80             171.5              85.8
      16        100.94           159             340.4              85.2
      32        101.51           315             677.0              84.9
      64        104.72           611            1312.4              82.5
     128        196.07           653            1401.9              44.3
     256        382.22           670            1438.3              23.0
     512        808.79           633            1359.5              11.1
    1024       1670.57           613            1316.3               5.6
    2048       3378.34           606            1301.8               3.0
------------------------------------------------------------------------
Eff Tput = (DB_uint16_size * batch) / time  [amortized]
HW BW   = (2*DB_i32 + 2*queries_i32 + 2*output_i64) / time

=== cuBLAS Tensor Core: 4x int8 GEMM (DoublePIR Step 1) ===
DB: 32768 x 32768 int8 (1.0 GiB)
4 GEMMs with alpha/beta folding: int8 x int8 -> int32 (tensor cores via cuBLAS)
GPU memory: 14.6 GiB total, 14.4 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1         23.00            43              46.7             186.8
       2         23.83            84              90.1             180.3
       4         25.14           159             170.9             171.0
       8         32.77           244             262.1             131.3
      16         69.53           230             247.1              62.0
      32         21.59          1482            1591.2             200.4
      64         22.60          2832            3041.2             193.0
     128         24.38          5250            5637.1             181.7
     256         65.45          3912            4199.9              69.7
     512        127.46          4017            4313.2              37.9
    1024        262.10          3907            4195.0              20.5
    2048        569.80          3594            3859.3              11.3
------------------------------------------------------------------------
Eff Tput = (DB_int8_size * batch) / time  [amortized]
HW BW   = (4*DB + 4*q_bytes + output_writes) / time  [no L2 cache assumed]

=== cuBLAS Tensor Core: 15x int8 GEMM (uint16 DB x uint64 query) ===
DB: 32768 x 32768 uint16 (2.0 GiB)
15 GEMMs (2 byte DB x 8 byte query, skip power 8): int8 x int8 -> int32 (tensor cores)
GPU memory: 14.6 GiB total, 14.4 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1         88.96            11              24.1             181.1
       2         90.89            22              47.3             177.3
       4        101.10            40              85.0             159.5
       8        131.02            61             131.1             123.2
      16        265.83            60             129.3              60.8
      32         99.21           323             692.7             163.7
      64        121.28           528            1133.2             135.0
     128        134.61           951            2042.0             123.6
     256        244.76          1046            2246.1              70.2
     512        497.46          1029            2210.3              36.7
    1024        995.06          1029            2209.9              20.5
    2048       2186.05           937            2011.9              11.3
------------------------------------------------------------------------
Eff Tput = (DB_uint16_size * batch) / time  [amortized]
HW BW   = (15*db_slices + 15*q_slices + output_accesses) / time

=== CUTLASS uint8 TC: 1x wide GEMM (uint8 DB x uint32 query) ===
DB: 32768 x 32768 uint8 (1.0 GiB)
1 GEMM: uint8 × uint8 → int32, M×K × K×(4*batch)
DB read exactly 1x
GPU memory: 14.6 GiB total, 14.4 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1          5.62           178             191.1             191.3
       2          5.64           354             380.5             190.5
       4          5.80           689             740.3             185.6
       8          5.80          1380            1481.5             186.3
      16          5.86          2731            2932.0             185.4
      32          6.02          5314            5705.8             182.5
      64          6.75          9475           10174.2             166.4
     128         12.75         10042           10782.1              92.1
     256         30.40          8421            9042.1              41.9
     512        103.46          4949            5313.9              14.3
    1024        217.18          4715            5062.7               8.7
    2048        555.50          3687            3958.6               4.8
------------------------------------------------------------------------
Eff Tput = (DB_uint8_size * batch) / time  [amortized]
HW BW   = (db + packed_query + gemm_out + accum) / time

=== CUTLASS uint8 TC: 2x wide GEMM (uint16 DB x uint64 query) ===
DB: 32768 x 32768 uint16 (2.0 GiB)
2 GEMMs: uint8 × uint8 → int32, each M×K × K×(8*batch)
DB read exactly 2x (1x per byte slice)
GPU memory: 14.6 GiB total, 14.4 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1         11.27            89             190.5             190.8
       2         11.78           170             364.7             183.0
       4         11.99           334             716.7             180.3
       8         12.16           658            1412.9             178.9
      16         12.17          1315            2823.7             181.0
      32         13.97          2290            4918.2             161.5
      64         39.20          1633            3506.4              60.4
     128         94.42          1356            2911.1              27.4
     256        186.58          1372            2946.5              16.2
     512        450.47          1137            2440.8               8.6
    1024       1041.06           984            2112.3               5.4
    2048       2135.64           959            2059.4               4.3
------------------------------------------------------------------------
Eff Tput = (DB_uint16_size * batch) / time  [amortized]
HW BW   = (2*db_bytes + 2*packed_query + 2*gemm_out + accum) / time


=== uint32 DB x uint32 query ===
DB: 32768 x 32768 uint32 (4.0 GiB)
Query/Output type: uint32
GPU memory: 14.6 GiB total, 14.5 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1       46.25       47.96          21              89.6              92.9
       2       46.27       48.15          42             178.4              92.8
       4       46.95       48.94          82             351.0              91.5
       8       47.61       48.97         163             701.6              90.2
      16       47.90       48.02         333            1431.1              89.7
      32       48.78       49.99         640            2749.4              88.2
      64       49.17       53.31        1200            5155.8              87.7
     128       50.87       56.15        2279            9790.4              85.1
     256       97.30      107.76        2376           10203.0              44.8
     512      192.48      216.03        2370           10179.4              23.0
    1024      383.85      447.38        2289            9830.6              11.9
    2048      787.81      876.75        2336           10032.6               6.1
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_size * batch) / E2E_time  [amortized]
HW BW   = (DB + queries + output) / comp_time

=== uint32 DB x uint64 query ===
DB: 32768 x 32768 uint32 (4.0 GiB)
Query/Output type: uint64
GPU memory: 14.6 GiB total, 14.4 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1      160.73      161.57           6              26.6              26.7
       2      161.32      162.09          12              53.0              26.6
       4      161.65      159.11          25             108.0              26.6
       8      162.07      162.90          49             210.9              26.5
      16      161.40      162.56          98             422.7              26.7
      32      161.48      165.48         193             830.6              26.7
      64      162.56      169.23         378            1624.3              26.6
     128      163.95      178.22         718            3084.8              26.6
     256      309.44      340.55         752            3228.7              14.3
     512      630.13      680.28         753            3232.5               7.2
    1024     1267.20     1380.21         742            3186.5               3.8
    2048     2537.30     2762.93         741            3183.6               2.1
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_size * batch) / E2E_time  [amortized]
HW BW   = (DB + queries + output) / comp_time

=== uint16 DB x uint32 query ===
DB: 32768 x 32768 uint16 (2.0 GiB)
Query/Output type: uint32
GPU memory: 14.6 GiB total, 14.4 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1       45.83       45.61          22              47.1              46.9
       2       45.65       45.32          44              94.8              47.1
       4       45.32       46.30          86             185.5              47.4
       8       44.65       46.26         173             371.3              48.1
      16       44.50       46.47         344             739.5              48.4
      32       44.66       48.37         662            1420.6              48.3
      64       45.58       51.06        1253            2691.6              47.5
     128       47.14       55.11        2323            4988.2              46.3
     256       94.15      104.88        2441            5241.6              23.5
     512      184.03      209.35        2446            5252.1              12.4
    1024      370.89      426.42        2401            5157.0               6.5
    2048      753.60      850.73        2407            5169.7               3.6
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_size * batch) / E2E_time  [amortized]
HW BW   = (DB + queries + output) / comp_time

=== uint16 DB x uint64 query ===
DB: 32768 x 32768 uint16 (2.0 GiB)
Query/Output type: uint64
GPU memory: 14.6 GiB total, 14.4 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1      159.47      158.66           6              13.5              13.5
       2      158.59      159.09          13              27.0              13.5
       4      159.54      159.48          25              53.9              13.5
       8      158.43      160.48          50             107.1              13.6
      16      159.82      161.47          99             212.8              13.5
      32      159.92      163.75         195             419.7              13.5
      64      161.13      167.49         382             820.6              13.5
     128      162.41      175.42         730            1567.0              13.6
     256      304.13      333.07         769            1650.6               7.5
     512      615.08      668.49         766            1644.8               3.9
    1024     1230.31     1344.38         762            1635.7               2.2
    2048     2448.45     2675.50         765            1643.8               1.3
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_size * batch) / E2E_time  [amortized]
HW BW   = (DB + queries + output) / comp_time

=== CRT: 2x (uint16 DB x uint32 query) ===
DB: 32768 x 32768 uint16 (2.0 GiB)
Each GEMM: uint16 x uint32 -> uint32, combined time for both
GPU memory: 14.6 GiB total, 14.4 GiB free
GPU: Tesla T4
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1       92.03       90.89          11              23.6              46.7
       2       90.56       92.16          22              46.6              47.4
       4       92.14       91.17          44              94.2              46.6
       8       91.72       92.46          87             185.8              46.9
      16       92.09       93.08         172             369.1              46.7
      32       91.95       95.95         334             716.2              46.9
      64       93.16      101.08         633            1359.7              46.5
     128       97.01      108.91        1175            2523.8              45.0
     256      185.31      210.54        1216            2611.1              23.9
     512      374.58      426.24        1201            2579.5              12.2
    1024      758.28      859.08        1192            2559.7               6.4
    2048     1518.94     1731.01        1183            2540.7               3.5
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_size * batch) / E2E_time  [amortized]
HW BW   = (2*DB + 2*queries + 2*output) / comp_time  [both GEMMs]

=== uint8 DB x uint32 query -> uint32 (DoublePIR Step 1) ===
DB: 32768 x 32768 uint8 (1.0 GiB)
GPU memory: 14.6 GiB total, 14.4 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1       24.40       30.30          33              35.4              44.0
       2       28.85       28.86          69              74.4              37.2
       4       28.64       29.10         137             147.6              37.5
       8       28.30       30.29         264             283.6              38.0
      16       27.28       31.46         509             546.1              39.5
      32       27.98       33.00         970            1041.1              38.7
      64       30.22       37.58        1703            1828.8              36.1
     128       68.04       71.85        1781            1912.8              16.3
     256      141.99      154.63        1656            1777.7               8.0
     512      286.42      333.28        1536            1649.5               4.2
    1024      609.54      639.89        1600            1718.3               2.2
    2048     1189.60     1304.58        1570            1685.6               1.4
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_uint8_size * batch) / E2E_time  [amortized]
HW BW   = (DB + queries + output) / comp_time

=== CRT-i64: 2x (int32 DB x int32 query -> int64 accum) ===
DB: 32768 x 32768 int32 (widened from uint16, 4.0 GiB)
Each GEMM: int32 x int32 -> int64, combined time for both
GPU memory: 14.6 GiB total, 14.4 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1      100.38       99.49          10              21.6              85.6
       2       99.19       99.79          20              43.0              86.6
       4       99.55      102.67          39              83.7              86.3
       8      100.74      102.26          78             168.0              85.3
      16      101.68      103.70         154             331.3              84.6
      32      102.66      108.02         296             636.2              83.9
      64      104.94      114.72         558            1198.1              82.3
     128      200.59      221.77         577            1239.5              43.3
     256      398.46      440.37         581            1248.4              22.1
     512      828.62      908.43         564            1210.3              10.9
    1024     1679.98     1848.18         554            1189.8               5.6
    2048     3405.52     3744.28         547            1174.6               3.0
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_uint16_size * batch) / E2E_time  [amortized]
HW BW   = (2*DB_i32 + 2*queries_i32 + 2*output_i64) / comp_time

=== cuBLAS Tensor Core: 4x int8 GEMM (DoublePIR Step 1) ===
DB: 32768 x 32768 int8 (1.0 GiB)
4 GEMMs with alpha/beta folding: int8 x int8 -> int32 (tensor cores via cuBLAS)
GPU memory: 14.6 GiB total, 14.4 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1       23.08       23.60          42              45.5             186.1
       2       23.94       24.51          82              87.6             179.5
       4       25.28       26.56         151             161.7             170.0
       8       33.48       35.60         225             241.3             128.5
      16       71.51       72.20         222             237.9              60.3
      32       21.65       24.10        1328            1425.7             200.0
      64       22.52       30.29        2113            2268.8             193.7
     128       24.05       43.91        2915            3129.7             184.2
     256       66.83       77.38        3308            3552.2              68.3
     512      127.79      153.68        3332            3577.3              37.8
    1024      260.96      319.87        3201            3437.4              20.6
    2048      584.89      698.62        2932            3147.7              11.0
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_int8_size * batch) / E2E_time  [amortized]
HW BW   = (4*DB + 4*q_bytes + output_writes) / comp_time

=== cuBLAS Tensor Core: 15x int8 GEMM (uint16 DB x uint64 query) ===
DB: 32768 x 32768 uint16 (2.0 GiB)
15 GEMMs (2 byte DB x 8 byte query, skip power 8): int8 x int8 -> int32 (tensor cores)
GPU memory: 14.6 GiB total, 14.4 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1       89.29       89.38          11              24.0             180.4
       2       91.10       92.09          22              46.6             176.9
       4      102.69      104.86          38              81.9             157.0
       8      135.55      135.74          59             126.6             119.1
      16      267.35      273.19          59             125.8              60.5
      32       99.60      102.34         313             671.5             163.0
      64      121.60      119.89         534            1146.4             134.6
     128      138.16      145.91         877            1883.9             120.4
     256      247.53      265.04         966            2074.2              69.4
     512      502.65      539.62         949            2037.6              36.3
    1024     1006.44     1134.89         902            1937.7              20.2
    2048     2180.75     2464.13         831            1784.8              11.3
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_uint16_size * batch) / E2E_time  [amortized]
HW BW   = (15*db_slices + 15*q_slices + output_accesses) / comp_time

=== CUTLASS uint8 TC: 1x wide GEMM (uint8 DB x uint32 query) ===
DB: 32768 x 32768 uint8 (1.0 GiB)
1 GEMM: uint8 × uint8 → int32, M×K × K×(4*batch)
DB read exactly 1x
GPU memory: 14.6 GiB total, 14.4 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1        5.63        5.74         174             187.0             191.0
       2        5.74        5.99         334             358.5             187.5
       4        5.91        6.33         632             679.0             182.3
       8        5.89        6.67        1199            1287.4             183.5
      16        5.90        7.21        2219            2382.7             184.0
      32        6.01        8.21        3896            4183.6             182.8
      64        6.90       10.80        5926            6363.1             163.0
     128       11.87       23.11        5538            5946.8              98.9
     256       35.43       61.14        4187            4496.1              36.0
     512      102.90      111.71        4583            4921.2              14.3
    1024      219.66      275.88        3712            3985.5               8.6
    2048      560.40      605.99        3380            3628.8               4.8
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_uint8_size * batch) / E2E_time  [amortized]
HW BW   = (db + packed_query + gemm_out + accum) / comp_time

=== CUTLASS uint8 TC: 1x stacked GEMM (uint16 DB x uint64 query) ===
DB: 32768 x 32768 uint16 (2.0 GiB)
1 GEMM: (2M)×K × K×(8*batch), DB bytes stacked vertically
DB + query each read exactly 1x
GPU memory: 14.6 GiB total, 14.5 GiB free
GPU: Tesla T4
HBM BW (theoretical): ~320.1 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1       11.70       12.51          80             171.6             183.7
       2       12.29       12.67         158             338.9             175.2
       4       12.77       13.17         304             652.0             169.0
       8       14.21       14.19         564            1210.5             152.6
      16       15.37       16.41         975            2093.9             142.4
      32       26.90       26.77        1195            2566.7              82.9
      64       45.84       48.42        1322            2838.8              50.5
     128       89.69       92.34        1386            2976.9              27.7
     256      188.59      206.84        1238            2657.9              14.9
     512      490.21      553.04         926            1988.1               7.1
    1024     1089.67     1212.60         844            1813.5               4.4
    2048     2199.85     2465.31         831            1784.0               3.4
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_uint16_size * batch) / E2E_time  [amortized]
HW BW   = (stacked_db + packed_query + gemm_out + accum) / comp_time


# doublepir


Hint Prep. Throughput 2.68 GB/sec
Throughput: 729.34 GB/sec
Step1 39.207 ms, Steps2-5 (parallel, 256 streams) 254.898 ms
Measurement completed. See the README for details on what the following fields mean.
Result:
{
  "offline": {
    "uploadBytes": 0,
    "downloadBytes": 0,
    "serverTimeMs": 32105,
    "clientTimeMs": 0,
    "simplepirPrepTimeMs": 1491,
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
    "serverTimeMs": 351,
    "clientQueryGenTimeMs": 1732,
    "clientDecodeTimeMs": 1,
    "firstPassTimeMs": 0,
    "secondPassTimeMs": 0,
    "ringPackingTimeMs": 0,
    "sqrtNBytes": 65536,
    "allServerTimesMs": [
      351
    ],
    "stdDevServerTimeMs": 0.0
  }
}

# cdks word

# cdks inspiring