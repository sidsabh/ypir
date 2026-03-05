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

=== CUTLASS uint8 TC: 2x wide GEMM (uint16 DB x uint64 query) ===
DB: 32768 x 32768 uint16 (2.0 GiB)
2 GEMMs: uint8 × uint8 → int32, each M×K × K×(8*batch)
DB read exactly 2x (1x per byte slice)
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1          3.01           332             714.0             715.1
       2          2.61           766            1644.5             824.9
       4          2.18          1833            3936.7             990.4
       8          2.18          3666            7873.3             996.7
      16          2.22          7215           15495.0             993.0
      32          2.73         11725           25179.7             826.8
      64          4.82         13280           28518.1             490.9
     128         10.16         12594           27044.8             254.2
     256         19.89         12870           27637.1             151.8
     512         41.32         12392           26612.5              94.2
    1024         87.82         11660           25040.1              64.2
    2048        188.93         10840           23279.0              48.3
------------------------------------------------------------------------

=== CUTLASS uint8 TC: 1x wide GEMM (uint8 DB x uint32 query) ===
DB: 32768 x 32768 uint8 (1.0 GiB)
1 GEMM: uint8 × uint8 → int32, M×K × K×(4*batch)
DB read exactly 1x
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
------------------------------------------------------------------------
   Batch     Time (ms)           QPS   Eff Tput (GB/s)      HW BW (GB/s)
------------------------------------------------------------------------
       1          1.45           691             742.0             742.6
       2          1.45          1380            1481.2             741.7
       4          1.41          2842            3051.3             765.1
       8          1.22          6548            7030.3             883.9
      16          1.23         13050           14012.5             886.0
      32          1.24         25824           27728.6             886.8
      64          1.61         39662           42587.2             696.6
     128          2.69         47534           51039.2             436.1
     256          4.68         54715           58750.2             272.5
     512          9.69         52864           56761.9             152.4
    1024         20.33         50359           54072.6              92.4
    2048         42.77         47884           51415.3              62.8
------------------------------------------------------------------------


=== uint32 DB x uint32 query ===
DB: 32768 x 32768 uint32 (4.0 GiB)
Query/Output type: uint32
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1       19.34       19.41          52             221.3             222.1
       2       19.34       19.43         103             442.1             222.1
       4       19.34       19.49         205             881.3             222.1
       8       19.36       19.64         407            1749.2             222.0
      16       19.36       19.81         808            3469.5             222.1
      32       19.37       20.14        1589            6825.4             222.2
      64       19.33       20.75        3084           13244.8             223.1
     128       19.43       22.13        5783           24838.8             222.8
     256       32.42       37.74        6783           29133.2             134.6
     512       64.07       75.08        6819           29287.8              69.1
    1024      122.36      143.22        7150           30707.5              37.3
    2048      244.35      286.14        7157           30740.6              19.8
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_size * batch) / E2E_time  [amortized]
HW BW   = (DB + queries + output) / comp_time

=== uint32 DB x uint64 query ===
DB: 32768 x 32768 uint32 (4.0 GiB)
Query/Output type: uint64
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1       72.20       72.30          14              59.4              59.5
       2       72.20       72.34          28             118.7              59.5
       4       72.19       72.47          55             237.1              59.5
       8       72.19       72.63         110             473.1              59.6
      16       72.21       72.97         219             941.8              59.6
      32       72.30       73.62         435            1866.9              59.6
      64       72.21       74.92         854            3669.0              59.9
     128       72.24       77.73        1647            7072.8              60.4
     256      121.37      132.08        1938            8324.5              36.5
     512      241.17      261.54        1958            8408.0              18.9
    1024      461.36      503.34        2034            8737.7              10.5
    2048      922.67     1005.56        2037            8747.5               5.8
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_size * batch) / E2E_time  [amortized]
HW BW   = (DB + queries + output) / comp_time

=== uint16 DB x uint32 query ===
DB: 32768 x 32768 uint16 (2.0 GiB)
Query/Output type: uint32
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1       20.26       20.32          49             105.7             106.0
       2       20.26       20.34          98             211.1             106.0
       4       20.29       20.45         196             420.1             105.9
       8       20.28       20.57         389             835.2             106.0
      16       20.28       20.74         772            1656.8             106.1
      32       20.29       21.07        1519            3262.0             106.2
      64       20.29       21.71        2948            6331.7             106.7
     128       20.31       23.04        5556           11932.3             107.4
     256       33.82       39.14        6541           14045.8              65.5
     512       66.81       77.30        6624           14224.8              34.2
    1024      127.44      148.29        6905           14829.2              19.0
    2048      254.46      296.01        6919           14857.7              10.5
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_size * batch) / E2E_time  [amortized]
HW BW   = (DB + queries + output) / comp_time

=== uint16 DB x uint64 query ===
DB: 32768 x 32768 uint16 (2.0 GiB)
Query/Output type: uint64
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1       72.71       72.81          14              29.5              29.5
       2       72.71       72.87          27              58.9              29.6
       4       72.71       73.01          55             117.6              29.6
       8       72.72       73.18         109             234.8              29.6
      16       72.73       73.52         218             467.3              29.6
      32       72.74       74.18         431             926.4              29.8
      64       72.75       75.48         848            1821.0              30.0
     128       72.70       77.96        1642            3525.7              30.5
     256      122.22      132.77        1928            4140.6              18.7
     512      242.90      263.92        1940            4166.1               9.9
    1024      464.86      506.52        2022            4341.4               5.8
    2048      929.54     1012.97        2022            4341.7               3.5
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_size * batch) / E2E_time  [amortized]
HW BW   = (DB + queries + output) / comp_time

=== CRT: 2x (uint16 DB x uint32 query) ===
DB: 32768 x 32768 uint16 (2.0 GiB)
Each GEMM: uint16 x uint32 -> uint32, combined time for both
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1       40.53       40.63          25              52.9             106.0
       2       40.53       40.69          49             105.6             106.0
       4       40.53       40.84          98             210.3             106.0
       8       40.52       41.10         195             418.0             106.1
      16       40.54       41.44         386             829.2             106.2
      32       40.58       42.13         759            1631.0             106.3
      64       40.58       43.42        1474            3165.1             106.7
     128       40.64       46.06        2779            5968.3             107.3
     256       67.63       78.42        3264            7010.1              65.5
     512      133.57      154.58        3312            7113.0              34.2
    1024      254.93      296.77        3450            7409.9              19.0
    2048      508.88      592.20        3458            7426.7              10.5
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_size * batch) / E2E_time  [amortized]
HW BW   = (2*DB + 2*queries + 2*output) / comp_time  [both GEMMs]

=== uint8 DB x uint32 query -> uint32 (DoublePIR Step 1) ===
DB: 32768 x 32768 uint8 (1.0 GiB)
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1        9.68        9.74         103             110.2             110.9
       2        9.70        9.77         205             219.8             110.8
       4        9.70        9.85         406             436.1             110.8
       8        9.72       10.01         799             857.8             110.7
      16        9.74       10.20        1569            1685.1             110.6
      32        9.76       10.53        3038            3262.0             110.9
      64        9.79       11.21        5710            6130.9             111.4
     128       18.51       21.29        6012            6455.6              59.8
     256       34.72       40.02        6396            6868.0              32.9
     512       68.77       79.45        6444            6919.3              17.6
    1024      136.82      157.77        6490            6969.1               9.8
    2048      271.38      313.41        6535            7016.4               5.9
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_uint8_size * batch) / E2E_time  [amortized]
HW BW   = (DB + queries + output) / comp_time

=== CRT-i64: 2x (int32 DB x int32 query -> int64 accum) ===
DB: 32768 x 32768 int32 (widened from uint16, 4.0 GiB)
Each GEMM: int32 x int32 -> int64, combined time for both
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1       36.29       36.44          27              58.9             236.7
       2       41.72       42.05          48             102.1             206.0
       4       41.71       42.02          95             204.4             206.0
       8       41.59       42.45         188             404.7             206.7
      16       41.66       42.89         373             801.1             206.5
      32       41.75       43.83         730            1567.9             206.3
      64       41.78       45.90        1394            2994.5             206.8
     128       69.35       77.41        1654            3551.0             125.3
     256      131.96      147.75        1733            3720.9              66.6
     512      262.06      293.56        1744            3745.4              34.3
    1024      522.04      584.61        1752            3761.5              18.0
    2048     1035.96     1160.97        1764            3788.3               9.8
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_uint16_size * batch) / E2E_time  [amortized]
HW BW   = (2*DB_i32 + 2*queries_i32 + 2*output_i64) / comp_time

=== cuBLAS Tensor Core: 4x int8 GEMM (DoublePIR Step 1) ===
DB: 32768 x 32768 int8 (1.0 GiB)
4 GEMMs with alpha/beta folding: int8 x int8 -> int32 (tensor cores via cuBLAS)
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1       10.50       10.55          95             101.8             409.2
       2       10.04        7.07         283             303.7             428.2
       4        7.09        7.25         552             592.6             606.4
       8        6.63        6.96        1149            1233.5             648.9
      16       13.39       13.96        1146            1231.0             322.1
      32        3.29        4.07        7855            8433.7            1315.2
      64        3.41        4.87       13151           14120.4            1277.7
     128        3.72        6.47       19793           21253.0            1192.2
     256        6.25       12.06       21231           22796.1             730.7
     512       14.11       26.64       19222           20639.4             342.4
    1024       32.16       50.94       20102           21584.5             166.9
    2048       67.30       99.94       20493           22004.4              95.7
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_int8_size * batch) / E2E_time  [amortized]
HW BW   = (4*DB + 4*q_bytes + output_writes) / comp_time

=== cuBLAS Tensor Core: 15x int8 GEMM (uint16 DB x uint64 query) ===
DB: 32768 x 32768 uint16 (2.0 GiB)
15 GEMMs (2 byte DB x 8 byte query, skip power 8): int8 x int8 -> int32 (tensor cores)
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1       25.65       25.74          39              83.4             628.0
       2       26.20       26.36          76             162.9             615.2
       4       26.59       26.90         149             319.3             606.2
       8       25.79       26.32         304             652.7             625.8
      16       51.05       51.89         308             662.1             316.8
      32       12.28       13.79        2320            4981.7            1322.9
      64       12.77       15.64        4092            8787.1            1282.0
     128       16.40       22.48        5695           12230.1            1014.5
     256       22.19       38.26        6692           14370.5             773.9
     512       56.07       77.84        6577           14124.9             325.3
    1024      120.83      151.85        6744           14481.8             168.6
    2048      280.60      324.90        6303           13536.6              87.8
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_uint16_size * batch) / E2E_time  [amortized]
HW BW   = (15*db_slices + 15*q_slices + output_accesses) / comp_time

=== CUTLASS uint8 TC: 1x wide GEMM (uint8 DB x uint32 query) ===
DB: 32768 x 32768 uint8 (1.0 GiB)
1 GEMM: uint8 × uint8 → int32, M×K × K×(4*batch)
DB read exactly 1x
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1        1.46        1.52         658             706.7             737.4
       2        1.46        1.56        1283            1377.5             739.0
       4        1.46        1.61        2479            2662.1             738.5
       8        1.46        1.78        4498            4829.4             739.6
      16        1.47        1.95        8188            8792.2             739.8
      32        1.48        2.30       13890           14914.7             740.6
      64        2.19        3.65       17529           18821.9             514.2
     128        3.65        6.47       19798           21258.5             321.9
     256        4.63       10.03       25512           27393.5             275.5
     512        8.83       20.38       25123           26975.5             167.2
    1024       20.16       40.88       25052           26899.1              93.2
    2048       44.30       81.39       25161           27016.8              60.6
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_uint8_size * batch) / E2E_time  [amortized]
HW BW   = (db + packed_query + gemm_out + accum) / comp_time

=== CUTLASS uint8 TC: 1x stacked GEMM (uint16 DB x uint64 query) ===
DB: 32768 x 32768 uint16 (2.0 GiB)
1 GEMM: (2M)×K × K×(8*batch), DB bytes stacked vertically
DB + query each read exactly 1x
GPU memory: 39.5 GiB total, 39.1 GiB free
GPU: NVIDIA A100-PCIE-40GB
HBM BW (theoretical): ~1555.2 GB/s
--------------------------------------------------------------------------------
   Batch   Comp (ms)    E2E (ms)     E2E QPS   Eff Tput (GB/s)      HW BW (GB/s)
--------------------------------------------------------------------------------
       1        2.30        3.15         318             682.6             935.8
       2        1.91        1.97        1014            2178.3            1128.8
       4        1.81        2.13        1881            4039.2            1191.2
       8        1.85        2.29        3487            7488.4            1174.6
      16        1.86        2.66        6020           12928.2            1177.5
      32        2.35        4.07        7871           16903.9             950.6
      64        4.82        7.81        8195           17599.3             480.1
     128        9.92       14.57        8783           18862.2             250.3
     256       20.70       28.95        8844           18991.5             136.2
     512       42.30       58.28        8786           18866.8              82.5
    1024       89.77      123.07        8320           17867.6              53.8
    2048      192.29      263.53        7771           16688.8              39.1
--------------------------------------------------------------------------------
Comp    = GPU compute only (no PCIe)
E2E     = H->D query upload + compute + D->H result download
Eff Tput = (DB_uint16_size * batch) / E2E_time  [amortized]
HW BW   = (2*db_bytes + 2*packed_query + 2*gemm_out + accum) / comp_time



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


# first try corrected TC code
# word cdks
Hint Prep. Throughput 4.74 GB/sec
Throughput: 918.37 GB/sec
Word Step1(TC) 180.386 ms, Step2 (149 streams, 1 chunks) 197.356 ms
Measurement completed. See the README for details on what the following fields mean.
Result:
{
  "offline": {
    "uploadBytes": 0,
    "downloadBytes": 0,
    "serverTimeMs": 64978,
    "clientTimeMs": 0,
    "simplepirPrepTimeMs": 1186,
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
    "serverTimeMs": 392,
    "clientQueryGenTimeMs": 1978,
    "clientDecodeTimeMs": 6,
    "firstPassTimeMs": 387,
    "secondPassTimeMs": 0,
    "ringPackingTimeMs": 0,
    "sqrtNBytes": 0,
    "allServerTimesMs": [
      392
    ],
    "stdDevServerTimeMs": 0.0
  }
}
# word inspiring
Word InspiRING Step1(TC) 177.948 ms, Step2 (256 streams, 1 chunks) 82.084 ms
Hint Prep. Throughput 4.72 GB/sec
Throughput: 1343.28 GB/sec
Measurement completed. See the README for details on what the following fields mean.
Result:
{
  "offline": {
    "uploadBytes": 0,
    "downloadBytes": 0,
    "serverTimeMs": 176450,
    "clientTimeMs": 0,
    "simplepirPrepTimeMs": 1192,
    "simplepirHintBytes": 0,
    "doublepirHintBytes": 0
  },
  "online": {
    "uploadBytes": 1134592,
    "downloadBytes": 9437184,
    "simplepirQueryBytes": 0,
    "doublepirQueryBytes": 0,
    "simplepirRespBytes": 0,
    "doublepirRespBytes": 0,
    "serverTimeMs": 268,
    "clientQueryGenTimeMs": 1902,
    "clientDecodeTimeMs": 6,
    "firstPassTimeMs": 262,
    "secondPassTimeMs": 0,
    "ringPackingTimeMs": 0,
    "sqrtNBytes": 0,
    "allServerTimesMs": [
      268
    ],
    "stdDevServerTimeMs": 0.0
  }
}


# new corrected TC code
# doublepir
Hint Prep. Throughput 25.00 GB/sec
Throughput: 1454.55 GB/sec
Step1 10.932 ms, Steps2-5 (parallel, 256 streams) 149.678 ms
Measurement completed. See the README for details on what the following fields mean.
Result:
{
  "offline": {
    "uploadBytes": 0,
    "downloadBytes": 0,
    "serverTimeMs": 16878,
    "clientTimeMs": 0,
    "simplepirPrepTimeMs": 160,
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
    "serverTimeMs": 176,
    "clientQueryGenTimeMs": 1045,
    "clientDecodeTimeMs": 0,
    "firstPassTimeMs": 0,
    "secondPassTimeMs": 0,
    "ringPackingTimeMs": 0,
    "sqrtNBytes": 65536,
    "allServerTimesMs": [
      176
    ],
    "stdDevServerTimeMs": 0.0
  }
}

# ypir-word cdks
RUST_LOG=debug cargo run --release --features cuda -- 65536 737280 64 0 --word
Word Step1(TC) 27.822 ms, Step2 (71 streams, 1 chunks) 392.933 ms
Hint Prep. Throughput 8.01 GB/sec
Throughput: 831.41 GB/sec
Measurement completed. See the README for details on what the following fields mean.
Result:
{
  "offline": {
    "uploadBytes": 0,
    "downloadBytes": 0,
    "serverTimeMs": 125307,
    "clientTimeMs": 0,
    "simplepirPrepTimeMs": 702,
    "simplepirHintBytes": 0,
    "doublepirHintBytes": 0
  },
  "online": {
    "uploadBytes": 997376,
    "downloadBytes": 18874368,
    "simplepirQueryBytes": 0,
    "doublepirQueryBytes": 0,
    "simplepirRespBytes": 0,
    "doublepirRespBytes": 0,
    "serverTimeMs": 433,
    "clientQueryGenTimeMs": 1052,
    "clientDecodeTimeMs": 12,
    "firstPassTimeMs": 431,
    "secondPassTimeMs": 0,
    "ringPackingTimeMs": 0,
    "sqrtNBytes": 0,
    "allServerTimesMs": [
      433
    ],
    "stdDevServerTimeMs": 0.0
  }
}
# ypir-word inspiring
RUST_LOG=debug cargo run --release --features cuda -- 65536 737280 64 0 --word --inspiring
Hint Prep. Throughput 8.01 GB/sec
Throughput: 1782.18 GB/sec
Word InspiRING Step1(TC) 25.604 ms, Step2 (256 streams, 1 chunks) 169.113 ms
Measurement completed. See the README for details on what the following fields mean.
Result:
{
  "offline": {
    "uploadBytes": 0,
    "downloadBytes": 0,
    "serverTimeMs": 339356,
    "clientTimeMs": 0,
    "simplepirPrepTimeMs": 702,
    "simplepirHintBytes": 0,
    "doublepirHintBytes": 0
  },
  "online": {
    "uploadBytes": 610304,
    "downloadBytes": 18874368,
    "simplepirQueryBytes": 0,
    "doublepirQueryBytes": 0,
    "simplepirRespBytes": 0,
    "doublepirRespBytes": 0,
    "serverTimeMs": 202,
    "clientQueryGenTimeMs": 1003,
    "clientDecodeTimeMs": 12,
    "firstPassTimeMs": 199,
    "secondPassTimeMs": 0,
    "ringPackingTimeMs": 0,
    "sqrtNBytes": 0,
    "allServerTimesMs": [
      202
    ],
    "stdDevServerTimeMs": 0.0
  }
}