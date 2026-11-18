# Workspace dataset check

This committed inventory describes the permitted development partitions only. The generated row-level inventory is in `artifacts/exploration/raw_inventory.csv`; the partition summary is in `artifacts/exploration/partition_summary.csv`.

| Partition | Trips | GPS observations | Mean observations/trip | Node rows | Arc rows | Route rows |
|---|---:|---:|---:|---:|---:|---:|
| Training | 80 | 208,417 | 2,605.21 | 1,933,485 | 3,876,904 | 31,336 |
| Student test | 10 | 19,160 | 1,916.00 | 90,689 | 185,041 | 3,477 |

The evaluator partition is provided separately and must remain on Laptop 1. Its trip records and evaluator-derived scores are omitted from the public repository. Dataset counts describe files and do not independently verify route labels.
