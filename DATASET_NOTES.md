# Workspace dataset check

Observed directly from the files present in this workspace on 2026-11-18. The generated row-level inventory is in `artifacts/exploration/raw_inventory.csv`.

| Partition | Trips | GPS observations | Mean observations/trip | Node rows | Arc rows | Route rows |
|---|---:|---:|---:|---:|---:|---:|
| Training | 80 | 208,417 | 2,605.21 | 1,933,485 | 3,876,904 | 31,336 |
| Student test | 10 | 19,160 | 1,916.00 | 90,689 | 185,041 | 3,477 |
| Evaluator test | 0 | 0 | N/A | 0 | 0 | 0 |

Each supplied train and student-test trip directory has one `.track`, `.nodes`, `.arcs`, and `.route` file. No evaluator-test trip directories are present. Dataset counts describe files, not independent validation of the route labels. See the report template for remaining analysis and interpretation tasks.
