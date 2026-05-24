# Configuration files

Configurations are JSON files consumed by `scripts/run_from_config.py`.
Keys under `args` are converted to command-line flags.

Examples:

```bash
python3 scripts/run_from_config.py configs/small_case_2x16.json
python3 scripts/run_from_config.py configs/academic_trace_2x16.json
```

The `paper_case_4x35.json` file is a paper-scale template and may require a
working Gurobi license and a longer runtime.

