# Baosteel Multi-base Procurement Replication Package

This repository contains the reproducibility package for a multi-base
collaborative iron-ore procurement optimization study. It provides a
GA-hybrid solver, reviewer-facing algorithm snippets, a complete small-scale
instance, typical running logs, and source data for supplementary figures such
as convergence curves and ablation comparisons.

The package is organized so that reviewers can inspect the core algorithm
without reading the whole project, while still being able to rerun a small
instance end to end.

## Repository Structure

```text
.
├── configs/                    # JSON configs for runnable experiments
├── data/
│   ├── raw_excel_optional/      # Excel inputs used by the runnable code
│   └── small_complete_case/     # Complete 2-base, 16-supplier CSV instance
├── docs/                       # Model, algorithm, data, and experiment notes
├── examples/                   # Simple shell entry points
├── experiments/                # Copied experiment scripts from the paper work
├── figures/
│   ├── source_data/             # Raw data for supplementary figures
│   ├── scripts/                 # Figure-generation scripts
│   └── output/                  # Representative generated figures
├── logs/                       # Typical readable and JSONL running traces
├── results/                    # Curated convergence, ablation, runtime outputs
├── scripts/                    # GitHub-friendly one-command utilities
├── src/
│   ├── ga_hybrid/               # Runnable GA-hybrid implementation
│   └── snippets/                # Reviewer-facing code excerpts
└── tests/                      # Placeholder for future smoke/unit tests
```

`outputs/` is retained only as a compatibility copy for original experiment
scripts. New curated artifacts should be read from `results/`, `figures/`, and
`logs/`.

## Environment Installation

Python 3.9 or later is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The exact MIP decoder and lower-bound scripts require a working Gurobi
installation and license. The greedy decoder and most data/plot inspection
steps do not require Gurobi.

Conda users may instead run:

```bash
conda env create -f environment.yml
conda activate baosteel-procurement-replication
```

## Data Description

Two data forms are provided:

- `data/raw_excel_optional/`: copied Excel workbooks used by the original code.
- `data/small_complete_case/`: a complete small-scale CSV instance for review
  and reproduction. It includes base parameters, supplier parameters, demand,
  safety stock, compatibility, transport costs, lead times, long-contract
  suppliers, and discount parameters `R_j`, `Q_j`, `r_j`, and `P_j_tau`.

See `docs/data_dictionary.md` for field-level definitions.

## Running the Solver

Run a small smoke instance directly:

```bash
python3 src/ga_hybrid/main.py \
  --num-bases 2 \
  --num-suppliers 16 \
  --pop-size 4 \
  --max-gen 4 \
  --k-explore 2 \
  --k-ls 2 \
  --elite-ls 1 \
  --ls-max-iter 1 \
  --instance-name quickstart_2x16
```

Or run through a JSON config:

```bash
python3 scripts/run_from_config.py configs/academic_trace_2x16.json
```

Generated event logs are written to `logs/generated/`.

## Experiment Reproduction

Common entry points:

```bash
# Complete small case
bash scripts/run_small_case.sh

# Typical academic trace with decoder switching and local search
bash scripts/run_academic_trace.sh

# Ablation experiment script
bash scripts/run_ablation.sh

# Supplier-count sensitivity experiment
bash scripts/run_supplier_sensitivity.sh

# Summarize available curated result files
python3 scripts/summarize_results.py
```

The lightweight configurations in `configs/` are intended for quick verification.
Full paper-scale settings can be created by increasing population size,
generation count, supplier count, and time limits.

## Output Results

Important curated outputs:

- `logs/typical_run_excerpt.log`: manuscript-ready trace excerpt.
- `logs/typical_run_full.log`: full readable trace.
- `logs/typical_run_events.jsonl`: machine-readable event trace.
- `results/ablation/ablation_raw_results.csv`: seed-level ablation results.
- `results/ablation/ablation_summary.csv`: derived ablation summary.
- `results/convergence/convergence_history.csv`: convergence raw history.
- `results/runtime/runtime_distribution_summary.csv`: runtime summary.
- `figures/source_data/`: source CSVs for supplementary figures.

## Reviewer Material Map

| Reviewer request | Repository path |
|---|---|
| Core algorithm code snippets | `src/snippets/` |
| Runnable implementation | `src/ga_hybrid/main.py` |
| Complete small-scale instance | `data/small_complete_case/` |
| Typical algorithm log | `logs/typical_run_excerpt.log` |
| Full trace | `logs/typical_run_full.log`, `logs/typical_run_events.jsonl` |
| Supplementary figure source data | `figures/source_data/` |

## Notes

Some large or generated outputs are intentionally ignored by `.gitignore` for
future runs. The curated artifacts already included in this package remain in
their documented folders.
