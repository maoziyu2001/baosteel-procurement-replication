# Ablation Experiment Outputs

This folder stores the outputs for paper Section 3.4, "算法组件消融实验".

- `Full-GA-hybrid`: complete GA-hybrid with staged greedy/solver decoding, mixed mutation, and all local-search neighborhoods.
- `w/o staged decoding`: removes staged decoding and uses greedy decoding only.
- `solver-only decoding`: uses mathematical-programming decoding throughout; set a solver time limit for large cases.
- `w/o local search`: disables the local-search improvement step.
- `adaptive mutation only`: keeps only `mutate2`, the adaptive supplier-transfer mutation.
- `random mutation only`: keeps only `mutate1`, the random bit-flip mutation.
- `w/o single_flip neighborhood`: removes the single purchase-decision flip neighborhood.
- `w/o single_base_consolidation neighborhood`: removes same-base supplier consolidation.
- `w/o cross_period_consolidation neighborhood`: removes cross-period supplier consolidation.

Run the experiment with:

```bash
python3 experiments/run_ablation.py --cases case_2 case_10 case_13 case_20 --seeds 0 1 2 3 4
```

Main outputs:

- `ablation_raw_results.csv`: seed-level results for every case and variant.
- `ablation_summary.csv`: mean/std/min/max summaries by case and variant.
- `ablation_table_latex.tex`: manuscript-ready LaTeX tables.
- `figures/convergence_*.png`: convergence curves generated with matplotlib.
