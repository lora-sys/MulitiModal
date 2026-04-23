# Paper Artifacts (Auto-collected)

This folder is a lightweight, venue-agnostic bundle of the minimum artifacts needed to draft and reproduce results.

## What To Cite In The Paper

- Protocol: Leave-One-Subject-Out (LOSO) on WESAD.
- Label mapping: {1: 1.0, 2: 0.0, 3: 0.6}, transition label 0 dropped.
- Windowing: window_size=1000, overlap=0.5 (stride=500).
- TCM prior: 4D static -> scaler -> frozen FT-Transformer -> 9D probability (eval + no_grad + requires_grad=False).
- Paper-safe gate setting: Gate A disabled by default in LOSO fixed-encoder runs (gate_a_scale=0.0).

## Key Outputs

- Main tables: `tables/loso_main_table.csv` and `tables/loso_main_table.tsv`
- LOSO summary json: `results/experiments_summary_loso.json`
- Per-fold results: `results/loso_folds/experiments_summary_S*.json`
- Figures: `figures/fig1_loso_comparison.*`, `figures/fig3_loso_ablation.*`

## Re-run Command (Paper-safe)

```bash
nohup python3 -u run_experiments.py \
  --protocol loso \
  --fixed-encoder inceptiontime \
  --no-fold-search \
  --override-params "lr=5e-4,weight_decay=1e-5,batch_size=32" \
  --gate-b-scale 0.1 \
  --final-lr-mult 0.7 \
  --epochs 50 \
  > logs/loso_paper_safe_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo "PID=$!"
```

