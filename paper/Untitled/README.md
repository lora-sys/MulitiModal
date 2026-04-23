# Paper Pack (WESAD LOSO + BUT Cross-domain)

This folder is the "teacher-ready" bundle: protocols, evidence, plots, and exact commands to reproduce.

## Why Are We Doing This?

Most physiological deep models treat everyone as a blank slate and learn shortcuts from subject identity. That often yields good within-dataset scores but weak generalization.

Our goal is to prove a stronger claim:
- The model can use a **frozen TCM prior** (static constitution) to represent **individual differences**.
- The temporal encoder focuses on **dynamic physiological responses** rather than memorizing subjects.
- The learned representation remains useful **across domains** (WESAD -> BUT).

## What Is Different vs Typical Work?

- **Not a pure black box**: we explicitly expose a 9-D "constitution prior" and can probe its relationship with physiology.
- **Protocol rigor**: WESAD uses **LOSO (15 folds)** to prevent subject leakage. Backbone/hparams are chosen on a dev protocol and then frozen for LOSO.
- **Cross-domain evidence**: on BUT PPG we freeze the feature extractor and train only a small head; we also report a mechanism probe on the prior.

## Assets in This Folder

### Main (WESAD LOSO)
- Results JSON: `paper/Untitled/results/experiments_summary_loso.json`
- Plots:
  - Fig.1 LOSO comparison: `paper/Untitled/figures/fig1_loso_comparison.(png|pdf|svg)`
  - Fig.3 LOSO ablation: `paper/Untitled/figures/fig3_loso_ablation.(png|pdf|svg)`

### Dev (Encoder Selection)
- Fig.2 encoder selection: `experiment1/0.9/figures/fig2_encoder_selection.(png|pdf|svg)`
- Dev run summary: `experiment1/0.9/experiments_summary.json`

### Cross-domain (BUT)
- Aggregated metrics: `paper/Untitled/results/cross_domain_results.json`
- Per-seed table: `paper/Untitled/results/cross_domain_seed_metrics.tsv`
- Scatter plot (best seed): `paper/Untitled/results/fig_cross_domain_hr_probe.(png|pdf|svg)`

## Protocols (Paper-Grade)

### Labels and Windowing (WESAD)
- Label mapping (regression target): `{1: 1.0, 2: 0.0, 3: 0.6}` (drop label 0)
- Sliding window: `window_size=1000`, `overlap=0.5`

### Main Evaluation (WESAD)
- Protocol: **Leave-One-Subject-Out (LOSO)**, 15 folds
- Report: mean ± std across folds

### Development (Backbone / Hyperparams)
We do model selection on a development protocol (non-LOSO), then freeze choices for LOSO to avoid test-fold tuning.

Evidence from `experiment1/0.9/experiments_summary.json`:
- Encoders compared: `tcn`, `inceptiontime`, `os-cnn`, `xcm`, `1d-resnet`
- Best encoder on dev selection: **InceptionTime** (best val MSE = **0.09295**)
- Chosen conservative hyperparams (stage2): `lr=5e-4`, `weight_decay=1e-5`, `batch_size=32`

## Model Storyline (One Coherent Narrative)

1. **Static prior (TCM)** encodes individual differences:
   - Static 4-D features `[Age, Gender, BMI, Baseline_HR]`
   - Standardized with a fixed scaler (`scaler_params.npz`)
   - Passed through a frozen FT-Transformer to obtain 9-D probabilities
2. **Dynamic encoder** learns temporal physiological response (InceptionTime in the final LOSO runs).
3. **Gating / reinjection** uses the prior to modulate or contextualize the dynamic representation (ablation demonstrates contribution).
4. **Cross-domain HR probe** on BUT shows the representation transfers beyond WESAD:
   - Feature extractor frozen
   - Train a small head only
   - Report mean ± std across multiple seeds

## Key Results (Numbers You Can Quote)

### WESAD LOSO (15 folds)
From `paper/Untitled/results/experiments_summary_loso.json`:
- Baseline A: mean MSE = **0.1818 ± 0.0472**
- Baseline B: mean MSE = **0.1903 ± 0.0728**
- Final Ours: mean MSE = **0.1346 ± 0.1122**

Interpretation:
- Final Ours reduces MSE vs Baseline A by **25.93%** (Fig.1(b)).
- Baseline B is worse than Baseline A here (small-sample shortcut removal can hurt within-domain error).

### BUT Cross-domain HR Probe
From `paper/Untitled/results/cross_domain_results.json` (3 seeds, head=MLP):
- MAE = **9.96 ± 0.50 BPM**
- MSE = **164.33 ± 3.79**
- Pearson = **0.208 ± 0.178**

Mechanism probe (TCM probs vs HR):
- max |r| ≈ **0.517**

Notes:
- Negative correlation between WESAD relaxation and HR is physiologically plausible; for HR probe we report direct HR regression metrics.

## Repro Commands (Server)

### 1) WESAD: Full LOSO (15 folds) + auto-archive to paper/
```bash
cd /root/work/MulitiModal
mkdir -p logs

nohup python3 -u main_pipeline.py \
  --skip-optuna \
  --protocol loso \
  --wesad-dir /root/work/MulitiModal/data/wesad \
  --fixed-encoder inceptiontime \
  --no-fold-search \
  --override-params "lr=5e-4,weight_decay=1e-5,batch_size=32" \
  --epochs 50 \
  --gate-a-scale 0.0 \
  --gate-b-scale 0.10 \
  --final-lr-mult 0.70 \
  --tcm-checkpoint /root/work/MulitiModal/tcm_ft_transformer/checkpoints/best_model.pth \
  --tcm-scaler /root/work/MulitiModal/tcm_ft_transformer/scaler_params.npz \
  --output-json /root/work/MulitiModal/experiments_summary_loso.json \
  > logs/pipeline_loso_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $!
```

### 2) BUT: Cross-domain (3 seeds, stable head training) + archive to paper/
```bash
cd /root/work/MulitiModal
mkdir -p logs

nohup python3 -u main_pipeline.py \
  --only-cross-domain \
  --epochs 80 \
  --seeds 42,43,44 \
  --head mlp \
  --target-standardize \
  --loss huber \
  --huber-delta 5.0 \
  --head-weight-decay 1e-4 \
  --early-stop-patience 15 \
  --but-dir /root/work/MulitiModal/data/but_ppg \
  --tcm-checkpoint /root/work/MulitiModal/tcm_ft_transformer/checkpoints/best_model.pth \
  --tcm-scaler /root/work/MulitiModal/tcm_ft_transformer/scaler_params.npz \
  --strict-tcm-paths \
  --paper-dir /root/work/MulitiModal/paper \
  > logs/pipeline_cross_domain_strong_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $!
```

## What To Tell the Advisor (1-minute Summary)

- We validated the core claim under a leakage-safe protocol (WESAD LOSO 15 folds).
- Encoder/hparams are determined on a dev protocol, then frozen for LOSO.
- Full model outperforms baselines and all ablations (Fig.1 and Fig.3).
- Cross-domain HR probe on BUT uses a frozen extractor and achieves MAE ~10 BPM (3 seeds), plus a mechanism probe showing TCM prior correlates with HR.

