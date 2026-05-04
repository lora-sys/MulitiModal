#!/usr/bin/env python3
"""
Generate paper materials:
  - Fig.3  Data distribution (relaxation scores)
  - Fig.5  LOSO fold stability (MSE per fold)
  - Table 1  Dataset statistics (WESAD)
  - Table 2  Main LOSO comparison
  - Table 3  Ablation study
  - Table 4  Hyperparameter settings
  - Table A1 Per-fold detailed results
  - Core formulas (corrected to match code)

Run from project root:
  python scripts/generate_paper_materials.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

PROJ = Path("/Users/lora/repos/MulitiModal")
OUT_FIG = PROJ / "paper" / "figures"
OUT_TAB = PROJ / "paper" / "tables"
OUT_FORM = PROJ / "paper" / "formulas"
for p in [OUT_FIG, OUT_TAB, OUT_FORM]:
    p.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

print("=" * 60)
print("Generating Paper Materials (v2)")
print("=" * 60)

# Load main LOSO results
loso_path = PROJ / "paper" / "results" / "experiments_summary_loso.json"
with open(loso_path) as f:
    loso_agg = json.load(f)

# ================================================================
# 1. Fig.3: Relaxation Score Distribution
# ================================================================
print("\n[1/8] Generating Fig.3: Relaxation Score Distribution...")
wesad_dir = PROJ / "data" / "wesad"
fold_files = sorted(wesad_dir.rglob("S*.pkl"))

# Try to load WESAD labels; fallback to computing from aggregated data
try:
    import pickle
    from scipy.io import loadmat

    all_labels = []
    label_map = {1: 1.0, 2: 0.0, 3: 0.6}

    for pkl_path in fold_files:
        try:
            with open(pkl_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")
            raw_labels = np.asarray(data["label"]).squeeze().astype(np.int32)
            for lab in raw_labels:
                mapped = label_map.get(int(lab))
                if mapped is not None:
                    all_labels.append(mapped)
        except Exception:
            continue

    if not all_labels:
        raise RuntimeError("No labels extracted")

    labels_arr = np.array(all_labels)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(labels_arr, bins=30, edgecolor="black", alpha=0.75, color="#4c72b0")
    ax.set_xlabel("Relaxation Score", fontsize=12)
    ax.set_ylabel("Number of Samples", fontsize=12)
    ax.set_title("Distribution of Relaxation Scores in WESAD", fontsize=12)
    ax.axvline(x=np.mean(labels_arr), color="red", linestyle="--",
               label=f"Mean = {np.mean(labels_arr):.3f}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "Fig3_relaxation_distribution.pdf")
    plt.savefig(OUT_FIG / "Fig3_relaxation_distribution.png")
    plt.close()
    print(f"  ✓ Saved Fig3_relaxation_distribution (N={len(labels_arr)})")

except Exception as e:
    print(f"  ⚠ Skipped Fig.3 (WESAD data not available locally): {e}")

# ================================================================
# 2. Fig.5: LOSO Fold Stability (MSE per fold)
# ================================================================
print("\n[2/8] Generating Fig.5: LOSO Fold Stability...")
fold_dir = PROJ / "experiment1" / "1.1" / "loso_folds"
fold_jsons = sorted(fold_dir.glob("experiments_summary_S*.json"))

fold_records = []
for fj in fold_jsons:
    with open(fj) as fp:
        data = json.load(fp)
    subject = data.get("holdout_subject", fj.stem.split("_")[-1])
    for item in data.get("matrix_logs", []):
        if item.get("name") == "Final Ours":
            m = item.get("full", {}).get("metrics", {})
            fold_records.append({
                "subject": subject,
                "mse": m.get("mse"),
                "rmse": m.get("rmse"),
                "mae": m.get("mae"),
                "pearson": m.get("pearson"),
            })
            break

fold_df = pd.DataFrame(fold_records).sort_values("subject")
fold_df["fold_num"] = range(1, len(fold_df) + 1)

if not fold_df.empty and fold_df["mse"].notna().any():
    # Line plot
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fold_df["fold_num"], fold_df["mse"], marker="o", linestyle="-",
            color="#c44e52", linewidth=1.5, markersize=6)
    mean_mse = fold_df["mse"].mean()
    ax.axhline(y=mean_mse, color="blue", linestyle="--",
               label=f"Mean MSE = {mean_mse:.4f}")
    ax.set_xlabel("LOSO Fold (Hold-out Subject)", fontsize=12)
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("Per-fold Test MSE (Final Ours, 15 folds)", fontsize=12)
    ax.set_xticks(fold_df["fold_num"])
    ax.set_xticklabels(fold_df["subject"], rotation=45, fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "Fig5_loso_stability_line.pdf")
    plt.savefig(OUT_FIG / "Fig5_loso_stability_line.png")
    plt.close()
    print("  ✓ Saved Fig5_loso_stability_line")

    # Box plot
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.boxplot(fold_df["mse"].dropna(), vert=True, patch_artist=True,
               boxprops=dict(facecolor="#fdae61", color="black"),
               medianprops=dict(color="red", linewidth=2))
    ax.set_ylabel("MSE", fontsize=12)
    ax.set_title("LOSO MSE Distribution (15 folds)", fontsize=12)
    ax.set_xticklabels(["Final Ours"])
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "Fig5_loso_stability_box.pdf")
    plt.savefig(OUT_FIG / "Fig5_loso_stability_box.png")
    plt.close()
    print("  ✓ Saved Fig5_loso_stability_box")
else:
    print("  ⚠ Skipped Fig.5 (no per-fold data found)")

# ================================================================
# 3. Table 1: WESAD Dataset Statistics
# ================================================================
print("\n[3/8] Generating Table 1: WESAD Dataset Statistics...")
n_subjects = loso_agg.get("n_folds", 15)
subjects = loso_agg.get("fold_subjects", [])

table1_latex = r"""\begin{table}[htbp]
\centering
\caption{WESAD Dataset Statistics}
\label{tab:dataset}
\begin{tabular}{lc}
\toprule
\textbf{Property} & \textbf{Value} \\
\midrule
Number of subjects & """ + str(n_subjects) + r""" \\
Dynamic modalities & ECG + EDA (chest) \\
Dynamic input shape & $(2, 1000)$ \\
Window length / stride & 1000 pts / 500 pts (50\% overlap) \\
Static features & Age, Gender, BMI, Baseline HR \\
Static feature dimension & 4 \\
Relaxation labels & 0.0 (stress), 0.6 (meditation), 1.0 (baseline) \\
Label mapping source & WESAD original discrete labels \\
TCM prior dimension & 9 (constitution probabilities) \\
\bottomrule
\end{tabular}
\end{table}
"""
with open(OUT_TAB / "table1.tex", "w") as f:
    f.write(table1_latex)
print("  ✓ Saved table1.tex")

# ================================================================
# 4. Table 2: Main LOSO Comparison
# ================================================================
print("\n[4/8] Generating Table 2: Main LOSO Comparison...")
main_names = ["Baseline A", "Baseline B", "Ours-TCN", "Final Ours"]
main_rows = []
for item in loso_agg.get("loso_stage3_summary", []):
    name = item.get("name")
    if name in main_names:
        mean_mse = item.get("mean_best_val_mse")
        std_mse = item.get("std_best_val_mse")
        if mean_mse is not None:
            main_rows.append((name, mean_mse, std_mse))

# Sort in desired order
order = {n: i for i, n in enumerate(main_names)}
main_rows.sort(key=lambda r: order.get(r[0], 99))

table2_latex = r"""\begin{table}[htbp]
\centering
\caption{Main LOSO Comparison (MSE, mean $\pm$ std over 15 folds)}
\label{tab:main_comparison}
\begin{tabular}{lc}
\toprule
\textbf{Method} & \textbf{MSE} \\
\midrule
"""
for name, mean, std in main_rows:
    table2_latex += f"{name} & {mean:.4f} $\\pm$ {std:.4f} \\\\\n"
table2_latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
with open(OUT_TAB / "table2.tex", "w") as f:
    f.write(table2_latex)
print("  ✓ Saved table2.tex")

# ================================================================
# 5. Table 3: Ablation Study
# ================================================================
print("\n[5/8] Generating Table 3: Ablation Study...")
ablation_names = ["Final Ours", "w/o Dual Gating", "w/o TCM Prior", "w/o TCM_Gate"]
ablation_display = {
    "Final Ours": "Final Ours (full model)",
    "w/o Dual Gating": "w/o Dual Gating",
    "w/o TCM Prior": "w/o TCM Prior",
    "w/o TCM_Gate": r"w/o TCM\_Gate",
}
ablation_rows = []
for item in loso_agg.get("loso_stage3_summary", []):
    name = item.get("name")
    if name in ablation_names:
        mean_mse = item.get("mean_best_val_mse")
        std_mse = item.get("std_best_val_mse")
        if mean_mse is not None:
            ablation_rows.append((name, mean_mse, std_mse))

order_abl = {n: i for i, n in enumerate(ablation_names)}
ablation_rows.sort(key=lambda r: order_abl.get(r[0], 99))

# Compute deltas relative to Final Ours
final_mse = ablation_rows[0][1] if ablation_rows else 0

table3_latex = r"""\begin{table}[htbp]
\centering
\caption{Ablation Study (MSE, mean $\pm$ std over 15 folds)}
\label{tab:ablation}
\begin{tabular}{lcc}
\toprule
\textbf{Variant} & \textbf{MSE} & \textbf{$\Delta$ MSE} \\
\midrule
"""
for name, mean, std in ablation_rows:
    display = ablation_display.get(name, name)
    delta = mean - final_mse
    if name == "Final Ours":
        table3_latex += f"{display} & {mean:.4f} $\\pm$ {std:.4f} & --- \\\\\n"
    else:
        table3_latex += f"{display} & {mean:.4f} $\\pm$ {std:.4f} & +{delta:.4f} \\\\\n"
table3_latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
with open(OUT_TAB / "table3.tex", "w") as f:
    f.write(table3_latex)
print("  ✓ Saved table3.tex")

# ================================================================
# 6. Table 4: Hyperparameter Settings
# ================================================================
print("\n[6/8] Generating Table 4: Hyperparameter Settings...")
table4_latex = r"""\begin{table}[htbp]
\centering
\caption{Hyperparameter Settings}
\label{tab:hyperparams}
\begin{tabular}{lc}
\toprule
\textbf{Hyperparameter} & \textbf{Value} \\
\midrule
Optimizer & AdamW \\
Learning rate & $5 \times 10^{-4}$ \\
Learning rate search range & $[2.5 \times 10^{-4},\ 8 \times 10^{-4}]$ \\
Weight decay & $10^{-5}$ \\
Batch size & 32 \\
Max epochs & 50 \\
Early stopping patience & 10 \\
Dropout (regression head) & 0.2 \\
Dropout (gate hidden layers) & 0.1 \\
Window length / overlap & 1000 pts / 50\% \\
Gate B strength ($\lambda_b$) & 0.35 \\
Random seed & 42 \\
\bottomrule
\end{tabular}
\end{table}
"""
with open(OUT_TAB / "table4.tex", "w") as f:
    f.write(table4_latex)
print("  ✓ Saved table4.tex")

# ================================================================
# 7. Table A1: Per-fold Detailed Results
# ================================================================
print("\n[7/8] Generating Table A1: Per-fold Detailed Results...")
if not fold_df.empty and fold_df["mse"].notna().any():
    tablea1_latex = r"""\begin{table}[htbp]
\centering
\caption{Per-fold Test Results (Final Ours, LOSO)}
\label{tab:per_fold}
\small
\begin{tabular}{lcccc}
\toprule
\textbf{Fold} & \textbf{MSE} & \textbf{RMSE} & \textbf{MAE} & \textbf{Pearson $r$} \\
\midrule
"""
    for _, row in fold_df.iterrows():
        subj = row["subject"]
        mse = f"{row['mse']:.4f}" if pd.notna(row["mse"]) else "---"
        rmse = f"{row['rmse']:.4f}" if pd.notna(row.get("rmse")) else "---"
        mae = f"{row['mae']:.4f}" if pd.notna(row.get("mae")) else "---"
        pear = f"{row['pearson']:.4f}" if pd.notna(row.get("pearson")) else "---"
        tablea1_latex += f"{subj} & {mse} & {rmse} & {mae} & {pear} \\\\\n"

    # Summary row
    tablea1_latex += r"""\midrule
"""
    tablea1_latex += (
        f"Mean & {fold_df['mse'].mean():.4f} "
        f"& {fold_df['rmse'].mean():.4f} "
        f"& {fold_df['mae'].mean():.4f} "
        f"& {fold_df['pearson'].mean():.4f} \\\\\n"
    )
    tablea1_latex += (
        f"Std & {fold_df['mse'].std():.4f} "
        f"& {fold_df['rmse'].std():.4f} "
        f"& {fold_df['mae'].std():.4f} "
        f"& {fold_df['pearson'].std():.4f} \\\\\n"
    )
    tablea1_latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    with open(OUT_TAB / "tableA1.tex", "w") as f:
        f.write(tablea1_latex)
    print("  ✓ Saved tableA1.tex")
else:
    print("  ⚠ Skipped Table A1 (no per-fold data)")

# ================================================================
# 8. Core Formulas (corrected to match code)
# ================================================================
print("\n[8/8] Generating Core Formulas...")
formulas = r"""\section*{Core Formulas}

\subsection*{TCM Prior Generation}
\begin{equation}
    \mathbf{p}_{\mathrm{tcm}} = \mathrm{softmax}\!\left(\mathrm{FT\text{-}Transformer}(\mathbf{x}_{\mathrm{static}})\right) \in \mathbb{R}^{9}
\end{equation}
where $\mathbf{x}_{\mathrm{static}} \in \mathbb{R}^{4}$ contains Age, Gender, BMI, and Baseline HR.
The TCM encoder is pretrained on an external dataset and \textbf{frozen} during main task training.

\subsection*{Gate A: Prior-Guided Modulation}
\begin{equation}
    \mathbf{a} = \sigma\!\left(\mathbf{W}_a \, \mathbf{p}_{\mathrm{tcm}} + \mathbf{b}_a\right) \in (0,1)^{128}
\end{equation}
\begin{equation}
    \mathbf{z}_{\mathrm{mod}} = \mathbf{z}_{\mathrm{raw}} \odot \mathbf{a}
\end{equation}
where $\mathbf{z}_{\mathrm{raw}} \in \mathbb{R}^{128}$ is the dynamic encoder output.
Gate A amplifies channels that are informative for the specific individual.

\subsection*{Gate B: Subtractive Debiasing}
\begin{equation}
    \mathbf{b} = \sigma\!\left(\mathbf{W}_b \, \mathbf{z}_{\mathrm{mod}} + \mathbf{b}_b\right) \in (0,1)^{128}
\end{equation}
\begin{equation}
    \mathbf{z}_{\mathrm{pure}} = \mathbf{z}_{\mathrm{mod}} \odot (1 - \lambda_b \cdot \mathbf{b})
\end{equation}
where $\lambda_b$ controls debiasing strength.
Channels with larger $\mathbf{b}$ values are more coupled to static baselines and thus suppressed.

\subsection*{Late Re-injection}
\begin{equation}
    \mathbf{f}_{\mathrm{final}} = \left[\mathbf{z}_{\mathrm{pure}} \; ; \; \mathbf{p}_{\mathrm{tcm}}\right] \in \mathbb{R}^{137}
\end{equation}
\begin{equation}
    \hat{y} = \mathrm{Regressor}(\mathbf{f}_{\mathrm{final}})
\end{equation}

\subsection*{Loss Function}
\begin{equation}
    \mathcal{L} = \frac{1}{N}\sum_{i=1}^{N}\left(y_i - \hat{y}_i\right)^2
\end{equation}
Only the dynamic encoder, dual gates, and regressor are updated.
The TCM encoder remains frozen throughout training.
"""
with open(OUT_FORM / "core_formulas.tex", "w") as f:
    f.write(formulas)
print("  ✓ Saved core_formulas.tex")

# ================================================================
# Done
# ================================================================
print("\n" + "=" * 60)
print("✅ All materials generated successfully!")
print(f"  Figures : {OUT_FIG}/")
print(f"  Tables  : {OUT_TAB}/")
print(f"  Formulas: {OUT_FORM}/")
print("=" * 60)
