from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from src.config import ROOT


plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 14,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 300,
    }
)

FIG_DIR = ROOT / "figures"


def _ensure_fig_dir() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    return FIG_DIR


def _save_all_formats(fig: plt.Figure, stem: str) -> Path:
    fig_dir = _ensure_fig_dir()
    png = fig_dir / f"{stem}.png"
    svg = fig_dir / f"{stem}.svg"
    pdf = fig_dir / f"{stem}.pdf"
    fig.savefig(png, format="png")
    fig.savefig(svg, format="svg")
    fig.savefig(pdf, format="pdf")
    return png


def plot_comparison(data: List[Dict]) -> Path:
    """1x2 figure: absolute MSE bars + relative error reduction bars."""
    fig_dir = _ensure_fig_dir()
    ordered_names = ["Baseline A", "Baseline B", "Ours"]
    m = {d["name"]: d["metrics"]["mse"] for d in data}
    labels = [n for n in ordered_names if n in m]
    mse_vals = [m[n] for n in labels]

    if len(labels) < 3:
        raise ValueError("plot_comparison requires Baseline A/B/Ours results.")

    baseline_a = mse_vals[0]
    reduction = [0.0]
    for v in mse_vals[1:]:
        reduction.append((baseline_a - v) / max(baseline_a, 1e-12) * 100.0)

    colors = ["#7f8c8d", "#3498db", "#c0392b"]
    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    bars0 = axes[0].bar(x, mse_vals, color=colors)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("MSE")
    axes[0].set_title("Fig.1(a) Absolute MSE")
    for b, v in zip(bars0, mse_vals):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.4f}", ha="center", va="bottom", fontsize=11)

    bars1 = axes[1].bar(x, reduction, color=colors)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Error Reduction (%)")
    axes[1].set_title("Fig.1(b) Error Reduction vs Baseline A")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    for b, v in zip(bars1, reduction):
        axes[1].text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.2f}%", ha="center", va="bottom", fontsize=11)

    fig.tight_layout()
    out = _save_all_formats(fig, "fig1_comparison")
    plt.close(fig)
    return out


def plot_selection(data: List[Dict]) -> Path:
    """5 encoders; best highlighted red and others gray."""
    fig_dir = _ensure_fig_dir()
    labels = [d["name"] for d in data]
    mse_vals = [d["metrics"]["mse"] for d in data]
    if len(labels) != 5:
        raise ValueError("plot_selection requires exactly 5 encoder results.")

    best_idx = int(np.argmin(mse_vals))
    colors = ["#bdc3c7"] * len(labels)
    colors[best_idx] = "#c0392b"
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(x, mse_vals, color=colors)
    ax.set_xticks(x, labels)
    ax.set_ylabel("MSE")
    ax.set_title("Fig.2 Encoder Selection")
    for i, (b, v) in enumerate(zip(bars, mse_vals)):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.4f}", ha="center", va="bottom", fontsize=10)
        if i == best_idx:
            ax.text(b.get_x() + b.get_width() / 2, -0.06 * max(mse_vals), "Best", ha="center", va="top", color="#c0392b", fontsize=11)

    ax.set_ylim(bottom=-0.1 * max(mse_vals))
    fig.tight_layout()
    out = _save_all_formats(fig, "fig2_encoder_selection")
    plt.close(fig)
    return out


def plot_ablation(data: List[Dict]) -> Path:
    """4 bars, sorted by MSE ascending, green->red gradient."""
    fig_dir = _ensure_fig_dir()
    items = sorted(data, key=lambda d: d["metrics"]["mse"])
    if len(items) != 4:
        raise ValueError("plot_ablation requires Full Model + 3 ablations.")

    labels = [d["name"] for d in items]
    mse_vals = np.array([d["metrics"]["mse"] for d in items], dtype=float)

    c0 = np.array([0x27, 0xAE, 0x60]) / 255.0  # green
    c1 = np.array([0xC0, 0x39, 0x2B]) / 255.0  # red
    t = np.linspace(0.0, 1.0, len(labels))
    colors = [tuple((1 - a) * c0 + a * c1) for a in t]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(x, mse_vals, color=colors)
    ax.set_xticks(x, labels)
    ax.set_ylabel("MSE")
    ax.set_title("Fig.3 Ablation Study (sorted by MSE)")
    for b, v in zip(bars, mse_vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{v:.4f}", ha="center", va="bottom", fontsize=10)

    fig.tight_layout()
    out = _save_all_formats(fig, "fig3_ablation")
    plt.close(fig)
    return out
