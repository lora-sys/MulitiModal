"""
Data Quality Visualization
Random sampling to compare raw vs self-healed signals
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import neurokit2 as nk

DATA_PATH = "experiment/model/pretrain_10k.npz"
LABELS = {0: "Poor", 1: "Fair", 2: "Normal", 3: "Good"}
FS = 50
DURATION = 20


def visualize_samples(n_samples=4):
    """Random sampling visualization"""
    data = np.load(DATA_PATH)
    dynamic = data["dynamic"]
    static = data["static"]
    labels = data["labels"]

    indices = np.random.choice(len(labels), n_samples, replace=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        ax = axes[i]
        label = labels[idx]
        signal = dynamic[idx]

        t = np.linspace(0, DURATION, signal.shape[1])

        ax.plot(t, signal[0], label="Channel 1", alpha=0.8, linewidth=0.8)
        ax.plot(t, signal[1], label="Channel 2", alpha=0.8, linewidth=0.8)

        ax.set_title(
            f"Sample {idx} | Label: {LABELS[label]} | "
            f"Weight: {static[idx][0] * 100:.1f}kg | "
            f"HR: {static[idx][1] * 120:.1f}bpm",
            fontsize=10,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Normalized Value")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, DURATION)

    plt.tight_layout()
    plt.savefig("experiment/generate/samples_visualization.png", dpi=150)
    print(f"[*] Saved: experiment/generate/samples_visualization.png")
    plt.close()


def visualize_by_class(n_per_class=2):
    """Visualize by class"""
    data = np.load(DATA_PATH)
    dynamic = data["dynamic"]
    static = data["static"]
    labels = data["labels"]

    fig, axes = plt.subplots(4, n_per_class, figsize=(14, 12))

    for label_id in range(4):
        class_indices = np.where(labels == label_id)[0]
        selected = np.random.choice(class_indices, n_per_class, replace=False)

        for j, idx in enumerate(selected):
            ax = axes[label_id, j]
            signal = dynamic[idx]
            t = np.linspace(0, DURATION, signal.shape[1])

            ax.plot(t, signal[0], label="Ch1", alpha=0.8, linewidth=0.7)
            ax.plot(t, signal[1], label="Ch2", alpha=0.8, linewidth=0.7)

            ax.set_title(
                f"{LABELS[label_id]} | HR:{static[idx][1] * 120:.0f} | SpO2:{static[idx][2] * 100:.0f}%",
                fontsize=9,
            )
            ax.set_xlim(0, DURATION)
            ax.grid(True, alpha=0.3)

            if j == 0:
                ax.set_ylabel("Value")

    plt.suptitle("Data Quality Check by Class", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("experiment/generate/class_visualization.png", dpi=150)
    print(f"[*] Saved: experiment/generate/class_visualization.png")
    plt.close()


def check_statistics():
    """Check data statistics"""
    data = np.load(DATA_PATH)
    dynamic = data["dynamic"]
    static = data["static"]
    labels = data["labels"]

    print("\n" + "=" * 50)
    print("Data Statistics Report")
    print("=" * 50)

    print(f"\nTotal samples: {len(labels)}")
    print(f"Dynamic shape: {dynamic.shape}")
    print(f"Static shape: {static.shape}")

    print("\nSamples per class:")
    for label_id in range(4):
        count = np.sum(labels == label_id)
        print(f"  {LABELS[label_id]}: {count}")

    print("\nDynamic features (should be ~N(0,1)):")
    for i, ch in enumerate(["Channel 1", "Channel 2"]):
        ch_data = dynamic[:, i, :].flatten()
        print(f"  {ch}:")
        print(f"    Mean: {ch_data.mean():.6f} (ideal: 0)")
        print(f"    Std:  {ch_data.std():.6f} (ideal: 1)")
        print(f"    Range: [{ch_data.min():.3f}, {ch_data.max():.3f}]")

    print("\nStatic features range:")
    static_names = ["Weight", "HR", "SpO2", "Height"]
    for i, name in enumerate(static_names):
        print(f"  {name}: [{static[:, i].min():.3f}, {static[:, i].max():.3f}]")

    print("\n" + "=" * 50)


def visualize_signal_comparison():
    """Compare raw signal vs self-healed signal"""
    from generate_10k import self_heal_signal, TOTAL_POINTS

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))

    np.random.seed(42)

    for i, label in enumerate([0, 2, 3]):
        t = np.linspace(0, DURATION, TOTAL_POINTS)

        hr_base = [90, 70, 65][i]
        freq = hr_base / 60
        p_amplitude = 25 + label * 2
        p_offset = 35

        noise_level = 8 - label * 1.2
        noise = np.random.normal(0, noise_level, TOTAL_POINTS)

        spike_prob = 0.001 if label >= 2 else 0.003
        spikes = np.zeros(TOTAL_POINTS)
        spike_idx = np.random.choice(TOTAL_POINTS, int(TOTAL_POINTS * spike_prob))
        spikes[spike_idx] = np.random.uniform(
            30, 60, len(spike_idx)
        ) * np.random.choice([-1, 1], len(spike_idx))

        raw_signal = (
            p_offset + p_amplitude * np.sin(2 * np.pi * freq * t) + noise + spikes
        )
        healed_signal = self_heal_signal(raw_signal)

        ax_raw = axes[i, 0]
        ax_healed = axes[i, 1]

        ax_raw.plot(t, raw_signal, linewidth=0.7, color="red", alpha=0.8)
        ax_raw.set_title(f"Raw Signal | Label: {LABELS[label]}", fontsize=10)
        ax_raw.set_ylabel("Value")
        ax_raw.grid(True, alpha=0.3)
        ax_raw.set_xlim(0, DURATION)

        ax_healed.plot(t, healed_signal, linewidth=0.7, color="green", alpha=0.8)
        ax_healed.set_title(f"Healed Signal | Label: {LABELS[label]}", fontsize=10)
        ax_healed.grid(True, alpha=0.3)
        ax_healed.set_xlim(0, DURATION)

        if i == 2:
            ax_raw.set_xlabel("Time (s)")
            ax_healed.set_xlabel("Time (s)")

    plt.suptitle("Signal Comparison: Raw vs Self-Healed", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig("experiment/generate/signal_comparison.png", dpi=150)
    print(f"[*] Saved: experiment/generate/signal_comparison.png")
    plt.close()


if __name__ == "__main__":
    check_statistics()
    visualize_samples(n_samples=4)
    visualize_by_class(n_per_class=2)
    visualize_signal_comparison()
    print("\n[*] Visualization complete!")
