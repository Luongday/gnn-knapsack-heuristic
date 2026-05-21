"""Visualization utilities for Knapsack solver comparison — v2 Publication Ready."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib + seaborn required. pip install matplotlib seaborn")

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)
plt.rcParams["font.family"] = "DejaVu Sans"  # Hoặc "Times New Roman" nếu dùng LaTeX


def mark(msg: str) -> None:
    print(f"[PLOT] {msg}", flush=True)


# ===================================================================
# 1. Solver Comparison (Bar chart)
# ===================================================================
def plot_solver_comparison(summary_path: Path, out_dir: Path) -> None:
    with summary_path.open() as f:
        summary = json.load(f)

    solver_order = ["dp", "bb", "ga", "greedy", "gnn", "dqn", "s2v", "reinforce"]
    solver_names = ["DP", "B&B", "GA", "Greedy", "GNN", "DQN", "S2V-DQN", "REINFORCE"]
    colors = ["#5F5E5A", "#888888", "#1D9E75", "#7F77DD", "#D85A30", "#378ADD", "#E8A33D", "#E74C3C"]

    data = []
    for name, display in zip(solver_order, solver_names):
        if name not in summary:
            continue
        d = summary[name]
        ratio = d.get("avg_ratio_vs_dp_feasible", 0.0)
        time_ms = d.get("avg_time_ms", 0.0)
        data.append({"Solver": display, "Ratio": ratio, "Time_ms": time_ms})

    df = pd.DataFrame(data)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Ratio
    ax = axes[0]
    sns.barplot(data=df, x="Solver", y="Ratio", palette=colors, ax=ax, edgecolor="white")
    ax.set_ylabel("Approximation Ratio (vs DP)")
    ax.set_title("Solution Quality")
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.4, label="Optimal (DP)")
    ax.legend()
    for i, v in enumerate(df["Ratio"]):
        ax.text(i, v + 0.003, f"{v:.4f}", ha="center", va="bottom", fontsize=10)

    # Time (log)
    ax = axes[1]
    sns.barplot(data=df, x="Solver", y="Time_ms", palette=colors, ax=ax, edgecolor="white")
    ax.set_ylabel("Average Time (ms)")
    ax.set_title("Solve Speed")
    ax.set_yscale("log")
    for i, v in enumerate(df["Time_ms"]):
        label = f"{v:.2f}" if v < 10 else f"{v:.0f}"
        ax.text(i, v * 1.15, label, ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_dir / "solver_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    mark(f"Saved: solver_comparison.png")


# ===================================================================
# 2. Ratio Distribution (Faceted histograms — one subplot per solver)
# ===================================================================
def plot_ratio_distribution(merged_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(merged_csv)

    solver_cols = {
        "GNN":       ("gnn_ratio",       "#D85A30"),
        "Greedy":    ("greedy_ratio",    "#7F77DD"),
        "GA":        ("ga_ratio",        "#1D9E75"),
        "BB":        ("bb_ratio",        "#5F5E5A"),
        "DQN":       ("dqn_ratio",       "#378ADD"),
        "S2V-DQN":   ("s2v_ratio",       "#E8A33D"),
        "REINFORCE": ("reinforce_ratio", "#E74C3C"),
    }

    series = []
    for name, (col, color) in solver_cols.items():
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(vals) > 0:
                series.append((name, vals, color))

    n = len(series)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
    axes = axes.flatten()

    x_global_min = min(s[1].min() for s in series)
    x_lim = (max(x_global_min - 0.01, 0.70), 1.005)

    for ax, (name, vals, color) in zip(axes, series):
        if vals.std() < 1e-9:
            ax.axvline(vals.iloc[0], color=color, linewidth=3)
            ax.set_xlim(*x_lim)
            ax.set_ylim(0, 1)
            ax.text(0.5, 0.5, f"All = {vals.iloc[0]:.4f}",
                    transform=ax.transAxes, ha="center", va="center", fontsize=12)
        else:
            ax.hist(vals, bins=30, color=color, alpha=0.85, edgecolor="white", linewidth=0.4)
            med = vals.median()
            ax.axvline(med, color="black", linestyle="--", linewidth=1.5,
                       label=f"Median = {med:.4f}")
            ax.axvline(1.0, color="red", linestyle=":", linewidth=1.2, alpha=0.7,
                       label="Optimal (1.0)")
            ax.legend(fontsize=9, loc="upper left")
            ax.set_xlim(*x_lim)

        ax.set_title(name, fontsize=13, fontweight="bold", color=color)
        ax.set_xlabel("Ratio vs DP", fontsize=10)
        ax.set_ylabel("Count", fontsize=10)
        ax.grid(axis="y", alpha=0.3)

    # Hide unused subplots
    for ax in axes[n:]:
        ax.set_visible(False)

    fig.suptitle("Approximation Ratio Distribution per Solver", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_dir / "ratio_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    mark(f"Saved: ratio_distribution.png")


# ===================================================================
# 3. Ratio by Problem Size (Line + Error bar)
# ===================================================================
def plot_ratio_by_size(merged_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(merged_csv)
    df["n_items"] = pd.to_numeric(df["n_items"], errors="coerce")

    bins = [10, 48, 86, 124, 162, 201]
    labels = ["10-47", "48-85", "86-123", "124-161", "162-200"]
    df["size_bin"] = pd.cut(df["n_items"], bins=bins, labels=labels, right=False)

    solver_info = [
        ("GNN",       "gnn_ratio",       "#D85A30"),
        ("Greedy",    "greedy_ratio",    "#7F77DD"),
        ("GA",        "ga_ratio",        "#1D9E75"),
        ("BB",        "bb_ratio",        "#888888"),
        ("DQN",       "dqn_ratio",       "#378ADD"),
        ("S2V-DQN",   "s2v_ratio",       "#E8A33D"),
        ("REINFORCE", "reinforce_ratio", "#E74C3C"),
    ]

    fig, ax = plt.subplots(figsize=(11, 6))
    for name, col, color in solver_info:
        if col not in df.columns:
            continue
        group = df.groupby("size_bin")[col].agg(["mean", "std", "count"]).reset_index()
        mean = group["mean"]
        sem = group["std"] / np.sqrt(group["count"])   # standard error
        x = np.arange(len(mean))

        ax.plot(x, mean, "o-", label=name, color=color, linewidth=2.5, markersize=7)
        ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.15)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Problem Size (n_items)")
    ax.set_ylabel("Avg Approximation Ratio")
    ax.set_title("Solution Quality by Problem Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "ratio_by_size.png", dpi=300, bbox_inches="tight")
    plt.close()
    mark(f"Saved: ratio_by_size.png")


# ===================================================================
# 4. Training Curves (gnn & RL)
# ===================================================================
def plot_training_curve(log_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(log_csv)

    if "epoch" in df.columns and "gnn_ratio" in df.columns:
        _plot_gnn_curve(df, log_csv, out_dir)
    elif "step" in df.columns and "avg_value_val" in df.columns:
        _plot_rl_curve(df, log_csv, out_dir)
    else:
        mark(f"[SKIP] Unknown training log format: {log_csv.name}")


def _plot_gnn_curve(df: pd.DataFrame, log_csv: Path, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Quality
    ax = axes[0]
    ax.plot(df["epoch"], df["gnn_ratio"], color="#D85A30", linewidth=2.5, label="gnn")
    if "greedy_ratio" in df.columns:
        ax.axhline(df["greedy_ratio"].iloc[0], color="#7F77DD", linestyle="--", label="Greedy baseline")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Approximation Ratio")
    ax.set_title("gnn Training: Quality")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[1]
    loss = df["train_loss"]
    loss = loss.clip(upper=loss.quantile(0.98))  # clip extreme spikes
    ax.plot(df["epoch"], loss, color="#D85A30", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("gnn Training: Loss")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / f"GNN_{log_csv.stem}_curve.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    mark(f"Saved: {path.name}")


def _plot_rl_curve(df: pd.DataFrame, log_csv: Path, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    parent = log_csv.parent.name.upper()
    title = f"{parent} Training"

    # Value (drop missing rows — DQN buffer warmup)
    val_df = df[["step", "avg_value_val"]].copy()
    val_df["avg_value_val"] = pd.to_numeric(val_df["avg_value_val"], errors="coerce")
    val_df = val_df.dropna(subset=["avg_value_val"])
    axes[0].plot(val_df["step"], val_df["avg_value_val"], color="#378ADD", linewidth=2)
    axes[0].set_title(f"{title}: Validation Quality")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Avg Value on Val Set")

    # Loss (drop rows where loss is missing — DQN buffer warmup)
    loss_df = df[["step", "loss"]].copy()
    loss_df["loss"] = pd.to_numeric(loss_df["loss"], errors="coerce")
    loss_df = loss_df.dropna(subset=["loss"])
    if not loss_df.empty:
        loss_clipped = loss_df["loss"].clip(upper=loss_df["loss"].quantile(0.98))
        axes[1].plot(loss_df["step"], loss_clipped, color="#D85A30")
    axes[1].set_title(f"{title}: Loss")
    axes[1].set_xlabel("Training Step")
    axes[1].set_ylabel("Training Loss")

    # Epsilon
    if "epsilon" in df.columns:
        axes[2].plot(df["step"], df["epsilon"], color="#1D9E75")
        axes[2].set_title("Epsilon-Greedy Schedule")
        axes[2].set_xlabel("Training Step")
        axes[2].set_ylabel("Epsilon")

    for ax in axes:
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = out_dir / f"{parent}_{log_csv.stem}_curve.png"
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    mark(f"Saved: {path.name}")


# ===================================================================
# 4b. Multi-variant training curves (all 4 graph structures together)
# ===================================================================

VARIANT_META = {
    'knn':            {'label': 'GNN-kNN',      'color': '#D85A30', 'marker': 'o'},
    'conflict_static':{'label': 'GNN-Conflict', 'color': '#D4537E', 'marker': '^'},
    'random':         {'label': 'GNN-Random',   'color': '#1D9E75', 'marker': 's'},
    'full':           {'label': 'GNN-Full',     'color': '#378ADD', 'marker': 'D'},
}


def plot_multi_variant_curves(log_dir: Path, out_dir: Path) -> None:
    """Plot training ratio, loss, and head-to-head for all 4 graph variants.

    Expects log_dir to contain subdirs gnn_knn/, gnn_conflict/, etc.,
    each with a training_log.csv file.
    """
    def _load_file(path: Path) -> dict | None:
        try:
            rows = list(csv.DictReader(open(path, encoding="utf-8")))
            if not rows:
                return None
            return {k: [float(r[k]) for r in rows if r.get(k) not in ("", None)] for k in rows[0].keys()}
        except (FileNotFoundError, ValueError):
            return None

    def _find_log(log_dir: Path, keyword: str) -> dict | None:
        """Find training log by keyword — tries subdir then flat file, case-insensitive glob."""
        keyword_lower = keyword.lower()
        candidates = sorted(log_dir.rglob("*.csv"))
        for p in candidates:
            name = p.name.lower()
            parent = p.parent.name.lower()
            if keyword_lower in name or keyword_lower in parent:
                if "training_log" in name or "training_log" in parent or p.name == "training_log.csv":
                    data = _load_file(p)
                    if data:
                        mark(f"  [{keyword}] loaded: {p.relative_to(log_dir)}")
                        return data
        mark(f"  [{keyword}] not found in {log_dir}")
        return None

    # Keywords per variant — ordered from most specific to least
    VARIANT_KEYWORDS = {
        'knn':            ['knn'],
        'conflict_static':['conflict_static', 'conflict'],
        'random':         ['random'],
        'full':           ['full'],
    }

    logs = {}
    for key, keywords in VARIANT_KEYWORDS.items():
        logs[key] = None
        for kw in keywords:
            result = _find_log(log_dir, kw)
            if result is not None:
                logs[key] = result
                break

    loaded = {k: v for k, v in logs.items() if v is not None}
    if not loaded:
        mark(f"No training logs found in {log_dir}. Pass correct --log_dir.")
        return

    # ---- Chart A: ratio ----
    fig, ax = plt.subplots(figsize=(11, 6))
    for key, d in loaded.items():
        m = VARIANT_META[key]
        epochs = d['epoch']
        ratios = d['gnn_ratio']
        best_i = ratios.index(max(ratios))
        ax.plot(epochs, ratios, color=m['color'], linewidth=2, label=m['label'])
        ax.scatter([epochs[best_i]], [ratios[best_i]],
                   color=m['color'], marker='*', s=120, zorder=5)
        ax.annotate(f"{ratios[best_i]:.4f}",
                    xy=(epochs[best_i], ratios[best_i]),
                    xytext=(epochs[best_i] + 1, ratios[best_i] + 0.0005),
                    fontsize=8, color=m['color'])

    first = next(iter(loaded.values()))
    if 'greedy_ratio' in first:
        g = first['greedy_ratio'][0]
        ax.axhline(g, color='#888', linestyle='--', linewidth=1.4,
                   label=f'Greedy baseline ({g:.4f})')

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Approximation Ratio (vs DP)', fontsize=11)
    ax.set_title('Training curve — Approximation Ratio (all graph structure variants)')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(axis='y', alpha=0.25)
    ax.annotate('★ best epoch per variant', xy=(0.01, 0.02),
                xycoords='axes fraction', fontsize=8, color='#555')
    plt.tight_layout()
    out = out_dir / 'training_ratio_all_variants.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    mark(f"Saved: {out.name}")

    # ---- Chart B: loss + val_acc side-by-side ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for key, d in loaded.items():
        m = VARIANT_META[key]
        axes[0].plot(d['epoch'], d['train_loss'], color=m['color'],
                     linewidth=2, label=m['label'])
        if 'val_acc' in d:
            axes[1].plot(d['epoch'], d['val_acc'], color=m['color'],
                         linewidth=2, label=m['label'])

    for ax, ylabel, title in zip(axes,
        ['Training Loss (BCE)', 'Validation Accuracy'],
        ['Training loss', 'Validation accuracy']):
        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title)
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.25)

    plt.suptitle('Training dynamics — all graph structure variants', y=1.01)
    plt.tight_layout()
    out = out_dir / 'training_loss_acc_all_variants.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    mark(f"Saved: {out.name}")

    # ---- Chart C: head-to-head wins per epoch ----
    fig, ax = plt.subplots(figsize=(11, 5))
    for key, d in loaded.items():
        m = VARIANT_META[key]
        if 'gnn_beats_greedy' in d:
            ax.plot(d['epoch'], d['gnn_beats_greedy'], color=m['color'],
                    linewidth=2, label=m['label'])

    ax.axhline(100, color='#aaa', linestyle=':', linewidth=1.2,
               label='50% threshold (100/200)')
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Instances where GNN wins (out of 200)', fontsize=11)
    ax.set_title('Head-to-head wins vs Greedy per epoch')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.25)
    plt.tight_layout()
    out = out_dir / 'training_h2h_all_variants.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    mark(f"Saved: {out.name}")


# ===================================================================
# 5. gnn vs Greedy Scatter
# ===================================================================
def plot_head_to_head(merged_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(merged_csv)
    if "gnn_ratio" not in df.columns or "greedy_ratio" not in df.columns:
        return

    x = df["greedy_ratio"].astype(float)
    y = df["gnn_ratio"].astype(float)

    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(x=x, y=y, alpha=0.6, s=25, color="#D85A30", edgecolor="white", ax=ax)
    ax.plot([0.75, 1.01], [0.75, 1.01], "--", color="black", alpha=0.5, label="Equal performance")

    gnn_wins = (y > x + 1e-6).sum()
    gr_wins = (x > y + 1e-6).sum()
    ties = len(x) - gnn_wins - gr_wins

    ax.text(0.05, 0.95, f"gnn wins: {gnn_wins}\nGreedy wins: {gr_wins}\nTies: {ties}",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7))

    ax.set_xlabel("Greedy Ratio")
    ax.set_ylabel("gnn Ratio")
    ax.set_title("gnn vs Greedy (per instance)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "gnn_vs_greedy.png", dpi=300, bbox_inches="tight")
    plt.close()
    mark(f"Saved: gnn_vs_greedy.png")


# ===================================================================
# 6. Flexible Scatter — compare any subset of solvers
# ===================================================================
SOLVER_META = {
    "dp":       {"label": "DP",        "color": "#5F5E5A"},
    "gnn":      {"label": "GNN",       "color": "#D85A30"},
    "greedy":   {"label": "Greedy",    "color": "#7F77DD"},
    "ga":       {"label": "GA",        "color": "#1D9E75"},
    "bb":       {"label": "BB",        "color": "#888888"},
    "dqn":      {"label": "DQN",       "color": "#378ADD"},
    "s2v":      {"label": "S2V-DQN",   "color": "#E8A33D"},
    "reinforce":{"label": "REINFORCE", "color": "#E74C3C"},
}


def plot_scatter(merged_csv: Path, solvers: list[str], out_dir: Path) -> None:
    """
    Compare solvers via scatter of value per instance.
    - 2 solvers  : one scatter (X = solver[0], Y = solver[1])
    - 3+ solvers : grid, each solver vs DP on X-axis
    """
    df = pd.read_csv(merged_csv)
    solvers = [s.lower() for s in solvers]

    def _get(s):
        col = "dp_value" if s == "dp" else f"{s}_value"
        if col not in df.columns:
            mark(f"[SKIP] column '{col}' not found for solver '{s}'")
            return None
        return pd.to_numeric(df[col], errors="coerce")

    if len(solvers) == 2:
        a, b = solvers
        va, vb = _get(a), _get(b)
        if va is None or vb is None:
            return
        meta_a, meta_b = SOLVER_META.get(a, {}), SOLVER_META.get(b, {})
        label_a = meta_a.get("label", a.upper())
        label_b = meta_b.get("label", b.upper())
        color_b = meta_b.get("color", "#D85A30")

        mask = va.notna() & vb.notna()
        va, vb = va[mask], vb[mask]
        lo = min(va.min(), vb.min()) * 0.98
        hi = max(va.max(), vb.max()) * 1.01

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(va, vb, color=color_b, alpha=0.55, s=22, edgecolors="white", linewidths=0.3)
        ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.2, alpha=0.5,
                label="Equal performance")

        a_wins = int((va > vb + 1e-3).sum())
        b_wins = int((vb > va + 1e-3).sum())
        ties   = int(mask.sum()) - a_wins - b_wins
        ax.text(0.04, 0.96,
                f"{label_a} better: {a_wins}\n{label_b} better: {b_wins}\nTies: {ties}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.75))

        ax.set_xlabel(f"{label_a} value", fontsize=11)
        ax.set_ylabel(f"{label_b} value", fontsize=11)
        ax.set_title(f"{label_a} vs {label_b} (per instance)", fontsize=13)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.25)
        plt.tight_layout()
        fname = f"scatter_{a}_vs_{b}.png"
        plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
        plt.close()
        mark(f"Saved: {fname}")

    else:
        # Multiple solvers: each vs DP in a grid
        dp_val = _get("dp")
        if dp_val is None:
            mark("[ERROR] DP values required for multi-solver scatter (column 'total_value')")
            return

        targets = [s for s in solvers if s != "dp"]
        ncols = min(3, len(targets))
        nrows = (len(targets) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
        axes_flat = axes.flatten()

        for ax, s in zip(axes_flat, targets):
            vs = _get(s)
            if vs is None:
                ax.set_visible(False)
                continue
            meta = SOLVER_META.get(s, {})
            label = meta.get("label", s.upper())
            color = meta.get("color", "#999999")

            mask = dp_val.notna() & vs.notna()
            x, y = dp_val[mask], vs[mask]
            lo = min(x.min(), y.min()) * 0.98
            hi = max(x.max(), y.max()) * 1.01

            ax.scatter(x, y, color=color, alpha=0.55, s=20, edgecolors="white", linewidths=0.3)
            ax.plot([lo, hi], [lo, hi], "--", color="black", linewidth=1.1, alpha=0.5)

            wins  = int((y >= x - 1e-3).sum())
            pct   = wins / mask.sum() * 100
            ax.text(0.04, 0.96, f"Optimal rate: {pct:.1f}%",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.75))
            ax.set_xlabel("DP value (optimal)", fontsize=10)
            ax.set_ylabel(f"{label} value", fontsize=10)
            ax.set_title(f"{label} vs DP", fontsize=12, fontweight="bold", color=color)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.grid(alpha=0.25)

        for ax in axes_flat[len(targets):]:
            ax.set_visible(False)

        fig.suptitle("Solver value vs DP optimal (per instance)", fontsize=14)
        plt.tight_layout()
        names = "_".join(targets)
        fname = f"scatter_{names}_vs_dp.png"
        plt.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
        plt.close()
        mark(f"Saved: {fname}")


# ===================================================================
# 7. Feasibility Rate (Bar chart)
# ===================================================================
def plot_feasibility_rate(summary_path: Path, out_dir: Path) -> None:
    with summary_path.open() as f:
        summary = json.load(f)

    solver_order = ["dp", "bb", "ga", "greedy", "gnn", "dqn", "s2v", "reinforce"]
    solver_names = ["DP", "B&B", "GA", "Greedy", "GNN", "DQN", "S2V-DQN", "REINFORCE"]
    colors       = ["#5F5E5A","#888888","#1D9E75","#7F77DD","#D85A30","#378ADD","#E8A33D","#E74C3C"]

    data = []
    for key, label, color in zip(solver_order, solver_names, colors):
        if key not in summary or not isinstance(summary[key], dict):
            continue
        feas = summary[key].get("feasible_rate")
        if feas is not None:
            data.append({"Solver": label, "Feasible%": feas * 100, "color": color})

    if not data:
        mark("[SKIP] No feasibility data in summary.json"); return

    data.sort(key=lambda d: d["Feasible%"], reverse=True)
    labels  = [d["Solver"]    for d in data]
    values  = [d["Feasible%"] for d in data]
    palette = [d["color"]     for d in data]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=palette, edgecolor="white", alpha=0.88)
    ax.axhline(100, color="#333", linestyle="--", linewidth=1.2, alpha=0.5, label="100% feasible")
    ax.set_ylim(0, 108)
    ax.set_ylabel("Feasibility Rate (%)", fontsize=11)
    ax.set_title("Feasibility Rate per Solver", fontsize=13)
    ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{val:.1f}%", ha="center", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "feasibility_rate.png", dpi=300, bbox_inches="tight")
    plt.close()
    mark("Saved: feasibility_rate.png")


# ===================================================================
# 8. Quality–Speed Pareto (Scatter)
# ===================================================================
def plot_pareto(summary_path: Path, out_dir: Path) -> None:
    with summary_path.open() as f:
        summary = json.load(f)

    solver_order = ["dp", "bb", "ga", "greedy", "gnn", "dqn", "s2v", "reinforce"]
    solver_names = ["DP", "B&B", "GA", "Greedy", "GNN", "DQN", "S2V-DQN", "REINFORCE"]
    colors       = ["#5F5E5A","#888888","#1D9E75","#7F77DD","#D85A30","#378ADD","#E8A33D","#E74C3C"]
    markers      = ["*",      "D",      "s",      "v",      "o",      "^",      "P",       "X"]

    points = []
    for key, label, color, marker in zip(solver_order, solver_names, colors, markers):
        d = summary.get(key)
        if not isinstance(d, dict): continue
        ratio = d.get("avg_ratio_vs_dp_feasible")
        time  = d.get("avg_time_ms")
        if ratio is not None and time is not None and time > 0:
            points.append({"label": label, "ratio": ratio, "time": time,
                           "color": color, "marker": marker})

    if len(points) < 2:
        mark("[SKIP] Not enough data for Pareto plot"); return

    fig, ax = plt.subplots(figsize=(9, 6))
    for p in points:
        ax.scatter(p["time"], p["ratio"], color=p["color"], marker=p["marker"],
                   s=160, zorder=4, edgecolors="white", linewidths=0.8)
        ax.annotate(p["label"], (p["time"], p["ratio"]),
                    textcoords="offset points", xytext=(8, 4),
                    fontsize=10, color=p["color"], fontweight="bold")

    # Pareto frontier (upper-left dominant)
    pts_sorted = sorted(points, key=lambda p: p["time"])
    pareto, best_ratio = [], -1
    for p in pts_sorted:
        if p["ratio"] > best_ratio:
            pareto.append(p)
            best_ratio = p["ratio"]
    if len(pareto) >= 2:
        px = [p["time"]  for p in pareto]
        py = [p["ratio"] for p in pareto]
        ax.plot(px, py, "--", color="#aaa", linewidth=1.5, zorder=2, label="Pareto frontier")

    ax.set_xscale("log")
    ax.set_xlabel("Avg Inference Time (ms, log scale)", fontsize=11)
    ax.set_ylabel("Avg Approximation Ratio (vs DP)", fontsize=11)
    ax.set_title("Quality–Speed Tradeoff (Pareto)", fontsize=13)
    ax.legend(fontsize=10); ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / "pareto_quality_speed.png", dpi=300, bbox_inches="tight")
    plt.close()
    mark("Saved: pareto_quality_speed.png")


# ===================================================================
# 9. Win / Tie / Loss Pie charts
# ===================================================================
def plot_win_pie(merged_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(merged_csv)

    # Pairs: (challenger, baseline) — show how challenger does vs baseline
    pairs = [
        ("gnn",       "greedy",  "GNN vs Greedy"),
        ("gnn",       "dp",      "GNN vs DP"),
        ("dqn",       "greedy",  "DQN vs Greedy"),
        ("s2v",       "greedy",  "S2V-DQN vs Greedy"),
        ("reinforce", "greedy",  "REINFORCE vs Greedy"),
    ]
    colors_pie = ["#2ecc71", "#bdc3c7", "#e74c3c"]   # win / tie / loss

    valid_pairs = []
    for a, b, title in pairs:
        col_a = "dp_value" if a == "dp" else f"{a}_value"
        col_b = "dp_value" if b == "dp" else f"{b}_value"
        if col_a in df.columns and col_b in df.columns:
            valid_pairs.append((a, b, title, col_a, col_b))

    if not valid_pairs:
        mark("[SKIP] No matching columns for win/pie chart"); return

    ncols = min(3, len(valid_pairs))
    nrows = (len(valid_pairs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for ax, (a, b, title, col_a, col_b) in zip(axes, valid_pairs):
        va = pd.to_numeric(df[col_a], errors="coerce")
        vb = pd.to_numeric(df[col_b], errors="coerce")
        mask = va.notna() & vb.notna()
        va, vb = va[mask], vb[mask]
        n = mask.sum()

        wins  = int((va > vb + 1e-3).sum())
        losses= int((vb > va + 1e-3).sum())
        ties  = n - wins - losses

        sizes  = [wins, ties, losses]
        labels = [f"Win\n{wins}", f"Tie\n{ties}", f"Loss\n{losses}"]
        # Hide slice if 0
        non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors_pie) if s > 0]
        if not non_zero:
            ax.set_visible(False); continue

        wedges, texts, autotexts = ax.pie(
            [s for s, _, _ in non_zero],
            labels=[l for _, l, _ in non_zero],
            colors=[c for _, _, c in non_zero],
            autopct="%1.1f%%", startangle=90,
            wedgeprops=dict(edgecolor="white", linewidth=1.5),
        )
        for t in autotexts:
            t.set_fontsize(10)
        ax.set_title(f"{title}\n(n={n})", fontsize=11, fontweight="bold")

    for ax in axes[len(valid_pairs):]:
        ax.set_visible(False)

    fig.suptitle("Win / Tie / Loss Distribution (per instance)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(out_dir / "win_tie_loss_pie.png", dpi=300, bbox_inches="tight")
    plt.close()
    mark("Saved: win_tie_loss_pie.png")


# ===================================================================
# 10. Print summary stats from summary.json
# ===================================================================
def print_summary(summary_path: Path) -> None:
    with summary_path.open() as f:
        data = json.load(f)

    order  = ["dp", "bb", "ga", "greedy", "gnn", "dqn", "s2v", "reinforce"]
    labels = {"dp": "DP", "bb": "BB", "ga": "GA", "greedy": "Greedy",
              "gnn": "GNN", "dqn": "DQN", "s2v": "S2V-DQN", "reinforce": "REINFORCE"}

    col_w = 10
    cols  = ["Solver", "N", "Feasible%", "Avg Ratio", "Std Ratio", "Avg Time(ms)"]
    header = f"{'Solver':<10} {'N':>6} {'Feasible%':>10} {'Avg Ratio':>10} {'Std Ratio':>10} {'Avg Time(ms)':>13}"
    sep    = "-" * len(header)

    print("\n" + "=" * len(header))
    print("SOLVER SUMMARY")
    print("=" * len(header))
    print(header)
    print(sep)

    active = [s for s in order if s in data] + \
             [s for s in data if s not in order and s != "n_common_instances"]

    for s in active:
        d = data.get(s)
        if not isinstance(d, dict):
            continue
        label     = labels.get(s, s.upper())
        n         = d.get("count", 0)
        feas_rate = f"{d.get('feasible_rate', 0)*100:.1f}%"
        avg_ratio = d.get("avg_ratio_vs_dp_feasible")
        std_ratio = d.get("std_ratio_vs_dp")
        avg_time  = d.get("avg_time_ms")

        ratio_str = f"{avg_ratio:.4f}" if avg_ratio is not None else "N/A"
        std_str   = f"{std_ratio:.4f}" if std_ratio is not None else "N/A"
        time_str  = f"{avg_time:.2f}"  if avg_time  is not None else "N/A"

        print(f"{label:<10} {n:>6} {feas_rate:>10} {ratio_str:>10} {std_str:>10} {time_str:>13}")

    n_inst = data.get("n_common_instances", "?")
    print(sep)
    print(f"Instances: {n_inst}")
    print("=" * len(header) + "\n")


# ===================================================================
# Main
# ===================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Plot Knapsack solver results — Publication Ready v2")
    parser.add_argument("--results_dir",  type=Path, default=None,
                        help="Directory with summary.json + merged_results.csv")
    parser.add_argument("--scatter",      type=str,  nargs="+", action="append", default=None,
                        metavar="SOLVER",
                        help="Flexible scatter: e.g. --scatter gnn dp  OR  --scatter gnn greedy dqn")
    parser.add_argument("--show_summary", action="store_true",
                        help="Print summary stats from summary.json")
    parser.add_argument("--training_log",  type=Path, default=None)
    parser.add_argument("--training_logs", type=Path, nargs="*", default=None)
    parser.add_argument("--log_dir",       type=Path, default=None,
                        help="Directory with gnn_knn/, gnn_conflict/, etc. subdirs "
                             "containing training_log.csv.")
    parser.add_argument("--out_dir",       type=Path, default=Path("plots"))
    return parser.parse_args()


def main():
    if not HAS_MPL:
        sys.exit(1)

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    merged  = None
    summary = None

    if args.results_dir:
        summary = args.results_dir / "summary.json"
        merged  = args.results_dir / "merged_results.csv"
        if summary.exists():
            plot_solver_comparison(summary, args.out_dir)
            plot_feasibility_rate(summary, args.out_dir)
            plot_pareto(summary, args.out_dir)
        if merged.exists():
            plot_ratio_distribution(merged, args.out_dir)
            plot_ratio_by_size(merged, args.out_dir)
            plot_head_to_head(merged, args.out_dir)
            plot_win_pie(merged, args.out_dir)

    if args.show_summary:
        path = summary or Path("results/compare/summary.json")
        if path.exists():
            print_summary(path)
        else:
            mark(f"[ERROR] summary.json not found at {path}")

    if args.scatter:
        path = merged or Path("results/compare/merged_results.csv")
        if path.exists():
            for solver_group in args.scatter:
                plot_scatter(path, solver_group, args.out_dir)
        else:
            mark(f"[ERROR] merged_results.csv not found at {path}")

    if args.log_dir:
        plot_multi_variant_curves(args.log_dir, args.out_dir)
    if args.training_log:
        plot_training_curve(args.training_log, args.out_dir)
    if args.training_logs:
        for p in args.training_logs:
            plot_training_curve(Path(p), args.out_dir)

    mark(f"All plots saved to -> {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()