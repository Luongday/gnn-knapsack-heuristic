"""Plot charts for the report: ablation, cross-scale, and training curves."""
import argparse
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from statistics import mean, stdev

sns.set_style("whitegrid", {"grid.linewidth": 0.8, "axes.edgecolor": ".25"})
sns.set_context("notebook", font_scale=1.05)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


def load_log(log_csv: Path) -> dict:
    """Load a training log CSV into column-keyed lists. Returns None if missing."""
    try:
        rows = list(csv.DictReader(open(log_csv)))
        if not rows:
            return None
        keys = rows[0].keys()
        return {k: [float(r[k]) for r in rows] for k in keys}
    except FileNotFoundError:
        return None


def ratio_from_csv(csv_path):
    """Mean of pre-computed 'ratio' column (total_value / dp_value) in solver CSV."""
    try:
        vals = [float(r['ratio']) for r in csv.DictReader(open(csv_path))
                if r.get('ratio') not in ('', None)]
        return mean(vals) if vals else None
    except FileNotFoundError:
        return None


def ratio_from_csv_stats(csv_path):
    """Return (mean, std, n) of 'ratio' column. Returns (None, None, 0) if file missing."""
    try:
        vals = [float(r['ratio']) for r in csv.DictReader(open(csv_path))
                if r.get('ratio') not in ('', None)]
        if not vals:
            return None, None, 0
        return mean(vals), (stdev(vals) if len(vals) > 1 else 0.0), len(vals)
    except FileNotFoundError:
        return None, None, 0


def ratio_vs_dp(gnn_csv, dp_csv):
    """Mean ratio = gnn_value / dp_value, joined by instance_file."""
    try:
        gnn_rows = {r['instance_file']: float(r['total_value'])
                    for r in csv.DictReader(open(gnn_csv))
                    if r.get('total_value') and r.get('feasible', '1') != '0'}
        dp_rows  = {r['instance_file']: float(r['total_value'])
                    for r in csv.DictReader(open(dp_csv))
                    if r.get('total_value')}
        ratios = [gnn_rows[k] / dp_rows[k] for k in gnn_rows if k in dp_rows and dp_rows[k] > 0]
        return mean(ratios) if ratios else None
    except FileNotFoundError:
        return None


def ratio_vs_dp_stats(gnn_csv, dp_csv):
    """Return (mean, std, n) of ratio = gnn_value / dp_value."""
    try:
        gnn_rows = {r['instance_file']: float(r['total_value'])
                    for r in csv.DictReader(open(gnn_csv))
                    if r.get('total_value') and r.get('feasible', '1') != '0'}
        dp_rows  = {r['instance_file']: float(r['total_value'])
                    for r in csv.DictReader(open(dp_csv))
                    if r.get('total_value')}
        ratios = [gnn_rows[k] / dp_rows[k] for k in gnn_rows if k in dp_rows and dp_rows[k] > 0]
        if not ratios:
            return None, None, 0
        return mean(ratios), (stdev(ratios) if len(ratios) > 1 else 0.0), len(ratios)
    except FileNotFoundError:
        return None, None, 0


def _sem(stats_tuple):
    """Standard error of the mean from a (mean, std, n) tuple."""
    _, s, n = stats_tuple
    if s is None or n < 2:
        return 0.0
    return s / (n ** 0.5)


def avg_time(csv_path):
    try:
        rows = list(csv.DictReader(open(csv_path)))
        if not rows:
            return None
        col = next((c for c in ['inference_time_ms', 'time_ms'] if c in rows[0]), None)
        if col is None: return None
        return mean([float(r[col]) for r in rows if r[col]])
    except (FileNotFoundError, StopIteration, KeyError):
        return None


# ---------------------------------------------------------------------------
# Smoothing helpers
# ---------------------------------------------------------------------------

VARIANT_META = {
    'knn':            {'label': 'GNN-kNN',     'color': '#D85A30', 'ls': '-',  'marker': 'o'},
    'conflict_static':{'label': 'GNN-Conflict','color': '#9B59B6', 'ls': '--', 'marker': '^'},
    'random':         {'label': 'GNN-Random',  'color': '#1D9E75', 'ls': '-.',  'marker': 's'},
    'full':           {'label': 'GNN-Full',    'color': '#378ADD', 'ls': ':',  'marker': 'D'},
}


def _smooth(vals: list, window: int = 7) -> np.ndarray:
    """Centered rolling mean; edges padded with edge values."""
    arr = np.array(vals, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, window // 2, mode='edge')
    return np.convolve(padded, kernel, mode='valid')[:len(arr)]


def _rolling_std(vals: list, window: int = 7) -> np.ndarray:
    """Rolling std for shaded bands; edges padded with edge values."""
    arr = np.array(vals, dtype=float)
    out = np.empty_like(arr)
    half = window // 2
    for i in range(len(arr)):
        lo = max(0, i - half)
        hi = min(len(arr), i + half + 1)
        out[i] = arr[lo:hi].std()
    return out


# ---------------------------------------------------------------------------
# Chart 3: Training ratio curves
# ---------------------------------------------------------------------------

def plot_training_ratio(logs: dict, out_dir: Path) -> None:
    """Chart 3: two-panel — full curve + zoomed convergence region with shaded std band."""
    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 6.5))

    first_d = next((v for v in logs.values() if v is not None), None)
    greedy_val = first_d['greedy_ratio'][0] if (first_d and 'greedy_ratio' in first_d) else None

    variant_data = {}
    for key, meta in VARIANT_META.items():
        d = logs.get(key)
        if d is None:
            continue
        epochs = np.array(d['epoch'])
        ratios = np.array(d['gnn_ratio'])
        variant_data[key] = {
            'epochs': epochs,
            'raw':    ratios,
            'smooth': _smooth(ratios.tolist()),
            'std':    _rolling_std(ratios.tolist()),
            'meta':   meta,
        }

    for ax, title, epoch_min, show_raw in [
        (ax_full, '(a) Full training curve',      1,  True),
        (ax_zoom, '(b) Convergence — epoch 10+', 10, False),
    ]:
        for key, vd in variant_data.items():
            meta = vd['meta']
            mask = vd['epochs'] >= epoch_min
            ep, sm, std = vd['epochs'][mask], vd['smooth'][mask], vd['std'][mask]

            if show_raw:
                ax.plot(vd['epochs'][mask], vd['raw'][mask],
                        color=meta['color'], linewidth=0.7, alpha=0.15,
                        linestyle=meta['ls'])
            ax.fill_between(ep, np.clip(sm - std, 0, 1), np.clip(sm + std, 1.01, None),
                            color=meta['color'], alpha=0.12)
            ax.plot(ep, sm, color=meta['color'], linewidth=2.2,
                    linestyle=meta['ls'], label=meta['label'])

        if greedy_val is not None:
            ax.axhline(greedy_val, color='#555', linestyle='--', linewidth=1.4,
                       label=f'Greedy baseline ({greedy_val:.4f})')

        ax.set_xlabel('Epoch', fontsize=11)
        ax.set_ylabel('Approximation Ratio (vs DP)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

    zoom_smoothed = []
    for key, vd in variant_data.items():
        mask = vd['epochs'] >= 10
        zoom_smoothed.extend(vd['smooth'][mask].tolist())

    if zoom_smoothed:
        ymin = max(min(zoom_smoothed) - 0.001, 0.975)
        ymax = min(max(zoom_smoothed) + 0.003, 1.001)
        ax_zoom.set_ylim(ymin, ymax)

    best_list = []
    for key, vd in variant_data.items():
        mask = vd['epochs'] >= 10
        ep_m, sm_m = vd['epochs'][mask], vd['smooth'][mask]
        bi = int(np.argmax(sm_m))
        best_list.append((ep_m[bi], sm_m[bi], vd['meta']['color'],
                          vd['meta']['label'], sm_m[bi]))

    best_list.sort(key=lambda x: x[4], reverse=True)
    for i, (bep, bval, bcol, blbl, _) in enumerate(best_list):
        ax_zoom.scatter([bep], [bval], color=bcol, marker='*', s=150, zorder=6)
        ax_zoom.annotate(f'{blbl}: {bval:.4f}',
                         xy=(bep, bval), xycoords='data',
                         xytext=(0.97, 0.95 - i * 0.10),
                         textcoords='axes fraction',
                         fontsize=8, color=bcol, ha='right',
                         arrowprops=dict(arrowstyle='->', color=bcol,
                                         lw=0.8, connectionstyle='arc3,rad=0.15'))

    fig.text(0.5, -0.02,
             'Shaded band = rolling std (window=7)  |  Bold lines = smoothed mean  |  Faint lines = raw data',
             fontsize=8, color='#888', ha='center')

    plt.suptitle('Training curve — Approximation Ratio per graph structure variant',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = out_dir / 'chart3_training_ratio.png'
    plt.savefig(out, dpi=130, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# Chart 4: Training loss + validation accuracy
# ---------------------------------------------------------------------------

def plot_training_loss(logs: dict, out_dir: Path) -> None:
    """Chart 4: loss + val_acc with smoothing, shaded std band, and random baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: training loss
    ax = axes[0]
    for key, meta in VARIANT_META.items():
        d = logs.get(key)
        if d is None:
            continue
        epochs   = np.array(d['epoch'])
        loss     = np.array(d['train_loss'])
        smoothed = _smooth(loss.tolist())
        std      = _rolling_std(loss.tolist())
        ax.fill_between(epochs, smoothed - std, smoothed + std,
                        color=meta['color'], alpha=0.12)
        ax.plot(epochs, loss, color=meta['color'], linewidth=0.7, alpha=0.15,
                linestyle=meta['ls'])
        ax.plot(epochs, smoothed, color=meta['color'], linewidth=2.2,
                linestyle=meta['ls'], label=meta['label'])
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Loss (BCE)', fontsize=11)
    ax.set_title('(a) Training loss', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Right: validation accuracy — envelope across variants shows inter-variant spread
    ax = axes[1]
    epoch_to_acc = {}   # epoch -> list of smoothed acc values (one per variant)
    acc_series = []     # (meta, epochs, raw_acc, smoothed)
    final_vals = []

    for key, meta in VARIANT_META.items():
        d = logs.get(key)
        if d is None or 'val_acc' not in d:
            continue
        epochs   = np.array(d['epoch'])
        acc      = np.array(d['val_acc'])
        smoothed = _smooth(acc.tolist())
        acc_series.append((meta, epochs, acc, smoothed))
        for i, e in enumerate(epochs):
            epoch_to_acc.setdefault(int(e), []).append(smoothed[i])
        final_vals.append((meta['label'], smoothed[-1], meta['color']))

    # Draw envelope band (spread across all variants at each epoch)
    if epoch_to_acc:
        ep_keys = sorted(epoch_to_acc.keys())
        env_lo  = np.array([min(epoch_to_acc[e]) for e in ep_keys])
        env_hi  = np.array([max(epoch_to_acc[e]) for e in ep_keys])
        ax.fill_between(ep_keys, env_lo, env_hi,
                        color='#999', alpha=0.18, label='Variant spread (min–max)')

    # Draw individual smoothed lines
    all_smoothed_vals = []
    for meta, epochs, acc, smoothed in acc_series:
        ax.plot(epochs, acc, color=meta['color'], linewidth=0.7, alpha=0.15,
                linestyle=meta['ls'])
        ax.plot(epochs, smoothed, color=meta['color'], linewidth=2.2,
                linestyle=meta['ls'], label=meta['label'])
        all_smoothed_vals.extend(smoothed[10:].tolist())

    # Random baseline for binary node classification
    ax.axhline(0.5, color='#bbb', linestyle=':', linewidth=1.2,
               label='Random baseline (0.50)')

    # Tight y-axis zoom around actual data range (exclude baseline from ylim calc)
    if all_smoothed_vals:
        ymin = max(min(all_smoothed_vals) - 0.005, 0.0)
        ymax = min(max(all_smoothed_vals) + 0.005, 1.00)
        ax.set_ylim(ymin, ymax)
        # Annotate baseline as off-chart arrow if 0.5 is below visible range
        if ymin > 0.5:
            ax.annotate('Random baseline (0.50) ↓ off-chart',
                        xy=(0.02, 0.02), xycoords='axes fraction',
                        fontsize=7, color='#aaa')

    if final_vals:
        lines = [f"{lbl}: {val:.4f}" for lbl, val, _ in sorted(final_vals, key=lambda x: -x[1])]
        ax.text(0.97, 0.05, "Final (smoothed)\n" + "\n".join(lines),
                transform=ax.transAxes, fontsize=8, va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5',
                          edgecolor='#ccc', alpha=0.9))

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Validation Accuracy', fontsize=11)
    ax.set_title('(b) Validation accuracy (zoomed)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(axis='y', alpha=0.3)

    fig.text(0.5, -0.02,
             'Shaded band = rolling std (window=7)  |  Faint lines = raw data',
             fontsize=8, color='#888', ha='center')

    plt.suptitle('Training dynamics — all graph structure variants', y=1.02)
    plt.tight_layout()
    out = out_dir / 'chart4_training_loss_acc.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# Chart 5: Head-to-head win rate + advantage
# ---------------------------------------------------------------------------

def plot_training_h2h(logs: dict, out_dir: Path) -> None:
    """Chart 5: stacked Win/Tie/Loss area (representative variant) + advantage lines (all variants)."""
    fig, (ax_stack, ax_adv) = plt.subplots(1, 2, figsize=(16, 5))

    # ── Panel left: stacked area Win / Tie / Loss = 100% (representative variant) ──
    rep_key = next((k for k in ['knn', 'full', 'random', 'conflict_static']
                    if logs.get(k) is not None), None)

    if rep_key is not None:
        d    = logs[rep_key]
        meta = VARIANT_META[rep_key]
        epochs = np.array(d['epoch'])
        wins   = np.array(d['gnn_beats_greedy'])
        losses = np.array(d.get('gnn_loses_greedy', np.zeros(len(epochs))))
        ties   = np.array(d.get('ties',             np.zeros(len(epochs))))
        total  = wins + losses + ties

        denom = np.where(total > 0, total, 1)
        win_pct  = _smooth((wins   / denom * 100).tolist(), window=9)
        tie_pct  = _smooth((ties   / denom * 100).tolist(), window=9)
        loss_pct = _smooth((losses / denom * 100).tolist(), window=9)

        # Normalise smoothed values so each column still sums to exactly 100
        row_sum  = win_pct + tie_pct + loss_pct
        win_pct  = win_pct  / row_sum * 100
        tie_pct  = tie_pct  / row_sum * 100
        loss_pct = loss_pct / row_sum * 100

        ax_stack.stackplot(epochs,
                           win_pct, tie_pct, loss_pct,
                           labels=['Win  (GNN > Greedy)',
                                   'Tie  (GNN = Greedy)',
                                   'Loss (GNN < Greedy)'],
                           colors=['#2ECC71', '#95A5A6', '#E74C3C'],
                           alpha=0.88)

        ax_stack.axhline(50, color='#222', linestyle='--',
                         linewidth=1.3, label='50% line')

        # In-band percentage labels at the final epoch
        y_refs = [win_pct[-1] / 2,
                  win_pct[-1] + tie_pct[-1] / 2,
                  win_pct[-1] + tie_pct[-1] + loss_pct[-1] / 2]
        pcts   = [win_pct[-1], tie_pct[-1], loss_pct[-1]]
        txts   = [f'Win\n{win_pct[-1]:.0f}%',
                  f'Tie\n{tie_pct[-1]:.0f}%',
                  f'Loss\n{loss_pct[-1]:.0f}%']
        for y_mid, pct, txt in zip(y_refs, pcts, txts):
            if pct >= 6:   # only label if band is wide enough to read
                ax_stack.text(epochs[-1] * 0.97, y_mid, txt,
                              fontsize=9, color='white', fontweight='bold',
                              ha='right', va='center')

        ax_stack.set_xlim(epochs[0], epochs[-1])
        ax_stack.set_ylim(0, 100)
        ax_stack.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
        ax_stack.set_xlabel('Epoch', fontsize=11)
        ax_stack.set_ylabel('% of test instances  (Win + Tie + Loss = 100%)', fontsize=11)
        ax_stack.set_title(f'(a) Win / Tie / Loss breakdown  [{meta["label"]}]',
                           fontsize=12, fontweight='bold')
        ax_stack.legend(loc='center right', fontsize=9, framealpha=0.85)
        ax_stack.grid(axis='y', alpha=0.2)

    # ── Panel right: quality advantage (gnn_ratio − greedy_ratio), all variants ──
    all_adv_smoothed = []
    for key, meta in VARIANT_META.items():
        d = logs.get(key)
        if d is None or 'advantage' not in d:
            continue
        epochs = np.array(d['epoch'])
        adv    = np.array(d['advantage'])
        sm     = _smooth(adv.tolist(), window=9)
        std    = _rolling_std(adv.tolist(), window=9)

        ax_adv.fill_between(epochs, sm - std, sm + std,
                            color=meta['color'], alpha=0.15)
        ax_adv.plot(epochs, sm, color=meta['color'], linewidth=2.4,
                    linestyle=meta['ls'], label=meta['label'])
        all_adv_smoothed.extend(sm.tolist())

        ax_adv.annotate(f'{sm[-1]:+.4f}',
                        xy=(epochs[-1], sm[-1]),
                        xytext=(4, 0), textcoords='offset points',
                        fontsize=8, color=meta['color'], va='center',
                        fontweight='bold')

    ax_adv.axhline(0, color='#333', linestyle='--', linewidth=1.4,
                   label='Greedy parity (0)')

    if all_adv_smoothed:
        lo = min(all_adv_smoothed)
        hi = max(all_adv_smoothed)
        margin = (hi - lo) * 0.15 + 0.001
        ax_adv.set_ylim(lo - margin, hi + margin)
        if hi > 0:
            ax_adv.fill_between(ax_adv.get_xlim(), 0, hi + margin,
                                color='#2ECC71', alpha=0.06, label='_nolegend_')
            ax_adv.text(0.98, 0.97, 'Above 0 → GNN ≥ Greedy ratio',
                        transform=ax_adv.transAxes, ha='right', va='top',
                        fontsize=8, color='#27AE60', style='italic')

    ax_adv.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f'{x:+.4f}'))
    ax_adv.set_xlabel('Epoch', fontsize=11)
    ax_adv.set_ylabel('GNN ratio − Greedy ratio', fontsize=11)
    ax_adv.set_title('(b) Quality advantage over Greedy — all variants',
                     fontsize=12, fontweight='bold')
    ax_adv.legend(loc='lower right', fontsize=9)
    ax_adv.grid(axis='y', alpha=0.25)

    fig.text(0.5, -0.03,
             'Panel (a): smoothed (window=9), Win+Tie+Loss = 100% per epoch  |  '
             'Panel (b): shaded band = rolling std',
             fontsize=8, color='#666', ha='center')
    plt.suptitle('Head-to-head vs Greedy baseline — training dynamics',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    out = out_dir / 'chart5_training_h2h.png'
    plt.savefig(out, dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out}')


# ---------------------------------------------------------------------------
# Main: chart 1, 1b, 2 (static result charts) + charts 3-5 (training curves)
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', '--output_dir', type=Path,
                        default=Path('results/cross_scale'))
    parser.add_argument('--log_dir', type=Path, default=None,
                        help='Directory with training log CSVs '
                             '(expects gnn_knn_training_log.csv, etc.). '
                             'Defaults to results/GNN if omitted.')
    args = parser.parse_args()
    BASE = args.results_dir
    Path('plots').mkdir(exist_ok=True)

    # ===================================================================
    # CHART 1: Ablation — quality with error bars + baselines
    # ===================================================================
    graph_labels = ['Full', 'kNN', 'Random', 'Conflict']
    graph_keys   = ['full', 'knn', 'random', 'conflict_static']
    colors       = ['#2C5F9E', '#D85A30', '#888888', '#1D9E75']

    dp_base_small = BASE / 'dp_small.csv'

    # Mean + std for each bar group
    greedy_stats = [ratio_vs_dp_stats(BASE / f'{g}_on_small_greedy.csv', dp_base_small)
                    for g in graph_keys]
    dp_stats     = [ratio_vs_dp_stats(BASE / f'{g}_on_small_dp.csv',     dp_base_small)
                    for g in graph_keys]
    greedy_r = [s[0] for s in greedy_stats]
    greedy_e = [_sem(s) for s in greedy_stats]   # SEM, not raw std
    dp_r     = [s[0] for s in dp_stats]
    dp_e     = [_sem(s) for s in dp_stats]

    greedy_base = ratio_vs_dp(BASE / 'greedy_small.csv', dp_base_small)
    # Load random-solver baseline if available; skip silently otherwise
    random_base = ratio_vs_dp(BASE / 'random_small.csv', dp_base_small)

    x = np.arange(len(graph_labels))
    w = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x - w / 2, [v or 0 for v in greedy_r], w,
                yerr=[e or 0 for e in greedy_e], capsize=4,
                error_kw=dict(elinewidth=1.2, alpha=0.8),
                label='Greedy decode', color='#4A90E2', alpha=0.85)
    b2 = ax.bar(x + w / 2, [v or 0 for v in dp_r], w,
                yerr=[e or 0 for e in dp_e], capsize=4,
                error_kw=dict(elinewidth=1.2, alpha=0.8),
                label='DP subset decode', color='#D85A30', alpha=0.85)

    ax.axhline(1.0, color='#222', linestyle='-', linewidth=1.5,
               label='DP optimal (1.0)')
    if greedy_base:
        ax.axhline(greedy_base, color='gray', linestyle='--', linewidth=1.5,
                   label=f'Greedy heuristic ({greedy_base:.4f})')
    if random_base is not None:
        ax.axhline(random_base, color='#E74C3C', linestyle=':', linewidth=1.5,
                   label=f'Random selection ({random_base:.4f})')

    ax.set_xticks(x)
    ax.set_xticklabels(graph_labels, fontsize=11)
    ax.set_ylabel('Approximation Ratio (vs DP optimal)', fontsize=11)
    ax.set_title('Ablation: Graph Structure × Decode Method — Quality (n≤50)')
    valid = [v for v in greedy_r + dp_r if v]
    ax.set_ylim(max(0.97, min(valid) - 0.005) if valid else 0.97, 1.003)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.5, linewidth=0.7)
    for bar, val in zip(list(b1) + list(b2), greedy_r + dp_r):
        if val:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                    f'{val:.4f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('plots/chart1_ablation.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: plots/chart1_ablation.png")

    # ===================================================================
    # CHART 1b: Speed — direct endpoint labels, independent y-axes per panel
    # ===================================================================
    test_labels_sp = ['n≤50', 'n=100', 'n=200']   # clean labels; no "(train)" in tick
    test_keys_sp   = ['small', 'n100', 'n200']
    greedy_base_ts = [avg_time(BASE / f'greedy_{t}.csv') for t in test_keys_sp]

    has_time_data = any(
        avg_time(BASE / f'{g}_on_{t}_greedy.csv')
        for g in graph_keys for t in test_keys_sp
    )
    if has_time_data:
        # sharey=False: panels have independent y-axes so greedy (~0.04 ms) stays
        # visible in panel (a) without being crushed by panel (b)'s DP scale.
        fig2, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
        markers = ['s', 'o', 'D', '^']

        for ax2, suffix, title in [
            (axL, 'greedy', '(a) Greedy Decode'),
            (axR, 'dp',     '(b) DP Subset Decode'),
        ]:
            all_panel_vals = []
            for g, lbl, c, m in zip(graph_keys, graph_labels, colors, markers):
                ys = [avg_time(BASE / f'{g}_on_{t}_{suffix}.csv') or 0
                      for t in test_keys_sp]
                ax2.plot(test_labels_sp, ys, marker=m, color=c,
                         linewidth=2, markersize=9, linestyle='-',
                         label=f'GNN-{lbl}')
                all_panel_vals.extend(y for y in ys if y > 0)
                # Direct label at the last (rightmost) data point
                last_y = ys[-1]
                if last_y > 0:
                    ax2.annotate(f'{last_y:.1f} ms',
                                 xy=(test_labels_sp[-1], last_y),
                                 xytext=(6, 0), textcoords='offset points',
                                 fontsize=8, color=c, va='center')

            valid_gh = [(tl, t) for tl, t in zip(test_labels_sp, greedy_base_ts) if t]
            if valid_gh:
                gt_times = [t for _, t in valid_gh]
                ax2.plot([tl for tl, _ in valid_gh], gt_times,
                         marker='x', color='gray', linewidth=1.5, linestyle=':',
                         markersize=9, label=f'Greedy heuristic (~{gt_times[0]:.2f} ms)')
                all_panel_vals.extend(gt_times)

            ax2.set_yscale('log')
            # Explicit ylim so greedy line is always in frame
            if all_panel_vals:
                ax2.set_ylim(min(all_panel_vals) * 0.4, max(all_panel_vals) * 3)
            ax2.set_xlabel('Test set (n = number of items; trained on n≤50)', fontsize=11)
            ax2.set_title(title, fontsize=11)
            ax2.legend(fontsize=9, loc='upper left')
            ax2.grid(axis='y', alpha=0.3)

        axL.set_ylabel('Avg inference time (ms, log scale)', fontsize=11)
        plt.suptitle('Speed: Graph Structure × Decode Method across Problem Sizes',
                     fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig('plots/chart1b_speed.png', dpi=180, bbox_inches='tight')
        plt.close()
        print("Saved: plots/chart1b_speed.png")

    # ===================================================================
    # CHART 2: Cross-scale — error bars + synchronised y-axis
    # ===================================================================
    tests     = ['n≤50', 'n=100', 'n=200']   # no "(train)" in tick label
    test_keys = ['small', 'n100', 'n200']

    S2V_GRAPH_KEYS = ['knn', 'conflict', 'random', 'full']
    RL_SINGLE = {
        'dqn':       {'label': 'DQN',       'color': '#F5A623', 'marker': 'p', 'ls': '-.'},
        'reinforce': {'label': 'REINFORCE', 'color': '#E74C3C', 'marker': '*', 'ls': '-.'},
    }

    def best_gnn(t):
        pairs = [(g, ratio_vs_dp(BASE / f'{g}_on_{t}_greedy.csv', BASE / f'dp_{t}.csv'))
                 for g in graph_keys]
        valid = [(g, v) for g, v in pairs if v is not None]
        if not valid: return None, None
        g, v = max(valid, key=lambda x: x[1])
        return v, g

    def best_s2v(t):
        pairs = [(gt, ratio_from_csv(BASE / f's2v_{gt}_on_{t}.csv')) for gt in S2V_GRAPH_KEYS]
        valid = [(gt, v) for gt, v in pairs if v is not None]
        if not valid: return None, None
        gt, v = max(valid, key=lambda x: x[1])
        return v, gt

    REINFORCE_VARIANTS = ['hard', 'none', 'polyak']
    def best_reinforce(t):
        pairs = [(rv, ratio_from_csv(BASE / f'reinforce_{rv}_on_{t}.csv'))
                 for rv in REINFORCE_VARIANTS]
        valid = [(rv, v) for rv, v in pairs if v is not None]
        if not valid: return None, None
        rv, v = max(valid, key=lambda x: x[1])
        return v, rv

    greedy_ys = [ratio_vs_dp(BASE / f'greedy_{t}.csv', BASE / f'dp_{t}.csv')
                 for t in test_keys]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)
    all_y_left  = []
    all_y_right = []

    # Panel left: GNN ablation — 4 graph structure variants with SEM error bars
    gnn_markers = {'knn': 'o', 'full': 's', 'conflict_static': '^', 'random': 'D'}
    for g, label, c in zip(graph_keys, graph_labels, colors):
        stats = [ratio_vs_dp_stats(BASE / f'{g}_on_{t}_greedy.csv', BASE / f'dp_{t}.csv')
                 for t in test_keys]
        ys      = [s[0] or 0 for s in stats]
        yes_err = [_sem(s) for s in stats]
        if any(y > 0 for y in ys):
            ax1.errorbar(tests, ys, yerr=yes_err,
                         marker=gnn_markers[g], label=f'GNN-{label}', color=c,
                         linewidth=2, markersize=9, capsize=4,
                         elinewidth=1.2, alpha=0.9)
            all_y_left.extend(y for y in ys if y > 0)

    if any(y is not None for y in greedy_ys):
        ys_g = [y or 0 for y in greedy_ys]
        ax1.plot(tests, ys_g, marker='x', linestyle='--',
                 label='Greedy heuristic', color='#7F77DD',
                 linewidth=1.5, markersize=8)
        all_y_left.extend(y for y in ys_g if y > 0)

    ax1.set_xlabel('Test set (n = number of items; trained on n≤50)', fontsize=11)
    ax1.set_ylabel('Approximation Ratio (vs DP)', fontsize=11)
    ax1.set_title('(a) GNN — Graph Structure Ablation', fontsize=11)
    ax1.legend(loc='lower left', fontsize=9)
    ax1.grid(axis='y', alpha=0.3)

    # Panel right: one line per METHOD FAMILY (not per variant) to avoid overlap with panel (a)
    gnn_results = [best_gnn(t) for t in test_keys]
    gnn_best_ys = [r[0] for r in gnn_results]
    gnn_best_var = [r[1] for r in gnn_results]
    if any(y is not None for y in gnn_best_ys):
        ys_b = [y or 0 for y in gnn_best_ys]
        picks = " / ".join(f"{t}={v.replace('conflict_static','conflict')}"
                           for t, v in zip(tests, gnn_best_var) if v)
        ax2.plot(tests, ys_b, marker='*', linestyle='-',
                 label=f'GNN (best: {picks})', color='#D85A30',
                 linewidth=2.5, markersize=11)
        all_y_right.extend(y for y in ys_b if y > 0)

    # S2V-DQN: single best-across-variants line — with picked variant disclosed
    s2v_results = [best_s2v(t) for t in test_keys]
    s2v_best_ys = [r[0] for r in s2v_results]
    s2v_best_var = [r[1] for r in s2v_results]
    if any(y is not None for y in s2v_best_ys):
        ys_s = [y or 0 for y in s2v_best_ys]
        stats_s = [ratio_from_csv_stats(BASE / f's2v_{v}_on_{t}.csv') if v else (None,None,0)
                   for t, v in zip(test_keys, s2v_best_var)]
        picks = " / ".join(f"{t}={v}" for t, v in zip(tests, s2v_best_var) if v)
        ax2.errorbar(tests, ys_s, yerr=[_sem(s) for s in stats_s],
                     marker='o', label=f'S2V-DQN (best: {picks})', color='#9B59B6',
                     linewidth=2, markersize=9, linestyle='--',
                     capsize=4, elinewidth=1.1, alpha=0.9)
        all_y_right.extend(y for y in ys_s if y > 0)

    # REINFORCE: best-across-variants line
    rf_results = [best_reinforce(t) for t in test_keys]
    rf_best_ys = [r[0] for r in rf_results]
    rf_best_var = [r[1] for r in rf_results]
    if any(y is not None for y in rf_best_ys):
        ys_r = [y or 0 for y in rf_best_ys]
        stats_r = [ratio_from_csv_stats(BASE / f'reinforce_{v}_on_{t}.csv') if v else (None,None,0)
                   for t, v in zip(test_keys, rf_best_var)]
        picks = " / ".join(f"{t}={v}" for t, v in zip(tests, rf_best_var) if v)
        meta = RL_SINGLE['reinforce']
        ax2.errorbar(tests, ys_r, yerr=[_sem(s) for s in stats_r],
                     marker=meta['marker'], label=f"REINFORCE (best: {picks})",
                     color=meta['color'], linewidth=2, markersize=9,
                     linestyle=meta['ls'], capsize=4, elinewidth=1.2, alpha=0.9)
        all_y_right.extend(y for y in ys_r if y > 0)

    # DQN — single-variant solver (no graph_type ablation in this run)
    for key, meta in [('dqn', RL_SINGLE['dqn'])]:
        stats = [ratio_from_csv_stats(BASE / f'{key}_on_{t}.csv') for t in test_keys]
        ys      = [s[0] or 0 for s in stats]
        yes_err = [_sem(s) for s in stats]
        if any(y > 0 for y in ys):
            ax2.errorbar(tests, ys, yerr=yes_err,
                         marker=meta['marker'], label=meta['label'], color=meta['color'],
                         linewidth=2, markersize=9, linestyle=meta['ls'],
                         capsize=4, elinewidth=1.2, alpha=0.9)
            all_y_right.extend(y for y in ys if y > 0)

    if any(y is not None for y in greedy_ys):
        ys_g = [y or 0 for y in greedy_ys]
        ax2.plot(tests, ys_g, marker='x', linestyle='--',
                 label='Greedy heuristic', color='#7F77DD',
                 linewidth=1.5, markersize=8)
        all_y_right.extend(y for y in ys_g if y > 0)

    ax2.set_xlabel('Test set (n = number of items; trained on n≤50)', fontsize=11)
    ax2.set_ylabel('Approximation Ratio (vs DP)', fontsize=11)
    ax2.set_title('(b) Cross-method Comparison (best per family)', fontsize=11)
    ax2.legend(loc='lower left', fontsize=9)
    ax2.grid(axis='y', alpha=0.3)

    # Independent y-axis per panel so panel (a) can zoom into GNN variant differences
    # (panel b must accommodate REINFORCE at ~0.93, which would collapse panel a's scale)
    if all_y_left:
        lo = max(min(all_y_left) - 0.003, 0.0)
        hi = min(max(all_y_left) + 0.003, 1.005)
        ax1.set_ylim(lo, hi)
    if all_y_right:
        lo = max(min(all_y_right) - 0.005, 0.0)
        hi = min(max(all_y_right) + 0.005, 1.005)
        ax2.set_ylim(lo, hi)

    plt.suptitle('Cross-scale Generalization (trained on n≤50, tested on larger n)',
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig('plots/chart2_crossscale.png', dpi=200, bbox_inches='tight')
    plt.close()
    print("Saved: plots/chart2_crossscale.png")

    # ===================================================================
    # CHARTS 3, 4, 5: Training curves
    # ===================================================================
    log_dir = args.log_dir or Path('results/GNN')
    variant_logs = {
        'knn':             log_dir / 'gnn_knn_training_log.csv',
        'conflict_static': log_dir / 'gnn_conflict_static_training_log.csv',
        'random':          log_dir / 'gnn_random_training_log.csv',
        'full':            log_dir / 'gnn_full_training_log.csv',
    }
    logs = {k: load_log(p) for k, p in variant_logs.items()}

    if any(v is not None for v in logs.values()):
        plot_training_ratio(logs, Path('plots'))
        plot_training_loss(logs, Path('plots'))
        plot_training_h2h(logs, Path('plots'))
    else:
        print(f"[INFO] No training logs found in {log_dir}. "
              f"Pass --log_dir to specify the directory. Skipping charts 3-5.")

    print("\nDone.")


if __name__ == "__main__":
    main()
