"""Analyze results: print 3 tables for the report."""
import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

def load_values(csv_path):
    try:
        rows = list(csv.DictReader(open(csv_path)))
        return {r['instance_file']: float(r['total_value']) for r in rows}
    except FileNotFoundError:
        return {}

def compute_stats(solver_csv, dp_csv):
    s = load_values(solver_csv)
    d = load_values(dp_csv)
    common = set(s) & set(d)
    if not common:
        return None, None, None
    ratios = [s[k] / d[k] if d[k] > 0 else 0 for k in common]
    avg = mean(ratios)
    sd = stdev(ratios) if len(ratios) > 1 else 0.0
    return avg, sd, len(common)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=Path, default=Path('results/cross_eval'))
    args = parser.parse_args()
    BASE = args.results_dir

    print("\n" + "=" * 80)
    print("TABLE 1: Ablation — Graph Structure x Decode Method (tested on n<=50)")
    print("=" * 80)
    print(f"{'Graph':<15} {'Greedy decode':>20} {'DP subset decode':>20}")
    print("-" * 60)
    for g in ['knn', 'conflict_static', 'random', 'full']:
        gd, _, _ = compute_stats(BASE / f'{g}_on_small_greedy.csv', BASE / 'dp_small.csv')
        dd, _, _ = compute_stats(BASE / f'{g}_on_small_dp.csv',     BASE / 'dp_small.csv')
        gd_str = f"{gd:.4f}" if gd else "MISSING"
        dd_str = f"{dd:.4f}" if dd else "MISSING"
        print(f"{g:<15} {gd_str:>20} {dd_str:>20}")

    # Greedy baseline reference
    gb, _, _ = compute_stats(BASE / 'greedy_small.csv', BASE / 'dp_small.csv')
    print("-" * 60)
    print(f"{'(Greedy heur)':<15} {gb:>20.4f} {'--':>20}")

    print("\n" + "=" * 80)
    print("TABLE 2: Cross-scale Generalization (Greedy decode, trained on n<=50)")
    print("=" * 80)
    print(f"{'Model':<15} {'test_small':>15} {'test_n100':>15} {'test_n200':>15}")
    print("-" * 60)
    for g in ['knn', 'conflict_static', 'random', 'full']:
        row = [g]
        for t in ['small', 'n100', 'n200']:
            r, _, _ = compute_stats(BASE / f'{g}_on_{t}_greedy.csv', BASE / f'dp_{t}.csv')
            row.append(f"{r:.4f}" if r else "MISSING")
        print(f"{row[0]:<15} {row[1]:>15} {row[2]:>15} {row[3]:>15}")

    # Greedy baseline per test
    print("-" * 60)
    row = ["(Greedy heur)"]
    for t in ['small', 'n100', 'n200']:
        r, _, _ = compute_stats(BASE / f'greedy_{t}.csv', BASE / f'dp_{t}.csv')
        row.append(f"{r:.4f}" if r else "MISSING")
    print(f"{row[0]:<15} {row[1]:>15} {row[2]:>15} {row[3]:>15}")

    print("\n" + "=" * 80)
    print("TABLE 3: Decode Method Comparison (gnn-kNN best model)")
    print("=" * 80)
    print(f"{'Test set':<15} {'Greedy decode':>18} {'DP subset decode':>20} {'Delta':>12}")
    print("-" * 70)
    for t in ['small', 'n100', 'n200']:
        gd, _, _ = compute_stats(BASE / f'knn_on_{t}_greedy.csv', BASE / f'dp_{t}.csv')
        dd, _, _ = compute_stats(BASE / f'knn_on_{t}_dp.csv',     BASE / f'dp_{t}.csv')
        if gd and dd:
            delta = dd - gd
            print(f"{t:<15} {gd:>18.4f} {dd:>20.4f} {delta:>+12.4f}")

    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD SUMMARY (on test_small, Greedy decode)")
    print("=" * 80)
    greedy_vals = load_values(BASE / 'greedy_small.csv')
    for g in ['knn', 'conflict_static', 'random', 'full']:
        gnn_vals = load_values(BASE / f'{g}_on_small_greedy.csv')
        common = set(gnn_vals) & set(greedy_vals)
        if not common:
            continue
        beats = sum(1 for k in common if gnn_vals[k] > greedy_vals[k] + 1e-6)
        ties  = sum(1 for k in common if abs(gnn_vals[k] - greedy_vals[k]) < 1e-6)
        loses = len(common) - beats - ties
        print(f"gnn-{g:<15} vs Greedy: {beats:>3} wins, {loses:>3} losses, {ties:>3} ties")

    print("\nDone. Copy tables above into your report.")

if __name__ == "__main__":
    main()