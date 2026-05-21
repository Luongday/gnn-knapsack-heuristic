"""Benchmark classical and neural solvers on Pisinger hard instances."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from Generate_Hard import (
    generate_hard_dataset, compile_genhard,
    TYPE_NAMES,
)
from GNNForKnapSack.src.core.instance_loader import load_instance, list_instances
from GNNForKnapSack.src.solvers.Greedy.greedy_baseline_eval import solve_knapsack_greedy
from GNNForKnapSack.src.solvers.GA.ga_baseline_eval import solve_knapsack_ga
from GNNForKnapSack.src.gnn.Knapsack_GNN.Dp import solve_knapsack_dp


def mark(msg: str) -> None:
    print(f"\n{'='*70}\n  {msg}\n{'='*70}", flush=True)


def run_cmd(label: str, cmd: list, cwd: Path = None) -> bool:
    mark(f"Running: {label}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    elapsed = time.perf_counter() - t0
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (code={result.returncode})"
    print(f"  {status} in {elapsed:.1f}s")
    return ok


# ---------------------------------------------------------------------------
# Solver wrappers
# ---------------------------------------------------------------------------

def greedy_solver(W, V, C):
    selected = solve_knapsack_greedy(W, V, int(C))
    sol = np.zeros(len(W), dtype=np.int8)
    for i in selected:
        sol[i] = 1
    value = float((V.astype(float) * sol.astype(float)).sum())
    return sol, value


def ga_solver(W, V, C, population=100, generations=500, seed=42):
    selected = solve_knapsack_ga(
        W, V, int(C),
        population_size=population,
        max_generations=generations,
        seed=seed,
    )
    sol = np.zeros(len(W), dtype=np.int8)
    for i in selected:
        sol[i] = 1
    value = float((V.astype(float) * sol.astype(float)).sum())
    return sol, value


def dp_solver(W, V, C):
    raw = solve_knapsack_dp(W.tolist(), V.tolist(), int(C))
    if raw is None:
        sol = np.zeros(len(W), dtype=np.int8)
        return sol, 0.0
    sol = np.zeros(len(W), dtype=np.int8)
    if len(raw) == len(W) and all(x in (0, 1) for x in raw):
        sol = np.asarray(raw, dtype=np.int8)
    else:
        for i in raw:
            idx = int(i)
            if 0 <= idx < len(W):
                sol[idx] = 1
    value = float((V.astype(float) * sol.astype(float)).sum())
    return sol, value


def evaluate_solver_on_dir(dataset_dir: Path, solver_fn, solver_name: str) -> dict:
    files = list_instances(dataset_dir)
    ratios, times, values = [], [], []
    feasible_count = 0

    for path in files:
        W, V, C = load_instance(path)
        arr = np.load(path, allow_pickle=True)
        dp_val = None
        for k in ["dp_value", "optimal_value"]:
            if k in arr.files:
                dp_val = int(arr[k])
                break

        t0 = time.perf_counter()
        solution, value = solver_fn(W, V, C)
        solve_ms = (time.perf_counter() - t0) * 1000.0

        weight = float((W.astype(float) * solution.astype(float)).sum())
        feasible = weight <= float(C) + 1e-6

        if dp_val and dp_val > 0:
            ratio = value / dp_val
            ratios.append(ratio)

        feasible_count += int(feasible)
        times.append(solve_ms)
        values.append(value)

    n = len(files)
    return {
        "solver":        solver_name,
        "n_instances":   n,
        "feasible_rate": feasible_count / max(n, 1),
        "avg_ratio":     float(np.mean(ratios)) if ratios else None,
        "std_ratio":     float(np.std(ratios)) if ratios else None,
        "min_ratio":     float(np.min(ratios)) if ratios else None,
        "avg_time_ms":   float(np.mean(times)),
        "avg_value":     float(np.mean(values)),
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark classical and neural solvers on Pisinger hard instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--types", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--n_items", type=int, default=100)
    parser.add_argument("--num_instances", type=int, default=200)
    parser.add_argument("--range", type=int, default=1000)
    parser.add_argument("--out_dir", type=Path, default=_HERE / "data" / "pisinger")
    parser.add_argument("--results_dir", type=Path, default=_HERE / "results" / "pisinger")
    parser.add_argument("--ga_population",  type=int, default=100)
    parser.add_argument("--ga_generations", type=int, default=500)
    parser.add_argument("--skip_dp", action="store_true")
    parser.add_argument("--skip_ga", action="store_true")
    parser.add_argument("--max_dp_capacity", type=int, default=500_000)

    # gnn evaluation
    parser.add_argument("--run_gnn", action="store_true")
    parser.add_argument("--gnn_model", type=Path, default=None)
    parser.add_argument("--gnn_graph_type", type=str, default="knn",
                        choices=["knn", "conflict_static", "random", "full"])
    parser.add_argument("--gnn_max_conflict_edges", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    py = sys.executable
    genhard_bin = compile_genhard()

    all_results = {}

    for t in args.types:
        type_name = TYPE_NAMES.get(t, f"type_{t}")
        mark(f"Type {t}: {type_name} (n={args.n_items}, {args.num_instances} instances)")

        dataset_dir = args.out_dir / f"type_{t:02d}_{type_name}"
        existing = list(dataset_dir.glob("instance_*.npz")) if dataset_dir.exists() else []
        if len(existing) < args.num_instances:
            generate_hard_dataset(
                out_dir=dataset_dir,
                instance_type=t,
                n_items=args.n_items,
                num_instances=args.num_instances,
                coeff_range=args.range,
                max_dp_capacity=args.max_dp_capacity,
                genhard_bin=genhard_bin,
            )
        else:
            print(f"  Using existing dataset: {dataset_dir} ({len(existing)} instances)")

        print(f"\n  Evaluating classical solvers on type {t}...")
        results_type = {}

        results_type["greedy"] = evaluate_solver_on_dir(dataset_dir, greedy_solver, "Greedy")

        if not args.skip_dp:
            results_type["dp"] = evaluate_solver_on_dir(dataset_dir, dp_solver, "DP")

        if not args.skip_ga:
            def ga_fn(W, V, C):
                return ga_solver(W, V, C, args.ga_population, args.ga_generations)
            results_type["ga"] = evaluate_solver_on_dir(dataset_dir, ga_fn, "GA")

        all_results[t] = results_type

        # gnn evaluation (if requested)
        if args.run_gnn:
            gnn_model = args.gnn_model or (_HERE.parent / "results" / "GNN" / "gnn_best.pt")
            if not gnn_model.exists():
                print(f"  [SKIP] gnn model not found: {gnn_model}")
            else:
                gnn_csv = args.results_dir / f"gnn_type_{t}.csv"
                gnn_cmd = [
                    py, str(_HERE.parent / "GNNForKnapSack" / "Evaluate_GNN.py"),
                    "--dataset_dir", str(dataset_dir),
                    "--model_path", str(gnn_model),
                    "--out_csv", str(gnn_csv),
                    "--graph_type", args.gnn_graph_type,
                ]
                if args.gnn_max_conflict_edges is not None:
                    gnn_cmd.extend(["--max_conflict_edges", str(args.gnn_max_conflict_edges)])
                run_cmd(f"gnn on type {t}", gnn_cmd)

    # Print comparison table
    mark("FINAL COMPARISON — ALL TYPES × CLASSICAL SOLVERS")
    header = f"{'Type':>5} {'Name':>25} | {'Greedy Ratio':>14} {'Greedy Time':>12}"
    if not args.skip_dp:
        header += f" | {'DP Ratio':>10} {'DP Time':>10}"
    if not args.skip_ga:
        header += f" | {'GA Ratio':>10} {'GA Time':>10}"
    print("\n" + header)
    print("-" * len(header))

    for t in args.types:
        name = TYPE_NAMES.get(t, f"type_{t}")
        r = all_results[t]
        gr = r["greedy"]
        gr_ratio = f"{gr['avg_ratio']:.4f}" if gr['avg_ratio'] else "N/A"
        line = f"{t:>5} {name:>25} | {gr_ratio:>14} {gr['avg_time_ms']:>10.3f}ms"

        if not args.skip_dp and "dp" in r:
            dp = r["dp"]
            dp_ratio = f"{dp['avg_ratio']:.4f}" if dp['avg_ratio'] else "N/A"
            line += f" | {dp_ratio:>10} {dp['avg_time_ms']:>8.1f}ms"

        if not args.skip_ga and "ga" in r:
            ga = r["ga"]
            ga_ratio = f"{ga['avg_ratio']:.4f}" if ga['avg_ratio'] else "N/A"
            line += f" | {ga_ratio:>10} {ga['avg_time_ms']:>8.1f}ms"

        print(line)

    # Save results
    args.results_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.results_dir / "pisinger_benchmark.json"
    json_results = {}
    for t, solvers in all_results.items():
        json_results[str(t)] = {}
        for s_name, s_data in solvers.items():
            json_results[str(t)][s_name] = {k: v for k, v in s_data.items() if v is not None}
    with results_path.open("w") as f:
        json.dump({
            "n_items": args.n_items,
            "num_instances": args.num_instances,
            "coeff_range": args.range,
            "types": args.types,
            "results": json_results,
        }, f, indent=2)
    print(f"\nResults saved -> {results_path}")

    print("\n" + "=" * 70)
    print("KEY FINDINGS:")
    for t in args.types:
        if t in all_results and all_results[t]["greedy"]["avg_ratio"]:
            gr_r = all_results[t]["greedy"]["avg_ratio"]
            name = TYPE_NAMES.get(t, "")
            if gr_r < 0.95:
                print(f"  Type {t} ({name}): Greedy ratio = {gr_r:.4f} ← GREEDY BREAKS DOWN")
            elif gr_r < 0.99:
                print(f"  Type {t} ({name}): Greedy ratio = {gr_r:.4f} ← Notable gap")
            else:
                print(f"  Type {t} ({name}): Greedy ratio = {gr_r:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()