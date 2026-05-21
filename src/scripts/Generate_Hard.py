"""Pisinger Hard Knapsack Instance Generator — improved."""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TYPE_NAMES = {
    1:  "uncorrelated",
    2:  "weakly_correlated",
    3:  "strongly_correlated",
    4:  "inverse_strongly_corr",
    5:  "almost_strongly_corr",
    6:  "subset_sum",
    7:  "even_odd_subset_sum",
    8:  "even_odd_knapsack",
    9:  "uncorr_similar_weights",
    11: "uncorr_span",
    12: "weak_corr_span",
    13: "strong_corr_span",
    14: "mstr",
    15: "pceil",
    16: "circle",
}

_HERE = Path(__file__).resolve().parent
DEFAULT_GENHARD_BIN = _HERE / ("genhard.exe" if os.name == "nt" else "genhard")


# ---------------------------------------------------------------------------
# Genhard C compilation
# ---------------------------------------------------------------------------

def compile_genhard(
    source_path: Path = None,
    output_path: Path = None,
) -> Path:
    """Compile Genhard.c if binary doesn't exist."""
    if output_path is None:
        output_path = DEFAULT_GENHARD_BIN
        if os.name == "nt" and not str(output_path).endswith(".exe"):
            output_path = output_path.with_suffix(".exe")
    if source_path is None:
        candidates = [_HERE / "Genhard.c"]
        source_path = next((p for p in candidates if p.exists()), None)
        if source_path is None:
            raise FileNotFoundError("Genhard.c not found")

    if output_path.exists():
        return output_path

    print(f"[GENHARD] Compiling {source_path} → {output_path}")
    src = source_path.read_text()
    if "void main" in src:
        src = src.replace("void main", "int main")
    if "void error(" not in src:
        src = src.replace(
            '#include <math.h>',
            '#include <math.h>\n\nvoid error(const char *msg) {\n  fprintf(stderr, "ERROR: %s\\n", msg);\n  exit(1);\n}\n'
        )
    with tempfile.NamedTemporaryFile(suffix=".c", delete=False, mode="w") as f:
        f.write(src)
        fixed_path = Path(f.name)
    result = subprocess.run(
        ["gcc", "-O2", "-o", str(output_path), str(fixed_path), "-lm"],
        capture_output=True, text=True,
    )
    fixed_path.unlink(missing_ok=True)  # dọn temp file sau khi compile
    if result.returncode != 0:
        raise RuntimeError(f"Compilation failed:\n{result.stderr}")
    print(f"[GENHARD] Compiled successfully: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Parse Genhard output
# ---------------------------------------------------------------------------

def parse_genhard_output(filepath: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    with open(filepath) as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    n = int(lines[0])
    values  = np.zeros(n, dtype=np.int32)
    weights = np.zeros(n, dtype=np.int32)
    for i in range(n):
        parts = lines[1 + i].split()
        idx = int(parts[0])
        values[idx]  = int(parts[1])
        weights[idx] = int(parts[2])
    capacity = int(lines[1 + n])
    return weights, values, capacity


# ---------------------------------------------------------------------------
# Call Genhard binary
# ---------------------------------------------------------------------------

def call_genhard(
    genhard_bin: Path,
    n_items: int,
    coeff_range: int,
    instance_type: int,
    instance_id: int,
    series_size: int,
    work_dir: Path,
) -> Tuple[np.ndarray, np.ndarray, int]:
    result = subprocess.run(
        [
            str(genhard_bin),
            str(n_items),
            str(coeff_range),
            str(instance_type),
            str(instance_id),
            str(series_size),
        ],
        capture_output=True, text=True,
        cwd=str(work_dir),
    )
    if result.returncode != 0:
        raise RuntimeError(f"Genhard failed: {result.stderr}")
    test_in = work_dir / "test.in"
    if not test_in.exists():
        raise FileNotFoundError("Genhard did not create test.in")
    return parse_genhard_output(test_in)


# ---------------------------------------------------------------------------
# DP solver (imported from Dp.py)
# ---------------------------------------------------------------------------

# Sử dụng DP từ module chung
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from GNNForKnapSack.src.gnn.Knapsack_GNN.Dp import solve_knapsack_dp


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_hard_dataset(
    out_dir:        Path,
    instance_type:  int,
    n_items:        int,
    num_instances:  int,
    coeff_range:    int   = 1000,
    series_size:    int   = 1000,
    max_dp_capacity: int  = 500_000,
    genhard_bin:    Optional[Path] = None,
    verbose:        bool   = True,
    prebuild_graph: bool   = False,
    seed:           Optional[int] = None,
) -> Dict:
    """Generate a dataset of Pisinger hard instances.

    Parameters
    ----------
    seed : int, optional
        Random seed for selecting instance IDs from the series.
        If None, deterministic (evenly spaced) selection is used.
    """
    if genhard_bin is None or isinstance(genhard_bin, str):
        genhard_bin = compile_genhard()
    if not genhard_bin.exists():
        genhard_bin = compile_genhard()

    out_dir.mkdir(parents=True, exist_ok=True)
    type_name = TYPE_NAMES.get(instance_type, f"type_{instance_type}")

    # ----- Chọn danh sách instance_id -----
    if seed is not None:
        rng = random.Random(seed)
        if num_instances <= series_size:
            instance_ids = rng.sample(range(1, series_size + 1), num_instances)
        else:
            # nếu cần nhiều hơn số lượng có sẵn, lấy mẫu có hoàn lại
            instance_ids = rng.choices(range(1, series_size + 1), k=num_instances)
        if verbose:
            print(f"[GENHARD] Using seed={seed}, random instance selection")
    else:
        # Công thức cũ – tất định
        instance_ids = [
            max(1, int((i + 1) / num_instances * series_size))
            for i in range(num_instances)
        ]
        if verbose:
            print("[GENHARD] Using deterministic (evenly spaced) instance selection")

    if verbose:
        print(f"\n[GENHARD] Generating {num_instances} instances")
        print(f"  Type: {instance_type} ({type_name})")
        print(f"  n_items: {n_items}, range: {coeff_range}, series: {series_size}")
        print(f"  Output: {out_dir}")

    stats = {
        "generated": 0,
        "dp_solved": 0,
        "dp_skipped": 0,
        "dp_values": [],
        "capacities": [],
    }

    start_time = time.perf_counter()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        for idx in range(num_instances):
            instance_id = instance_ids[idx]   # dùng ID đã chọn

            try:
                W, V, C = call_genhard(
                    genhard_bin, n_items, coeff_range,
                    instance_type, instance_id, series_size,
                    tmpdir,
                )
            except (RuntimeError, FileNotFoundError) as e:
                if verbose:
                    print(f"  [SKIP] instance {idx}: {e}")
                continue

            # Solve with DP
            if C <= max_dp_capacity:
                solution = solve_knapsack_dp(W.tolist(), V.tolist(), C)
                if solution is not None:
                    opt_value = int((np.array(solution) * V).sum())
                    stats["dp_solved"] += 1
                else:
                    solution = [0] * len(W)
                    opt_value = 0
                    stats["dp_skipped"] += 1
            else:
                solution = [0] * len(W)
                opt_value = 0
                stats["dp_skipped"] += 1

            # Save NPZ
            np.savez_compressed(
                out_dir / f"instance_{idx:04d}.npz",
                weights=W,
                values=V,
                capacity=np.int32(C),
                solution=np.array(solution, dtype=np.int8),
                dp_value=np.int32(opt_value),
            )

            # Prebuild PyG graph if requested
            if prebuild_graph:
                from GNNForKnapSack.src.gnn.Knapsack_GNN.Graph_builder import build_knapsack_graph
                graph = build_knapsack_graph(
                    W.tolist(), V.tolist(), C, solution,
                    k=16, graph_type="knn"
                )
                torch.save(graph, out_dir / f"instance_{idx:04d}.pt")

            stats["generated"] += 1
            stats["dp_values"].append(opt_value)
            stats["capacities"].append(C)

            if verbose and ((idx + 1) % 50 == 0 or idx == 0):
                elapsed = time.perf_counter() - start_time
                rate = (idx + 1) / elapsed
                eta = (num_instances - idx - 1) / rate if rate > 0 else 0
                print(f"  [{idx+1}/{num_instances}] elapsed={elapsed:.1f}s "
                      f"ETA={eta:.1f}s  cap={C}  opt={opt_value}")

    total_time = time.perf_counter() - start_time

    meta = {
        "generator":       "Pisinger/Genhard.c",
        "paper":           "Where are the hard knapsack problems (2005)",
        "instance_type":   instance_type,
        "type_name":       type_name,
        "n_items":         n_items,
        "coeff_range":     coeff_range,
        "series_size":     series_size,
        "num_instances":   stats["generated"],
        "dp_solved":       stats["dp_solved"],
        "dp_skipped":      stats["dp_skipped"],
        "avg_capacity":    float(np.mean(stats["capacities"])) if stats["capacities"] else 0,
        "avg_dp_value":    float(np.mean(stats["dp_values"])) if stats["dp_values"] else 0,
        "total_time_sec":  round(total_time, 3),
        "prebuild_graph":  prebuild_graph,
        "seed":            seed,                # lưu lại seed đã dùng
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    if verbose:
        print(f"\n  Done: {stats['generated']} instances in {total_time:.1f}s")
        print(f"  DP solved: {stats['dp_solved']}, skipped: {stats['dp_skipped']}")
        if stats["capacities"]:
            print(f"  Capacity range: {min(stats['capacities'])} - {max(stats['capacities'])}")

    return stats


def generate_multi_type_dataset(
    base_dir:       Path,
    types:          List[int],
    n_items:        int,
    num_per_type:   int,
    coeff_range:    int  = 1000,
    prebuild_graph: bool = False,
    seed:           Optional[int] = None,
    **kwargs,
) -> Dict[int, Dict]:
    """Generate datasets for multiple Pisinger types."""
    all_stats = {}
    for t in types:
        type_name = TYPE_NAMES.get(t, f"type_{t}")
        out_dir = base_dir / f"type_{t:02d}_{type_name}"
        stats = generate_hard_dataset(
            out_dir=out_dir,
            instance_type=t,
            n_items=n_items,
            num_instances=num_per_type,
            coeff_range=coeff_range,
            prebuild_graph=prebuild_graph,
            seed=seed,
            **kwargs,
        )
        all_stats[t] = stats

    print("\n" + "=" * 60)
    print("GENERATION SUMMARY")
    print("=" * 60)
    for t, s in all_stats.items():
        name = TYPE_NAMES.get(t, f"type_{t}")
        print(f"  Type {t:>2} ({name:>25}): {s['generated']:>4} instances, "
              f"DP solved: {s['dp_solved']}")

    return all_stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Pisinger hard Knapsack instances using Genhard.c",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--type", type=int, default=None,
                        help="Single instance type (1-16)")
    parser.add_argument("--types", type=int, nargs="+", default=None,
                        help="Multiple instance types to generate")
    parser.add_argument("--n_items", type=int, default=100)
    parser.add_argument("--num_instances", type=int, default=200)
    parser.add_argument("--range", type=int, default=1000)
    parser.add_argument("--series", type=int, default=1000)
    parser.add_argument("--max_dp_capacity", type=int, default=600_000)
    parser.add_argument("--out_dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data" / "pisinger")
    parser.add_argument("--prebuild_graph", action="store_true",
                        help="Prebuild PyG Data objects and save as .pt files")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for instance ID selection.")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    verbose = not args.quiet

    if args.types:
        types = args.types
    elif args.type:
        types = [args.type]
    else:
        types = [1, 2, 3, 5, 6]
        print("[GENHARD] No type specified, generating key types: "
              f"{types}")

    if len(types) == 1:
        t = types[0]
        type_name = TYPE_NAMES.get(t, f"type_{t}")
        out_dir = args.out_dir / f"type_{t:02d}_{type_name}"
        generate_hard_dataset(
            out_dir=out_dir,
            instance_type=t,
            n_items=args.n_items,
            num_instances=args.num_instances,
            coeff_range=args.range,
            series_size=args.series,
            max_dp_capacity=args.max_dp_capacity,
            verbose=verbose,
            prebuild_graph=args.prebuild_graph,
            seed=args.seed,          # truyền seed
        )
    else:
        generate_multi_type_dataset(
            base_dir=args.out_dir,
            types=types,
            n_items=args.n_items,
            num_per_type=args.num_instances,
            coeff_range=args.range,
            series_size=args.series,
            max_dp_capacity=args.max_dp_capacity,
            verbose=verbose,
            prebuild_graph=args.prebuild_graph,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()