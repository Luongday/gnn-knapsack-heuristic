"""Cross-type evaluation: 1 checkpoint × N Pisinger types × M decode strategies.

Outputs a tidy CSV (one row per type×strategy) plus a Markdown table to stdout —
designed for the defense-ready comparison "GNN+greedy_prob vs GNN+dp_subset vs Greedy".

Usage:
    python -m GNNForKnapSack.src.scripts.cross_type_eval \
        --model_path results/GNN_Pisinger_t3/pisinger_kNN/gnn_knn_best.pt \
        --types_root data \
        --strategies greedy_prob dp_subset \
        --out_csv results/GNN_Pisinger_t3/cross_type/summary.csv

Each `--types_root` is scanned for subdirs starting with `pisinger_t*`; inside each,
the script auto-detects the directory containing `instance_*.npz` files (flat or
under a `test/type_XX_*` subdir).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO_PARENT = _HERE.parents[2]
for _p in [str(_REPO_PARENT), str(_HERE.parent)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from GNNForKnapSack.src.core.config_loader import resolve_device, DECODE_STRATEGIES
from GNNForKnapSack.src.core.decode_utils import decode, greedy_ratio_decode
from GNNForKnapSack.src.core.instance_loader import list_instances, load_instance
from GNNForKnapSack.src.gnn.Knapsack_GNN.Graph_builder import build_knapsack_graph_inference
from GNNForKnapSack.src.gnn.Knapsack_GNN.model import load_checkpoint


def mark(msg: str) -> None:
    print(f"[CROSS-TYPE] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_instance_dir(root: Path) -> Optional[Path]:
    """Locate the directory that actually contains instance_*.npz files.

    Searches `root`, then `root/test/*`, then `root/*` one level deep — covering
    the three layouts present in this repo (t3 has train/val/test split; t4/t6
    have a single type subdir; some are flat).
    """
    if any(root.glob("instance_*.npz")):
        return root

    for candidate in [root / "test", root]:
        if not candidate.is_dir():
            continue
        for sub in sorted(candidate.iterdir()):
            if sub.is_dir() and any(sub.glob("instance_*.npz")):
                return sub
            if sub.is_dir():
                for sub2 in sorted(sub.iterdir()):
                    if sub2.is_dir() and any(sub2.glob("instance_*.npz")):
                        return sub2
    return None


def discover_types(root: Path, prefix: str = "pisinger_t") -> List[Tuple[str, Path]]:
    """Return [(label, instance_dir), ...] for every Pisinger-type folder under root."""
    found: List[Tuple[str, Path]] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or not sub.name.startswith(prefix):
            continue
        if sub.name.endswith("_sanity"):
            continue
        inst_dir = find_instance_dir(sub)
        if inst_dir is None:
            mark(f"  skip {sub.name}: no instance_*.npz found")
            continue
        found.append((sub.name, inst_dir))
    return found


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_one(
    model,
    device: torch.device,
    instance_dir: Path,
    strategy: str,
    *,
    k: int,
    graph_type: str,
    top_m: int,
    beam_width: int,
    n_limit: Optional[int],
) -> Dict[str, float]:
    """Evaluate one (model, instance_dir, strategy) combination.

    Always also runs greedy_ratio baseline per instance for head-to-head stats.
    """
    files = list_instances(instance_dir, limit=n_limit)

    ratios: List[float] = []
    greedy_ratios: List[float] = []
    feasible = 0
    beats = ties = loses = 0
    times_ms: List[float] = []

    for path in files:
        W, V, C = load_instance(path)
        n = len(W)
        npz = np.load(str(path), allow_pickle=True)
        dp_val = float(npz["dp_value"]) if "dp_value" in npz.files else 0.0

        graph = build_knapsack_graph_inference(
            W.tolist(), V.tolist(), int(C),
            k=min(k, max(n - 1, 1)),
            graph_type=graph_type,
        ).to(device)

        W_t = torch.tensor(W, dtype=torch.float32)
        V_t = torch.tensor(V, dtype=torch.float32)

        t0 = time.perf_counter()
        logits = model(graph)
        probs = torch.sigmoid(logits).detach().cpu()
        if probs.shape[0] != n:
            probs = probs[:n] if probs.shape[0] > n else probs
            if probs.shape[0] != n:
                continue

        x_hat = decode(
            probs, W_t, float(C),
            strategy=strategy,
            values=V_t,
            beam_width=beam_width,
            top_m=top_m,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        times_ms.append(elapsed_ms)

        gnn_val = float((x_hat * V_t).sum())
        gnn_w = float((x_hat * W_t).sum())
        is_feas = gnn_w <= float(C) + 1e-6
        feasible += int(is_feas)

        # Greedy ratio baseline (same per instance, independent of model)
        gr_sel = greedy_ratio_decode(V_t, W_t, float(C))
        gr_val = float((gr_sel * V_t).sum())

        if dp_val > 0:
            ratios.append(gnn_val / dp_val)
            greedy_ratios.append(gr_val / dp_val)

        if gnn_val > gr_val + 1e-6:
            beats += 1
        elif abs(gnn_val - gr_val) < 1e-6:
            ties += 1
        else:
            loses += 1

    n_inst = len(files)
    return {
        "n_instances": n_inst,
        "gnn_avg_ratio": float(np.mean(ratios)) if ratios else 0.0,
        "gnn_std_ratio": float(np.std(ratios)) if ratios else 0.0,
        "greedy_avg_ratio": float(np.mean(greedy_ratios)) if greedy_ratios else 0.0,
        "greedy_std_ratio": float(np.std(greedy_ratios)) if greedy_ratios else 0.0,
        "feasibility_rate": feasible / max(n_inst, 1),
        "beats_greedy": beats,
        "ties_greedy": ties,
        "loses_greedy": loses,
        "advantage": (float(np.mean(ratios)) - float(np.mean(greedy_ratios)))
                     if (ratios and greedy_ratios) else 0.0,
        "avg_time_ms": float(np.mean(times_ms)) if times_ms else 0.0,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-type evaluation of a trained GNN checkpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", type=Path, required=True,
                        help="Path to .pt checkpoint")
    parser.add_argument("--types_root", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data",
                        help="Root containing pisinger_t* subdirs")
    parser.add_argument("--explicit_dirs", type=Path, nargs="*", default=None,
                        help="If given, override discovery and use these dirs as-is")
    parser.add_argument("--strategies", type=str, nargs="+",
                        default=["greedy_prob", "dp_subset"],
                        choices=list(DECODE_STRATEGIES))
    parser.add_argument("--top_m", type=int, default=50)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--graph_type", type=str, default="knn",
                        choices=["knn", "conflict_static", "random", "full"])
    parser.add_argument("--n", type=int, default=None,
                        help="Limit instances per type (None = all)")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out_csv", type=Path,
                        default=Path(__file__).resolve().parents[2] /
                        "results" / "GNN_Pisinger_t3" / "cross_type" / "summary.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    mark(f"Loading checkpoint: {args.model_path}")
    model = load_checkpoint(args.model_path, device=device, dropout=0.0)
    model.eval()

    if args.explicit_dirs:
        types_to_eval: List[Tuple[str, Path]] = []
        for d in args.explicit_dirs:
            inst = find_instance_dir(Path(d))
            if inst is None:
                mark(f"  skip {d}: no instance_*.npz")
                continue
            types_to_eval.append((Path(d).name, inst))
    else:
        types_to_eval = discover_types(args.types_root)

    if not types_to_eval:
        mark("ERROR: no Pisinger types found.")
        sys.exit(1)

    mark(f"Found {len(types_to_eval)} type(s):")
    for label, inst_dir in types_to_eval:
        mark(f"  {label:<25} -> {inst_dir}")

    rows: List[Dict] = []
    for label, inst_dir in types_to_eval:
        for strat in args.strategies:
            mark(f"Evaluating {label} | strategy={strat}")
            metrics = evaluate_one(
                model, device, inst_dir, strat,
                k=args.k, graph_type=args.graph_type,
                top_m=args.top_m, beam_width=args.beam_width,
                n_limit=args.n,
            )
            row = {"type": label, "strategy": strat, **metrics}
            rows.append(row)
            mark(
                f"  -> ratio={metrics['gnn_avg_ratio']:.4f}±{metrics['gnn_std_ratio']:.3f} "
                f"| greedy={metrics['greedy_avg_ratio']:.4f} "
                f"| adv={metrics['advantage']:+.4f} "
                f"| beats/ties/loses={metrics['beats_greedy']}/"
                f"{metrics['ties_greedy']}/{metrics['loses_greedy']} "
                f"| feas={metrics['feasibility_rate']*100:.1f}% "
                f"| {metrics['avg_time_ms']:.1f}ms"
            )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["type", "strategy", "n_instances",
                  "gnn_avg_ratio", "gnn_std_ratio",
                  "greedy_avg_ratio", "greedy_std_ratio",
                  "advantage", "feasibility_rate",
                  "beats_greedy", "ties_greedy", "loses_greedy",
                  "avg_time_ms"]
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})
    mark(f"Wrote {len(rows)} rows -> {args.out_csv}")

    print()
    print_markdown_table(rows)


def print_markdown_table(rows: List[Dict]) -> None:
    print("| Type | Strategy | n | GNN ratio | Greedy ratio | Adv | Beats/Ties/Loses | Feas | ms |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['type']} | {r['strategy']} | {r['n_instances']} "
            f"| {r['gnn_avg_ratio']:.4f}±{r['gnn_std_ratio']:.3f} "
            f"| {r['greedy_avg_ratio']:.4f} "
            f"| {r['advantage']:+.4f} "
            f"| {r['beats_greedy']}/{r['ties_greedy']}/{r['loses_greedy']} "
            f"| {r['feasibility_rate']*100:.1f}% "
            f"| {r['avg_time_ms']:.1f} |"
        )


if __name__ == "__main__":
    main()
