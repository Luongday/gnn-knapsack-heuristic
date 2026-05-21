"""REINFORCE (gnn policy gradient) evaluation — CSV schema aligned with all solvers."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_REPO_PARENT = _HERE.parents[3]
for _p in [str(_REPO_PARENT), str(_HERE.parent), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from GNNForKnapSack.src.core.instance_loader import load_instance_with_solution, list_instances
from GNNForKnapSack.src.gnn.Knapsack_GNN.Graph_builder import build_knapsack_graph_inference
from GNNForKnapSack.src.core.decode_utils import decode
from GNNForKnapSack.src.gnn.Knapsack_GNN.model import load_checkpoint

def mark(msg: str) -> None:
    print(f"[REINFORCE-EVAL] {msg}", flush=True)


@torch.no_grad()
def run_reinforce_on_instance(
    model,
    device: torch.device,
    weights: np.ndarray,
    values:  np.ndarray,
    capacity: int,
    k: int = 16,
    graph_type: str = "knn",
    max_conflict_edges: Optional[int] = None,
    decode_strategy: str = "greedy_prob",
    beam_width: int = 5,
    top_m: int = 30,
) -> dict:
    n = len(weights)

    data = build_knapsack_graph_inference(
        weights=weights.tolist(),
        values=values.tolist(),
        capacity=int(capacity),
        k=min(k, max(n - 1, 1)),
        graph_type=graph_type,
        max_conflict_edges=max_conflict_edges,
    )
    data = data.to(device)

    t0 = time.perf_counter()
    logits = model(data)

    if logits.numel() != n:
        raise ValueError(
            f"Model output {logits.numel()} != n_items {n}. "
            f"Check feature dimensions and graph construction."
        )

    probs = torch.sigmoid(logits).cpu()
    w_tensor = torch.tensor(weights, dtype=torch.float32)
    v_tensor = torch.tensor(values,  dtype=torch.float32)

    x_hat = decode(probs, w_tensor, float(capacity),
                   strategy=decode_strategy,
                   values=v_tensor,
                   beam_width=beam_width,
                   top_m=top_m)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    total_weight = float((x_hat * w_tensor).sum().item())
    total_value  = float((x_hat * v_tensor).sum().item())
    feasible     = 1 if total_weight <= float(capacity) + 1e-6 else 0
    selected_idx = [int(i) for i in range(n) if x_hat[i].item() > 0.5]

    return {
        "total_weight":      total_weight,
        "total_value":       total_value,
        "feasible":          feasible,
        "inference_time_ms": round(infer_ms, 4),
        "selected_items":    selected_idx,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate REINFORCE-trained gnn on Knapsack instances.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset_dir", type=Path,
                        default=Path(__file__).resolve().parents[3] / "data" / "knapsack_ilp" / "test")
    parser.add_argument("--model_path",  type=Path,
                        default=Path(__file__).resolve().parents[3] / "results" / "GNN_REINFORCE" / "gnn_reinforce.pt")
    parser.add_argument("--out_csv",     type=Path,
                        default=Path(__file__).resolve().parents[3] / "results" / "GNN_REINFORCE" / "reinforce_eval_results.csv")
    parser.add_argument("--device",      type=str, default="auto",
                        help="Device: auto, cpu, cuda")
    parser.add_argument("--k",           type=int, default=16)
    parser.add_argument("--n",           type=int, default=None, help="Limit to first N instances")
    parser.add_argument("--graph_type", type=str, default="knn",
                        choices=["knn", "conflict_static", "random", "full"],
                        help="Type of graph to build")
    parser.add_argument("--max_conflict_edges", type=int, default=None,
                        help="Maximum number of conflict edges (None = unlimited)")
    parser.add_argument("--decode_strategy", type=str, default="greedy_prob",
                        choices=["greedy_prob", "beam_search", "dp_subset"],
                        help="Decode strategy after model inference")
    parser.add_argument("--beam_width", type=int, default=5,
                        help="Beam width (only for beam_search)")
    parser.add_argument("--top_m", type=int, default=30,
                        help="Top-m items to consider (only for dp_subset)")
    return parser.parse_args()


def main():
    args = parse_args()
    from GNNForKnapSack.src.core.config_loader import resolve_device
    device = resolve_device(args.device)

    mark(f"Loading model: {args.model_path}")
    model = load_checkpoint(args.model_path, device=device, dropout=0.0)
    model.eval()

    files = list_instances(args.dataset_dir, limit=args.n)
    mark(f"Found {len(files)} instances in {args.dataset_dir}")

    results = []
    total_start = time.perf_counter()

    for idx, path in enumerate(files):
        W, V, C, _, dp_value_int = load_instance_with_solution(path)
        dp_value = float(dp_value_int)
        sol = run_reinforce_on_instance(
            model, device, W, V, int(C),
            k=args.k, graph_type=args.graph_type,
            max_conflict_edges=args.max_conflict_edges,
            decode_strategy=args.decode_strategy,
            beam_width=args.beam_width,
            top_m=args.top_m,
        )

        ratio = round(sol["total_value"] / dp_value, 6) if dp_value > 0 else 0.0

        results.append({
            "instance_file":     path.name,
            "n_items":           len(W),
            "capacity":          float(C),
            "total_weight":      sol["total_weight"],
            "total_value":       sol["total_value"],
            "dp_value":          dp_value,
            "ratio":             ratio,
            "feasible":          sol["feasible"],
            "inference_time_ms": sol["inference_time_ms"],
            "selected_items":    json.dumps(sol["selected_items"]),
        })

        if (idx + 1) % 50 == 0:
            mark(f"[{idx+1}/{len(files)}] val={sol['total_value']:.1f} "
                 f"feasible={bool(sol['feasible'])}")

    fieldnames = [
        "instance_file", "n_items", "capacity",
        "total_weight", "total_value", "dp_value", "ratio",
        "feasible", "inference_time_ms", "selected_items",
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    total_time = time.perf_counter() - total_start
    if results:
        avg_val  = float(np.mean([r["total_value"]       for r in results]))
        avg_time = float(np.mean([r["inference_time_ms"] for r in results]))
        feas     = float(np.mean([r["feasible"]          for r in results]))
        ratios   = [r["ratio"] for r in results if r["ratio"] > 0]
        avg_ratio = float(np.mean(ratios)) if ratios else 0.0
        mark(f"Done: {len(results)} in {total_time:.1f}s")
        mark(f"avg_value={avg_val:.2f} | avg_time={avg_time:.3f}ms | "
             f"feasible={feas:.3f} | ratio={avg_ratio:.4f}")
    mark(f"Results -> {args.out_csv}")


if __name__ == "__main__":
    main()