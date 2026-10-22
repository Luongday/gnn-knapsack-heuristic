"""Does a learned scorer add anything over the value/weight ratio?

For each test dir and each GNN checkpoint, compares against DP optimum:
  - greedy by ratio vs greedy by model prob (identical-solution rate)
  - top-m DP subset seeded by ratio vs seeded by model prob
  - Spearman rank correlation between model prob and ratio

Usage:
  python src/scripts/greedy_equivalence.py --test_dirs DIR [DIR ...] \
      --models name=path.pt [name=path.pt ...] [--top_m 30]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.decode_utils import dp_subset_decode, greedy_feasible_decode
from src.core.instance_loader import load_instance_with_solution
from src.gnn.Knapsack_GNN.Graph_builder import build_knapsack_graph_inference
from src.gnn.Knapsack_GNN.model import load_checkpoint


def evaluate_dir(test_dir: Path, models: dict, top_m: int) -> dict:
    rows = {"greedy": [], f"ratio+DP{top_m}": []}
    for name in models:
        rows[name] = []
        rows[f"{name}+DP{top_m}"] = []
        rows[f"{name}:same_as_greedy"] = []
        rows[f"{name}:spearman"] = []

    for f in sorted(test_dir.glob("*.npz")):
        W, V, C, _, opt = load_instance_with_solution(f)
        w = torch.tensor(W, dtype=torch.float32)
        v = torch.tensor(V, dtype=torch.float32)
        ratio = v / w
        x_greedy = greedy_feasible_decode(ratio, w, C)
        rows["greedy"].append(float((x_greedy * v).sum()) / opt)
        rows[f"ratio+DP{top_m}"].append(
            float((dp_subset_decode(ratio, w, v, C, top_m=top_m) * v).sum()) / opt)

        g = build_knapsack_graph_inference(W.tolist(), V.tolist(), int(C))
        for name, model in models.items():
            with torch.no_grad():
                p = torch.sigmoid(model(g)).view(-1)
            x = greedy_feasible_decode(p, w, C)
            rows[name].append(float((x * v).sum()) / opt)
            rows[f"{name}+DP{top_m}"].append(
                float((dp_subset_decode(p, w, v, C, top_m=top_m) * v).sum()) / opt)
            rows[f"{name}:same_as_greedy"].append(float(torch.equal(x, x_greedy)))
            rows[f"{name}:spearman"].append(spearmanr(p.numpy(), ratio.numpy())[0])

    return {k: float(np.mean(vals)) for k, vals in rows.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test_dirs", type=Path, nargs="+", required=True)
    ap.add_argument("--models", nargs="+", required=True, help="name=checkpoint.pt")
    ap.add_argument("--top_m", type=int, default=30)
    args = ap.parse_args()

    device = torch.device("cpu")
    models = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        models[name] = load_checkpoint(path, device)

    for d in args.test_dirs:
        res = evaluate_dir(d, models, args.top_m)
        print(f"\n== {d}")
        for k, val in res.items():
            print(f"  {k:28} {val:.4f}")


if __name__ == "__main__":
    main()
