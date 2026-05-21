"""Knapsack graph builder — supports KNN, conflict, random, full."""

from __future__ import annotations

import math
from typing import List

import torch
from torch_geometric.data import Data

DEFAULT_K = 16
IN_DIM    = 7


def adaptive_k(n: int, base_k: int = 16) -> int:
    k_adaptive = max(base_k // 2, int(math.sqrt(n) * 1.6))
    return min(k_adaptive, max(1, n - 1))


def _build_random_edges(n: int, k: int, seed: int = 0) -> torch.Tensor:
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    k_eff = min(k, n - 1)
    g = torch.Generator().manual_seed(seed)
    neighbors = torch.randint(0, n - 1, (n, k_eff), generator=g, dtype=torch.long)
    row_idx = torch.arange(n).unsqueeze(1).expand(-1, k_eff)
    neighbors = torch.where(neighbors >= row_idx, neighbors + 1, neighbors)
    rows = row_idx.reshape(-1)
    cols = neighbors.reshape(-1)
    return torch.stack([rows, cols], dim=0)


def _build_full_edges(n: int) -> torch.Tensor:
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    idx = torch.arange(n)
    rows = idx.unsqueeze(1).expand(-1, n).reshape(-1)
    cols = idx.unsqueeze(0).expand(n, -1).reshape(-1)
    mask = rows != cols
    return torch.stack([rows[mask], cols[mask]], dim=0)


def _build_knn_edges(x: torch.Tensor, k: int) -> torch.Tensor:
    n = x.size(0)
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    k_eff = min(k, n - 1)
    dist  = torch.cdist(x, x, p=2)
    dist.fill_diagonal_(float("inf"))
    knn_idx = dist.topk(k_eff, largest=False).indices

    row = torch.arange(n).unsqueeze(1).expand(-1, k_eff).reshape(-1)
    col = knn_idx.reshape(-1)
    return torch.stack([row, col], dim=0)


def _build_conflict_edges_static(
    weights: torch.Tensor,
    capacity: float,
    max_edges: int | None = None
) -> torch.Tensor:
    n = weights.size(0)
    if n <= 1:
        return torch.empty((2, 0), dtype=torch.long)

    w_i = weights.unsqueeze(1)
    w_j = weights.unsqueeze(0)
    conflict_mask = (w_i + w_j) > capacity
    conflict_mask.fill_diagonal_(False)
    conflict_mask = torch.triu(conflict_mask, diagonal=1)

    row, col = conflict_mask.nonzero(as_tuple=True)
    if row.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)

    edge_index = torch.stack([row, col], dim=0)
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

    if max_edges is not None and edge_index.size(1) > max_edges:
        perm = torch.randperm(edge_index.size(1))[:max_edges]
        edge_index = edge_index[:, perm]

    return edge_index


def build_edges(
    x: torch.Tensor,
    weights: torch.Tensor,
    capacity: float,
    graph_type: str = "knn",
    k: int = DEFAULT_K,
    max_edges: int | None = None,
) -> torch.Tensor:
    n = x.size(0)
    if graph_type == "knn":
        if n <= 1:
            return torch.empty((2, 0), dtype=torch.long)
        k_used = adaptive_k(n, base_k=k)
        return _build_knn_edges(x, k=k_used)
    elif graph_type == "conflict_static":
        return _build_conflict_edges_static(weights, capacity, max_edges)
    elif graph_type == "conflict_dynamic":
        raise ValueError(
            "conflict_dynamic only applies inside RL environments, "
            "not for supervised gnn training. Use 'conflict_static' instead."
        )
    elif graph_type == "random":
        return _build_random_edges(n, k=k)
    elif graph_type == "full":
        return _build_full_edges(n)
    else:
        raise ValueError(f"Unsupported graph_type: {graph_type}")


def build_knapsack_graph(
    weights:  List[int],
    values:   List[int],
    capacity: int,
    solution: List[int],
    k:        int = DEFAULT_K,
    graph_type: str = "knn",
    max_conflict_edges: int | None = None,
) -> Data:
    w   = torch.tensor(weights,  dtype=torch.float32)
    v   = torch.tensor(values,   dtype=torch.float32)
    sol = torch.tensor(solution, dtype=torch.float32)
    n   = len(weights)

    ratio      = v / (w + 1e-8)
    w_norm     = w     / (w.max()     + 1e-8)
    v_norm     = v     / (v.max()     + 1e-8)
    ratio_norm = ratio / (ratio.max() + 1e-8)

    cap_ratio = w / (float(capacity) + 1e-8)

    w_sum     = w.sum().item() + 1e-8
    w_mean    = w_sum / max(n, 1)
    cap_util  = torch.full((n,), float(capacity) / w_sum)
    item_frac = torch.full((n,), 1.0 / max(n, 1))

    w_vs_mean = w / w_mean

    x = torch.stack([
        w_norm, v_norm, ratio_norm,
        cap_ratio, cap_util, item_frac,
        w_vs_mean,
    ], dim=1)

    edge_index = build_edges(
        x=x,
        weights=w,
        capacity=float(capacity),
        graph_type=graph_type,
        k=k,
        max_edges=max_conflict_edges,
    )

    y = sol.unsqueeze(1)

    return Data(
        x          = x,
        edge_index = edge_index,
        y          = y,
        wts        = w,
        vals       = v,
        cap        = torch.tensor([capacity], dtype=torch.float32),
    )


def build_knapsack_graph_inference(
    weights:  List[int],
    values:   List[int],
    capacity: int,
    k:        int = DEFAULT_K,
    graph_type: str = "knn",
    max_conflict_edges: int | None = None,
) -> Data:
    dummy_solution = [0] * len(weights)
    return build_knapsack_graph(
        weights, values, capacity, dummy_solution,
        k=k, graph_type=graph_type, max_conflict_edges=max_conflict_edges
    )