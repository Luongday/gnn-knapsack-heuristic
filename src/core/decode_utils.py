"""Centralized decoding utilities for Knapsack solvers"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple

import torch


DecodeStrategy = Literal[
    "greedy_prob", "greedy_ratio", "beam_search",
    "dp_subset", "diversified_beam", "sample"
]

# ---------------------------------------------------------------------------
# Core greedy decoder
# ---------------------------------------------------------------------------
def greedy_feasible_decode(
    scores:   torch.Tensor,
    weights:  torch.Tensor,
    capacity: float,
    tol:      float = 1e-6,
) -> torch.Tensor:
    """Sort by scores (descending) and greedily add items while respecting capacity."""
    assert scores.dim() == 1 and weights.dim() == 1, \
        f"Expected 1D tensors, got scores={scores.shape}, weights={weights.shape}"
    assert scores.shape == weights.shape, \
        f"Shape mismatch: scores={scores.shape}, weights={weights.shape}"

    # Move to CPU for stable sorting and looping (fast enough for n<=1000)
    scores = scores.detach().cpu()
    weights = weights.detach().cpu()

    idx = torch.argsort(scores, descending=True)
    x_hat = torch.zeros_like(scores)
    total_weight = 0.0

    for i in idx:
        w = float(weights[i])
        if total_weight + w <= capacity + tol:
            x_hat[i] = 1.0
            total_weight += w

    return x_hat


def greedy_ratio_decode(
    values:   torch.Tensor,
    weights:  torch.Tensor,
    capacity: float,
    tol:      float = 1e-6,
) -> torch.Tensor:
    """Classic greedy by value/weight ratio."""
    ratios = values / (weights + 1e-8)
    return greedy_feasible_decode(ratios, weights, capacity, tol)

# ---------------------------------------------------------------------------
# DP on top-k subset
# ---------------------------------------------------------------------------
def dp_subset_decode(
        probs: torch.Tensor,
        W: torch.Tensor,
        V: torch.Tensor,
        C: float,
        top_m: int = 50,
        max_top_m: int = 600,
) -> torch.Tensor:
    """Decode by picking top-m items by gnn score, then run DP on subset.

    Guarantees: local optimum within the top-m filtered set.
    """
    from GNNForKnapSack.src.gnn.Knapsack_GNN.Dp import solve_knapsack_dp

    n = len(probs)
    m = min(top_m, n, max_top_m)

    if top_m > max_top_m:
        print(f"[decode_utils] Warning: top_m={top_m} capped to {max_top_m} for performance.")

    top_indices = torch.topk(probs, m).indices
    W_sub = W[top_indices].round().long().tolist()
    V_sub = V[top_indices].round().long().tolist()

    sub_solution = solve_knapsack_dp(W_sub, V_sub, int(C))

    x_hat = torch.zeros(n, dtype=torch.float32)
    for local_i, selected in enumerate(sub_solution):
        if selected == 1:
            x_hat[top_indices[local_i]] = 1.0

    return x_hat

# def dp_subset_decode(
#     probs: torch.Tensor,
#     values: torch.Tensor,
#     weights: torch.Tensor,
#     capacity: float,
#     top_m: int = 400,
#     max_top_m: int = 700,
#     alpha: float = 0.62,
#     beta: float = 0.28,
#     gamma: float = 0.10,
#     tol: float = 1e-5,
# ) -> tuple[torch.Tensor, float]:
#     from GNNForKnapSack.src.gnn.Knapsack_GNN.Dp import solve_knapsack_dp
#
#     n = probs.shape[0]
#     device = probs.device
#
#     m = min(top_m, n, max_top_m)
#     if top_m > max_top_m:
#         print(f"[HighQ-Decode] top_m capped from {top_m} to {max_top_m}")
#     vw_ratio = values / (weights + 1e-8)
#     vw_norm = (vw_ratio - vw_ratio.min()) / (vw_ratio.max() - vw_ratio.min() + 1e-8)
#     value_norm = (values - values.min()) / (values.max() - values.min() + 1e-8)
#
#     combined_score = alpha * probs + beta * vw_norm + gamma * value_norm
#
#     # Lấy top-m items theo combined score
#     top_indices = torch.topk(combined_score, m).indices
#
#     # Chuẩn bị subset cho DP
#     W_sub = weights[top_indices].round().long().cpu().tolist()
#     V_sub = values[top_indices].round().long().cpu().tolist()
#     C_int = int(capacity + tol)
#
#     # ====================== CHẠY DP EXACT ======================
#     sub_solution = solve_knapsack_dp(W_sub, V_sub, C_int)
#
#     # Xây dựng nghiệm từ DP
#     x_hat = torch.zeros(n, dtype=torch.float32, device=device)
#     dp_value = 0.0
#     dp_weight = 0.0
#
#     for local_i, selected in enumerate(sub_solution):
#         if selected == 1:
#             idx = top_indices[local_i].item()
#             x_hat[idx] = 1.0
#             dp_value += float(values[idx])
#             dp_weight += float(weights[idx])
#
#     best_x = x_hat.clone()
#     best_value = dp_value
#
#     fallback_candidates = []
#
#     # 1. Greedy theo combined_score
#     x1 = greedy_feasible_decode(combined_score, weights, capacity, tol)
#     fallback_candidates.append((x1, float((x1 * values).sum())))
#
#     # 2. Greedy theo value/weight ratio cổ điển
#     x2 = greedy_ratio_decode(values, weights, capacity, tol)
#     fallback_candidates.append((x2, float((x2 * values).sum())))
#
#     # 3. Greedy theo probs GNN thuần
#     x3 = greedy_feasible_decode(probs, weights, capacity, tol)
#     fallback_candidates.append((x3, float((x3 * values).sum())))
#
#     # 4. Greedy theo value thuần (thêm diversity)
#     x4 = greedy_feasible_decode(values, weights, capacity, tol)
#     fallback_candidates.append((x4, float((x4 * values).sum())))
#
#     # Chọn nghiệm tốt nhất từ tất cả candidates
#     for x_cand, val_cand in fallback_candidates:
#         if val_cand > best_value + 1.0:  # cải thiện ít nhất 1 đơn vị value
#             best_x = x_cand
#             best_value = val_cand
#             print(f"[HighQ-Decode] Fallback improved value to {best_value:.2f}")
#
#     selected = int(best_x.sum().item())
#     print(f"[HighQ-Decode] FINAL RESULT → Selected: {selected} items | "
#           f"Best Value = {best_value:.2f} | Weight used ≈ {dp_weight:.1f}/{capacity}")
#
#     return best_x, best_value

# ---------------------------------------------------------------------------
# Sampling (for RL)
# ---------------------------------------------------------------------------
def sample_decode(
    logits:   torch.Tensor,
    weights:  torch.Tensor,
    values:   torch.Tensor,
    capacity: float,
    tol:      float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample a binary mask from Bernoulli(logits) and repair to feasibility.

    Returns:
        mask:   Binary selection mask.
        log_prob: Sum of log-probabilities of sampled actions.
    """
    probs = torch.sigmoid(logits.detach().cpu())
    dist = torch.distributions.Bernoulli(probs)
    sampled = dist.sample()
    log_prob = dist.log_prob(sampled).sum()

    # Repair
    mask = repair_solution(sampled, weights, values, capacity, tol)
    return mask, log_prob

# ---------------------------------------------------------------------------
# Repair utility
# ---------------------------------------------------------------------------
def repair_solution(
    mask:     torch.Tensor,
    weights:  torch.Tensor,
    values:   torch.Tensor,
    capacity: float,
    tol:      float = 1e-6,
) -> torch.Tensor:
    """Make a binary mask feasible by removing items with lowest value/weight ratio.

    Items are removed until total weight ≤ capacity.
    Returns a new mask (does not modify input).
    """
    mask = mask.detach().cpu().clone()
    weights = weights.detach().cpu()
    values = values.detach().cpu()

    total_w = float((mask * weights).sum())
    if total_w <= capacity + tol:
        return mask

    # Find indices of selected items
    sel_idx = torch.where(mask > 0.5)[0]
    if len(sel_idx) == 0:
        return mask

    # Compute ratio for selected items
    ratios = values[sel_idx] / (weights[sel_idx] + 1e-8)
    # Sort ascending (remove worst ratio first)
    order = sel_idx[torch.argsort(ratios)]

    for i in order:
        if total_w <= capacity + tol:
            break
        mask[i] = 0.0
        total_w -= float(weights[i])

    return mask

# ---------------------------------------------------------------------------
# Beam search (improved)
# ---------------------------------------------------------------------------
def beam_search_decode(
    probs:       torch.Tensor,
    weights:     torch.Tensor,
    capacity:    float,
    values: Optional[torch.Tensor] = None,
    beam_width:  int = 5,
    tol:         float = 1e-6,
) -> torch.Tensor:
    """Beam search decoding — usually better quality than pure greedy."""
    probs = probs.detach().cpu()
    weights = weights.detach().cpu()
    n = len(probs)

    item_order = torch.argsort(probs, descending=True)
    if values is not None:
        values = values.detach().cpu()
        score_fn = lambda val, wt, mask, i, p, w, v: (val + float(v), wt + w)
    else:
        score_fn = lambda val, wt, mask, i, p, w, v: (val + float(p), wt + w)
    # Each beam: (total_value, total_weight, selected_mask)
    beams = [(0.0, 0.0, torch.zeros(n, dtype=torch.float32))]

    for i in item_order:
        new_beams = []
        p = float(probs[i])
        w = float(weights[i])
        v = float(values[i]) if values is not None else 0.0

        for val, wt, mask in beams:
            # Không chọn item i
            new_beams.append((val, wt, mask.clone()))

            # Chọn item i (nếu feasible)
            if wt + w <= capacity + tol:
                new_mask = mask.clone()
                new_mask[i] = 1.0
                new_score, new_wt = score_fn(val, wt, new_mask, i, p, w, v)
                new_beams.append((new_score, new_wt, new_mask))

        # Giữ top beam_width beams
        beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:beam_width]

    # Chọn beam tốt nhất
    _, _, best_mask = beams[0]
    return best_mask

def diversified_beam_search_decode(
    probs:       torch.Tensor,
    weights:     torch.Tensor,
    capacity:    float,
    values:      torch.Tensor,
    beam_width:  int = 5,
    diversity_lambda: float = 0.1,
    tol:         float = 1e-6,
) -> torch.Tensor:
    """Beam search with diversity penalty to explore different weight combinations.

    Score = total_value - diversity_lambda * (total_weight / capacity).
    This encourages beams that use different portions of the capacity.
    """
    probs = probs.detach().cpu()
    weights = weights.detach().cpu()
    values = values.detach().cpu()
    n = len(probs)

    beams = [(0.0, 0.0, torch.zeros(n, dtype=torch.float32))]

    for i in range(n):
        new_beams = []
        p = float(probs[i])
        w = float(weights[i])
        v = float(values[i])

        for val, wt, mask in beams:
            # Skip
            new_beams.append((val, val, wt, mask.clone()))

            # Take
            if wt + w <= capacity + tol:
                new_mask = mask.clone()
                new_mask[i] = 1.0
                new_val = val + v
                new_wt = wt + w
                # Diversity penalty: slight preference for lower weight utilization
                # (so beams don't all fill the knapsack identically)
                score = new_val - diversity_lambda * (new_wt / capacity) * new_val
                new_beams.append((score, new_val, new_wt, new_mask))

        # Keep top beam_width distinct beams (by mask to avoid duplicates)
        unique_beams = []
        seen_masks = set()
        for item in sorted(new_beams, key=lambda x: x[0], reverse=True):
            mask_tuple = tuple(item[-1].tolist())
            if mask_tuple not in seen_masks:
                seen_masks.add(mask_tuple)
                unique_beams.append((item[1], item[2], item[3]))  # (val, wt, mask)
                if len(unique_beams) >= beam_width:
                    break
        beams = unique_beams

    _, _, best_mask = beams[0]
    return best_mask

# ---------------------------------------------------------------------------
# Unified decode interface
# ---------------------------------------------------------------------------

def decode(
    scores_or_probs: torch.Tensor,
    weights:         torch.Tensor,
    capacity:        float,
    strategy:        DecodeStrategy = "greedy_prob",
    values:          Optional[torch.Tensor] = None,
    beam_width:      int = 5,
    diversity_lambda: float = 0.1,
    tol:             float = 1e-6,
    top_m:           int = 30,
) -> torch.Tensor:
    """Unified decode interface — recommended way to call decoder."""
    if strategy == "greedy_prob":
        return greedy_feasible_decode(scores_or_probs, weights, capacity, tol)
    elif strategy == "greedy_ratio":
        assert values is not None, "values must be provided for greedy_ratio"
        return greedy_ratio_decode(values, weights, capacity, tol)
    elif strategy == "beam_search":
        return beam_search_decode(scores_or_probs, weights, capacity, beam_width=beam_width, tol=tol)
    elif strategy == "diversified_beam":
        assert values is not None, "values required for diversified_beam"
        return diversified_beam_search_decode(
            scores_or_probs, weights, capacity,
            values=values, beam_width=beam_width,
            diversity_lambda=diversity_lambda, tol=tol
        )
    elif strategy == "dp_subset":
        assert values is not None, "values (V) must be provided for dp_subset"
        return dp_subset_decode(
            probs=scores_or_probs,
            W=weights,
            V=values,
            C=capacity,
            top_m=top_m
        )
    elif strategy == "sample":
        assert values is not None, "values required for sample"
        mask, _ = sample_decode(scores_or_probs, weights, values, capacity, tol)
        return mask
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------
def decode_to_solution_dict(
    x_hat:    torch.Tensor,
    weights:  torch.Tensor,
    values:   torch.Tensor,
    capacity: float,
    tol:      float = 1e-6,
) -> Dict:
    """Convert binary solution to structured result."""
    x_hat = x_hat.detach().cpu()
    weights = weights.detach().cpu()
    values = values.detach().cpu()

    total_weight = float((x_hat * weights).sum().item())
    total_value = float((x_hat * values).sum().item())
    feasible = total_weight <= capacity + tol
    selected = [int(i) for i in range(len(x_hat)) if x_hat[i] > 0.5]

    return {
        "total_value":    round(total_value, 4),
        "total_weight":   round(total_weight, 4),
        "feasible":       feasible,
        "selected_items": selected,
        "n_selected":     len(selected),
        "ratio_to_optimal": None,   # sẽ được điền sau khi có DP value
    }


def compute_ratio(solver_value: float, optimal_value: float) -> Optional[float]:
    """Approximation ratio: solver / optimal"""
    if optimal_value is None or optimal_value <= 0:
        return None
    return solver_value / optimal_value


def compute_gap(solver_value: float, optimal_value: float) -> Optional[float]:
    """Optimality gap: (optimal - solver) / optimal"""
    if optimal_value is None or optimal_value <= 0:
        return None
    return (optimal_value - solver_value) / optimal_value