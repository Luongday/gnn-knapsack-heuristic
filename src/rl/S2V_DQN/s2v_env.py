"""S2V-DQN graph environment for 0/1 Knapsack.
   Now supports conflict graph (static/dynamic) via graph_type parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch
from torch_geometric.data import Data

from GNNForKnapSack.src.gnn.Knapsack_GNN.Graph_builder import (
    _build_knn_edges,
    _build_conflict_edges_static,
    _build_random_edges,
    _build_full_edges,
)

S2V_NODE_DIM = 7


@dataclass
class GraphStepOutput:
    next_state: Data
    reward: float
    done: bool
    info: Dict[str, Any]


class GraphKnapsackEnv:
    """Graph-based knapsack environment for S2V-DQN."""

    def __init__(
        self,
        weights: np.ndarray,
        values:  np.ndarray,
        capacity: int,
        k:       int = 16,
        eps:     float = 1e-8,
        graph_type: str = "knn",
        max_conflict_edges: int | None = None,
    ):
        self.w     = np.asarray(weights).astype(np.float32)
        self.v     = np.asarray(values).astype(np.float32)
        self.W     = float(capacity)
        self.k     = k
        self.eps   = eps
        self.n     = int(self.w.shape[0])
        self.graph_type = graph_type
        self.max_conflict_edges = max_conflict_edges

        # Normalization constants
        self.w_max = float(self.w.max()) if self.n > 0 else 1.0
        self.v_max = float(self.v.max()) if self.n > 0 else 1.0
        self.r_max = self.v_max / (self.w_max + self.eps)

        # Cache static features
        w_norm_np = self.w / (self.w_max + self.eps)
        v_norm_np = self.v / (self.v_max + self.eps)
        ratios_np = (self.v / (self.w + self.eps)) / (self.r_max + self.eps)
        self._static_x = torch.from_numpy(
            np.stack([w_norm_np, v_norm_np, ratios_np], axis=1).astype(np.float32)
        )

        self._wts_tensor  = torch.from_numpy(self.w.copy())
        self._vals_tensor = torch.from_numpy(self.v.copy())
        self._cap_tensor  = torch.tensor([self.W], dtype=torch.float32)

        # Build edges based on graph_type
        if self.graph_type == "knn":
            if self.n > 1:
                self.edge_index = _build_knn_edges(
                    self._static_x, k=min(self.k, self.n - 1)
                )
            else:
                self.edge_index = torch.zeros((2, 0), dtype=torch.long)
        elif self.graph_type == "conflict_static":
            self.edge_index = _build_conflict_edges_static(
                self._wts_tensor, float(capacity), max_conflict_edges
            )
        elif self.graph_type == "conflict_dynamic":
            # Will be computed per step
            self.edge_index = None
        elif self.graph_type == "random":
            self.edge_index = _build_random_edges(self.n, k=min(self.k, max(self.n - 1, 1)))
        elif self.graph_type == "full":
            self.edge_index = _build_full_edges(self.n)
        else:
            raise ValueError(f"Unknown graph_type: {self.graph_type}")

        self.reset()

    def reset(self) -> Data:
        self.cap          = float(self.W)
        self.total_value  = 0.0
        self.total_weight = 0.0
        self.selection    = np.zeros(self.n, dtype=np.int64)
        self.steps_taken  = 0
        return self._build_graph_state()

    def _build_graph_state(self) -> Data:
        """Construct PyG Data graph for current state."""
        selected_flag_np = self.selection.astype(np.float32)
        feasible_np      = (self.w <= self.cap + self.eps).astype(np.float32)
        feasible_flag_np = feasible_np * (1.0 - selected_flag_np)

        cap_norm = float(self.cap / (self.W + self.eps))
        frac     = float(self.selection.sum() / self.n) if self.n > 0 else 0.0

        dynamic_np = np.stack([
            selected_flag_np,
            feasible_flag_np,
            np.full(self.n, cap_norm, dtype=np.float32),
            np.full(self.n, frac,     dtype=np.float32),
        ], axis=1)
        dynamic_x = torch.from_numpy(dynamic_np)
        x = torch.cat([self._static_x, dynamic_x], dim=1)

        # Determine edge_index
        if self.graph_type == "knn":
            edge_index = self.edge_index
        elif self.graph_type == "conflict_static":
            edge_index = self.edge_index
        elif self.graph_type == "conflict_dynamic":
            edge_index = _build_conflict_edges_static(
                self._wts_tensor, self.cap, self.max_conflict_edges
            )
        elif self.graph_type in ("random", "full"):
            edge_index = self.edge_index
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(
            x=x,
            edge_index=edge_index,
            wts=self._wts_tensor,
            vals=self._vals_tensor,
            cap=self._cap_tensor,
        )

    def valid_actions_mask(self) -> np.ndarray:
        not_selected = (self.selection == 0)
        fits         = (self.w <= self.cap + self.eps)
        return (not_selected & fits).astype(np.int64)

    def step(self, action: int) -> GraphStepOutput:
        mask = self.valid_actions_mask()
        if mask.sum() == 0:
            return GraphStepOutput(
                self._build_graph_state(), 0.0, True,
                {"terminal": True, "reason": "no_valid_actions"}
            )

        if action < 0 or action >= self.n or mask[action] == 0:
            return GraphStepOutput(
                self._build_graph_state(), 0.0, True,
                {"terminal": True, "reason": "invalid_action"}
            )

        w_i = float(self.w[action])
        v_i = float(self.v[action])
        self.cap          -= w_i
        self.total_weight += w_i
        self.total_value  += v_i
        self.selection[action] = 1
        self.steps_taken  += 1

        reward = v_i / self.v_max
        new_mask = self.valid_actions_mask()
        done = (new_mask.sum() == 0)

        return GraphStepOutput(
            self._build_graph_state(), reward, done,
            {
                "selected": action,
                "cap_remaining": self.cap,
                "total_value": self.total_value,
                "total_weight": self.total_weight,
                "n_selected": int(self.selection.sum()),
            }
        )

    def compute_solution_value(self) -> float:
        return float(self.total_value)

    def compute_solution_weight(self) -> float:
        return float(self.total_weight)

    def get_selection(self) -> np.ndarray:
        return self.selection.copy()