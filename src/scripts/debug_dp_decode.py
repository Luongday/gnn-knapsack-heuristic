# File: debug_dp_decode.py
import torch
import sys
from pathlib import Path
sys.path.insert(0, '.')

from GNNForKnapSack.src.core.instance_loader import load_instance
from GNNForKnapSack.src.gnn.Knapsack_GNN.Dp import solve_knapsack_dp
from GNNForKnapSack.src.core.decode_utils import greedy_feasible_decode, dp_subset_decode

# Load 1 instance
W, V, C = load_instance('C:/Users/Admin/Python/GNNForKnapSack/data/sanity/test/instance_0000.npz')
n = len(W)
print(f"n={n}, C={C}")

# Ground truth DP
W_list = W.tolist()
V_list = V.tolist()
dp_sol = solve_knapsack_dp(W_list, V_list, int(C))
dp_value = sum(V_list[i] for i, s in enumerate(dp_sol) if s == 1)
print(f"DP full optimal value: {dp_value}")

# Fake probs: uniform (simulating untrained gnn)
fake_probs_uniform = torch.ones(n) * 0.5
x_uniform = dp_subset_decode(fake_probs_uniform, W, V, float(C), top_m=15)
uniform_value = float((x_uniform * V).sum())
print(f"DP subset decode (uniform probs, m=15): {uniform_value}")

# Fake probs: DP optimal probs
fake_probs_optimal = torch.tensor([float(s) for s in dp_sol])
x_optimal = dp_subset_decode(fake_probs_optimal, W, V, float(C), top_m=15)
optimal_value = float((x_optimal * V).sum())
print(f"DP subset decode (optimal probs, m=15): {optimal_value}")

# Greedy decode with uniform probs
W_tensor = torch.tensor(W, dtype=torch.float32)
V_tensor = torch.tensor(V, dtype=torch.float32)

x_greedy = greedy_feasible_decode(fake_probs_uniform, W_tensor, float(C))
greedy_value = float((x_greedy * V_tensor).sum())
print(f"Greedy decode (uniform probs): {greedy_value}")

# Compare with m smaller than n
for m in [3, 5, 10, 15]:
    x = dp_subset_decode(fake_probs_uniform, W, V, float(C), top_m=m)
    val = float((x * V).sum())
    print(f"  m={m}: value={val}")
