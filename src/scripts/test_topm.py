# test_topm.py
import csv
from pathlib import Path

for test in ['small', 'n100', 'n200']:
    dp = {r['instance_file']: int(len(eval(r['selected_items'])))
          for r in csv.DictReader(open(f'results/cross_eval/dp_{test}.csv'))}
    gnn_greedy = {r['instance_file']: int(len(eval(r['selected_items'])))
                  for r in csv.DictReader(open(f'results/cross_eval/knn_on_{test}_greedy.csv'))}

    if dp and gnn_greedy:
        common = set(dp) & set(gnn_greedy)
        avg_dp = sum(dp[k] for k in common) / len(common)
        avg_gnn = sum(gnn_greedy[k] for k in common) / len(common)
        print(f"{test}: DP chọn avg {avg_dp:.1f} items, GNN-greedy chọn avg {avg_gnn:.1f} items")