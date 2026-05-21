import csv
from statistics import mean, stdev
g = {r['instance_file']: float(r['total_value']) for r in csv.DictReader(open('../../results/pisinger_hard/greedy.csv'))}
d = {r['instance_file']: float(r['total_value']) for r in csv.DictReader(open('../../results/pisinger_hard/dp.csv'))}
common = set(g) & set(d)
ratios = [g[k]/d[k] for k in common if d[k] > 0]
print(f"N={len(ratios)}, Greedy ratio: {mean(ratios):.4f} ± {stdev(ratios):.4f}")
print(f"Min={min(ratios):.4f}, Max={max(ratios):.4f}")
print(f"< 0.98: {sum(1 for r in ratios if r < 0.98)}")
print(f"< 0.95: {sum(1 for r in ratios if r < 0.95)}")