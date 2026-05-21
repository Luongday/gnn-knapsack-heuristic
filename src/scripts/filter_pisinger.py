"""Filter Pisinger instances by capacity ratio to focus on hard regime."""
import numpy as np
import glob
import shutil
from pathlib import Path
from statistics import mean, stdev


def check_and_filter(src_dir, dst_dir=None, cap_min=0.35, cap_max=0.65):
    src_dir = Path(src_dir)
    files = sorted(glob.glob(str(src_dir / '*.npz')))

    ratios = []
    greedy_ratios = []
    for f in files:
        d = np.load(f)
        total_w = int(sum(d['weights']))
        cap = int(d['capacity'])
        ratios.append(cap / total_w)

    print(f"\n=== Capacity distribution: {src_dir.name} ===")
    print(f"N instances: {len(ratios)}")
    if ratios:
        print(f"Cap/TotalWeight: mean={mean(ratios):.1%}, std={stdev(ratios):.1%}")
        print(f"Min={min(ratios):.1%}, Max={max(ratios):.1%}")

    buckets = [
        ('<20%', 0.0, 0.20),
        ('20-35%', 0.20, 0.35),
        ('35-50%', 0.35, 0.50),
        ('50-65%', 0.50, 0.65),
        ('65-80%', 0.65, 0.80),
        ('>80%', 0.80, 1.01),
    ]
    print("\nPhân bố capacity:")
    for name, lo, hi in buckets:
        count = sum(1 for r in ratios if lo <= r < hi)
        bar = '█' * (count * 30 // max(len(ratios), 1))
        print(f"  {name:>8}: {count:>4}/{len(ratios)} ({count / max(len(ratios), 1) * 100:>5.1f}%) {bar}")

    target = sum(1 for r in ratios if cap_min <= r <= cap_max)
    print(f"\nInstances trong vùng [{cap_min:.0%}, {cap_max:.0%}]: {target}/{len(ratios)}")

    if dst_dir and target > 0:
        dst_dir = Path(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)
        kept = 0
        for f in files:
            d = np.load(f)
            total_w = int(sum(d['weights']))
            cap = int(d['capacity'])
            ratio = cap / total_w
            if cap_min <= ratio <= cap_max:
                shutil.copy(f, dst_dir / f'instance_{kept:04d}.npz')
                kept += 1
        print(f"Đã copy {kept} instances vào: {dst_dir}")
        return kept
    return target


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else 'data/pisinger_t3_v2/sanity/type_03_strongly_correlated'
    dst = sys.argv[2] if len(sys.argv) > 2 else None
    check_and_filter(src, dst)