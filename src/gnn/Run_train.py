"""Training entry point for KnapsackGNN — supports YAML config and CLI override."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

_HERE     = Path(__file__).resolve().parent
_GNN_ROOT = _HERE.parent
for _p in [str(_GNN_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from GNNForKnapSack.src.core.config_loader import (
        load_config, build_config_from_cli, ConfigDict, resolve_device,
        get_decode_cfg, DECODE_STRATEGIES,
    )
except ImportError:
    print("ERROR: config_loader.py not found. Please create it or adjust PYTHONPATH.")
    sys.exit(1)

from GNNForKnapSack.src.gnn.Knapsack_GNN.dataset import (
    GeneratedKnapsack01Dataset,
    KnapsackDataset,
    split_dataset_by_instances,
)
from GNNForKnapSack.src.gnn.Knapsack_GNN.model import (
    KnapsackGNN, save_checkpoint, load_checkpoint,
)
from GNNForKnapSack.src.gnn.Knapsack_GNN.Train_eval import (
    train_one_epoch, evaluate_node_accuracy,
)
from GNNForKnapSack.src.core.decode_utils import (
    decode as _decode,
    greedy_ratio_decode,
)


# ---------------------------------------------------------------------------
# Greedy baseline
# ---------------------------------------------------------------------------

@torch.no_grad()
def precompute_greedy_baseline(loader: DataLoader, label: str = "val") -> Dict:
    greedy_values:   List[float] = []
    dp_values:       List[float] = []
    greedy_ratios:   List[float] = []
    greedy_feasible: int = 0
    total:           int = 0

    for batch in loader:
        batch_vec = batch.batch if hasattr(batch, "batch") else \
                    torch.zeros(batch.num_nodes, dtype=torch.long)
        n_graphs  = int(batch_vec.max().item()) + 1

        for g in range(n_graphs):
            mask  = batch_vec == g
            w_g   = batch.wts[mask].cpu()
            v_g   = batch.vals[mask].cpu()
            cap_g = float(batch.cap[g].item() if batch.cap.dim() > 0
                          else batch.cap.item())
            dp_sol = batch.y[mask].view(-1).cpu()

            dp_val = float((dp_sol * v_g).sum())

            gr_sel = greedy_ratio_decode(v_g, w_g, cap_g)
            gr_val = float((gr_sel * v_g).sum())
            gr_w   = float((gr_sel * w_g).sum())

            greedy_values.append(gr_val)
            dp_values.append(dp_val)
            greedy_feasible += int(gr_w <= cap_g + 1e-6)

            ratio = gr_val / dp_val if dp_val > 0 else 0.0
            greedy_ratios.append(ratio)
            total += 1

    avg_ratio = float(np.mean(greedy_ratios)) if greedy_ratios else 0.0
    std_ratio = float(np.std(greedy_ratios))  if greedy_ratios else 0.0
    optimal_count = sum(1 for r in greedy_ratios if abs(r - 1.0) < 1e-6)

    print(f"  [{label}] Greedy baseline: ratio={avg_ratio:.4f} +/- {std_ratio:.4f} | "
          f"feasible={greedy_feasible}/{total} | "
          f"optimal={optimal_count}/{total}")

    return {
        "greedy_values":   greedy_values,
        "dp_values":       dp_values,
        "greedy_ratios":   greedy_ratios,
        "avg_ratio":       avg_ratio,
        "std_ratio":       std_ratio,
        "feasible_rate":   greedy_feasible / max(total, 1),
        "optimal_count":   optimal_count,
        "n_instances":     total,
    }


@torch.no_grad()
def evaluate_with_greedy(
    model:           nn.Module,
    loader:          DataLoader,
    device:          torch.device,
    greedy_baseline: Dict,
    decode_strategy: str = "greedy_prob",
    top_m:           int = 50,
    beam_width:      int = 5,
) -> Dict:
    model.eval()

    gnn_values:       List[float] = []
    gnn_ratios:       List[float] = []
    gnn_feasible:     int = 0
    gnn_beats_greedy: int = 0
    gnn_ties_greedy:  int = 0
    instance_idx = 0

    for batch in loader:
        batch = batch.to(device)
        batch_vec = batch.batch if hasattr(batch, "batch") else \
                    torch.zeros(batch.num_nodes, dtype=torch.long, device=device)
        n_graphs  = int(batch_vec.max().item()) + 1

        logits = model(batch)
        probs  = torch.sigmoid(logits)

        for g in range(n_graphs):
            mask  = batch_vec == g
            p_g   = probs[mask].detach().cpu()
            w_g   = batch.wts[mask].detach().cpu()
            v_g   = batch.vals[mask].detach().cpu()
            cap_g = float(batch.cap[g].item() if batch.cap.dim() > 0
                          else batch.cap.item())

            dp_sol = batch.y[mask].view(-1).detach().cpu()
            dp_val = float((dp_sol * v_g).sum())

            gnn_sel = _decode(p_g, w_g, float(cap_g),
                              strategy=decode_strategy,
                              values=v_g,
                              beam_width=beam_width,
                              top_m=top_m)
            gnn_val = float((gnn_sel * v_g).sum())
            gnn_w   = float((gnn_sel * w_g).sum())

            gnn_values.append(gnn_val)
            gnn_feasible += int(gnn_w <= cap_g + 1e-6)

            ratio = gnn_val / dp_val if dp_val > 0 else 0.0
            gnn_ratios.append(ratio)

            if instance_idx < len(greedy_baseline["greedy_values"]):
                gr_val = greedy_baseline["greedy_values"][instance_idx]
                if gnn_val > gr_val + 1e-6:
                    gnn_beats_greedy += 1
                elif abs(gnn_val - gr_val) < 1e-6:
                    gnn_ties_greedy += 1

            instance_idx += 1

    n = max(instance_idx, 1)
    gnn_avg_ratio    = float(np.mean(gnn_ratios)) if gnn_ratios else 0.0
    gnn_std_ratio    = float(np.std(gnn_ratios))  if gnn_ratios else 0.0
    greedy_avg_ratio = greedy_baseline["avg_ratio"]
    gnn_loses_greedy = n - gnn_beats_greedy - gnn_ties_greedy

    return {
        "gnn_avg_ratio":      gnn_avg_ratio,
        "gnn_std_ratio":      gnn_std_ratio,
        "greedy_avg_ratio":   greedy_avg_ratio,
        "gnn_feasible_rate":  gnn_feasible / n,
        "gnn_beats_greedy":   gnn_beats_greedy,
        "gnn_ties_greedy":    gnn_ties_greedy,
        "gnn_loses_greedy":   gnn_loses_greedy,
        "beats_rate":         gnn_beats_greedy / n,
        "n_instances":        n,
        "advantage":          gnn_avg_ratio - greedy_avg_ratio,
    }


class TrainingLogger:
    HEADER = [
        "epoch", "train_loss", "val_acc",
        "gnn_ratio", "gnn_std", "greedy_ratio",
        "gnn_beats_greedy", "gnn_loses_greedy", "ties",
        "advantage", "lr", "time_sec",
    ]

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path
        self.rows: List[dict] = []
        if log_path:
            log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: dict):
        self.rows.append(row)
        if self.log_path:
            with self.log_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.HEADER)
                writer.writeheader()
                writer.writerows(self.rows)


def get_warmup_lr(base_lr: float, epoch: int, warmup_epochs: int) -> float:
    if epoch >= warmup_epochs:
        return base_lr
    return base_lr * (0.1 + 0.9 * epoch / max(warmup_epochs, 1))


def _default_data_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "cross_scale" / "train"

def _default_save_path() -> Path:
    return Path(__file__).resolve().parents[2] / "results" / "GNN" / "gnn.pt"


# ---------------------------------------------------------------------------
# CLI Parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train KnapsackGNN (YAML config or CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file")

    parser.add_argument("--dataset_source", choices=["excel", "generated"])
    parser.add_argument("--generated_dir", type=str)
    parser.add_argument("--val_dir", type=str)
    parser.add_argument("--test_dir", type=str)
    parser.add_argument("--k", type=int)
    parser.add_argument("--train_ratio", type=float)
    parser.add_argument("--val_ratio", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--warmup_epochs", type=int)
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--num_layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--conv_type", choices=["gin", "sage", "hybrid"])
    parser.add_argument("--no_global_ctx", action="store_true")
    parser.add_argument("--graph_type", choices=["knn", "conflict_static", "random", "full"])
    parser.add_argument("--max_conflict_edges", type=int)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--early_stop_wait", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str, default=None,
                        help="Device: auto, cpu, cuda (overrides config)")
    parser.add_argument("--decode_strategy", type=str,
                        choices=list(DECODE_STRATEGIES),
                        help="Override decode.strategy from config")
    parser.add_argument("--top_m", type=int,
                        help="Override decode.top_m (dp_subset)")
    parser.add_argument("--beam_width", type=int,
                        help="Override decode.beam_width (beam_search)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = parse_args()

    # --- Xây dựng cấu hình ---
    if args.config:
        # Đọc từ file YAML
        config = load_config(args.config)
        # Ghi đè bằng CLI arguments (nếu được cung cấp)
        cli_dict = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
        for key, value in cli_dict.items():
            # Xác định section chứa key (dựa trên tên key)
            section = None
            if key in ["dataset_source", "generated_dir", "val_dir", "test_dir", "k", "train_ratio", "val_ratio"]:
                section = "dataset"
            elif key in ["hidden_dim", "num_layers", "dropout", "conv_type", "use_global_ctx", "no_global_ctx"]:
                section = "model"
                if key == "no_global_ctx":
                    config.model.use_global_ctx = not value
                    continue
            elif key in ["epochs", "batch_size", "lr", "warmup_epochs", "early_stop_wait", "seed"]:
                section = "training"
            elif key in ["graph_type", "max_conflict_edges"]:
                section = "graph"
            elif key in ["save_path"]:
                section = "paths"
            elif key in ["decode_strategy", "top_m", "beam_width"]:
                section = "decode"
                if "decode" not in config or not isinstance(config.get("decode"), dict):
                    config["decode"] = ConfigDict({})
                cfg_key = "strategy" if key == "decode_strategy" else key
                config["decode"][cfg_key] = value
                continue
            if section is not None:
                config[section][key] = value
    else:
        # Fallback: xây dựng config từ CLI arguments (tương thích ngược)
        config = build_config_from_cli(args, _default_data_dir(), _default_save_path())

    # Trích xuất các phần cấu hình
    dataset_cfg = config.dataset
    model_cfg   = config.model
    train_cfg   = config.training
    graph_cfg   = config.graph
    paths       = config.paths
    decode_cfg  = get_decode_cfg(config)

    device_str = args.device if args.device is not None else config.get("device", "auto")
    device = resolve_device(device_str)
    print(f"Device: {device}")

    torch.manual_seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)

    excel_path = str(_GNN_ROOT / "data" / "data.xlsx")

    # --- Dataset loading---
    if dataset_cfg.source == "excel":
        dataset = KnapsackDataset(
            excel_path=excel_path,
            k=dataset_cfg.k,
            graph_type=graph_cfg.type,
            max_conflict_edges=graph_cfg.max_conflict_edges,
        )
        train_set, val_set, test_set = split_dataset_by_instances(
            dataset, train_ratio=dataset_cfg.train_ratio, val_ratio=dataset_cfg.val_ratio,
        )
        print(f"Dataset: excel  total={len(dataset)} "
              f"train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")
    else:
        from torch.utils.data import Subset

        has_val_dir  = dataset_cfg.val_dir is not None
        has_test_dir = dataset_cfg.test_dir is not None

        if has_val_dir and has_test_dir:
            train_dataset = GeneratedKnapsack01Dataset.get_lazy(
                root_dir=dataset_cfg.generated_dir, k=dataset_cfg.k,
                graph_type=graph_cfg.type, max_conflict_edges=graph_cfg.max_conflict_edges
            )
            val_dataset   = GeneratedKnapsack01Dataset.get_lazy(
                root_dir=dataset_cfg.val_dir, k=dataset_cfg.k,
                graph_type=graph_cfg.type, max_conflict_edges=graph_cfg.max_conflict_edges
            )
            test_dataset  = GeneratedKnapsack01Dataset.get_lazy(
                root_dir=dataset_cfg.test_dir, k=dataset_cfg.k,
                graph_type=graph_cfg.type, max_conflict_edges=graph_cfg.max_conflict_edges
            )
            train_set = train_dataset
            val_set   = val_dataset
            test_set  = test_dataset
            dataset   = train_dataset
            print(f"Dataset: 3 SEPARATE dirs")
            print(f"  Train: {dataset_cfg.generated_dir} ({len(train_set)})")
            print(f"  Val:   {dataset_cfg.val_dir} ({len(val_set)})")
            print(f"  Test:  {dataset_cfg.test_dir} ({len(test_set)})")
        elif has_test_dir:
            train_val_dataset = GeneratedKnapsack01Dataset.get_lazy(
                root_dir=dataset_cfg.generated_dir, k=dataset_cfg.k,
                graph_type=graph_cfg.type, max_conflict_edges=graph_cfg.max_conflict_edges
            )
            test_dataset      = GeneratedKnapsack01Dataset.get_lazy(
                root_dir=dataset_cfg.test_dir, k=dataset_cfg.k,
                graph_type=graph_cfg.type, max_conflict_edges=graph_cfg.max_conflict_edges
            )
            n_tv    = len(train_val_dataset)
            n_train = max(1, int(n_tv * 0.9))
            train_set = Subset(train_val_dataset, list(range(n_train)))
            val_set   = Subset(train_val_dataset, list(range(n_train, n_tv)))
            test_set  = test_dataset
            dataset   = train_val_dataset
            print(f"Dataset: train+val from {dataset_cfg.generated_dir} "
                  f"({n_tv} -> train={len(train_set)}, val={len(val_set)})")
            print(f"  Test:      {dataset_cfg.test_dir} ({len(test_set)} instances)")
        else:
            dataset = GeneratedKnapsack01Dataset.get_lazy(
                root_dir=dataset_cfg.generated_dir, k=dataset_cfg.k,
                graph_type=graph_cfg.type, max_conflict_edges=graph_cfg.max_conflict_edges
            )
            train_set, val_set, test_set = split_dataset_by_instances(
                dataset, train_ratio=dataset_cfg.train_ratio, val_ratio=dataset_cfg.val_ratio,
            )
            print(f"Dataset: generated  total={len(dataset)} "
                  f"train={len(train_set)}  val={len(val_set)}  test={len(test_set)}")

    train_loader = DataLoader(train_set, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=train_cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=train_cfg.batch_size, shuffle=False)

    # --- Greedy baseline ---
    print("\nPre-computing Greedy baselines...")
    val_greedy  = precompute_greedy_baseline(val_loader,  "val")
    test_greedy = precompute_greedy_baseline(test_loader, "test")

    greedy_target = val_greedy["avg_ratio"]
    print(f"\n  Target to beat: Greedy val ratio = {greedy_target:.4f}")
    print(f"  Greedy optimal count: {val_greedy['optimal_count']}/{val_greedy['n_instances']}")

    # --- Model ---
    in_dim = dataset[0].num_node_features if len(dataset) > 0 else 7
    model  = KnapsackGNN(
        in_dim=in_dim,
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        dropout=model_cfg.dropout,
        use_global_ctx=model_cfg.use_global_ctx,
        conv_type=model_cfg.conv_type,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=train_cfg.lr, weight_decay=1e-5,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=25,
        T_mult=2,
        eta_min=train_cfg.lr * 0.01,
    )

    criterion = nn.BCEWithLogitsLoss()

    params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: indim={in_dim} hiddendim={model_cfg.hidden_dim} layers={model_cfg.num_layers} "
          f"conv={model_cfg.conv_type} ctx={model_cfg.use_global_ctx} params={params:,}")
    print(f"Training: epochs={train_cfg.epochs} batch={train_cfg.batch_size} lr={train_cfg.lr} "
          f"warmup={train_cfg.warmup_epochs}")
    print(f"Decode:   strategy={decode_cfg.strategy} top_m={decode_cfg.top_m} "
          f"beam_width={decode_cfg.beam_width}")

    log_path = Path(paths.log_dir) / f"gnn_{graph_cfg.type}_training_log.csv"
    logger = TrainingLogger(log_path)

    best_ratio  = 0.0
    best_epoch  = 0
    wait        = 0

    print(f"\n{'Epoch':>5} | {'Loss':>7} {'Acc':>6} | "
          f"{'gnn':>7} {'±':>6} | {'Greedy':>7} | "
          f"{'Beats':>5} {'Loses':>5} {'Ties':>4} | "
          f"{'Advantage':>7} | {'LR':>8} | {'Time':>5}")
    print("-" * 105)

    train_start = perf_counter()
    for epoch in range(1, train_cfg.epochs + 1):
        t0 = perf_counter()

        if epoch <= train_cfg.warmup_epochs:
            lr_epoch = get_warmup_lr(train_cfg.lr, epoch, train_cfg.warmup_epochs)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_epoch

        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device,
        )

        val_acc     = evaluate_node_accuracy(model, val_loader, device)
        val_metrics = evaluate_with_greedy(
            model, val_loader, device, val_greedy,
            decode_strategy=decode_cfg.strategy,
            top_m=decode_cfg.top_m,
            beam_width=decode_cfg.beam_width,
        )

        elapsed = perf_counter() - t0

        gnn_r  = val_metrics["gnn_avg_ratio"]
        gnn_s  = val_metrics["gnn_std_ratio"]
        gr_r   = val_metrics["greedy_avg_ratio"]
        beats  = val_metrics["gnn_beats_greedy"]
        loses  = val_metrics["gnn_loses_greedy"]
        ties_  = val_metrics["gnn_ties_greedy"]
        adv    = val_metrics["advantage"]
        cur_lr = optimizer.param_groups[0]["lr"]

        adv_str = f"{adv:>+7.4f}" if adv != 0 else "  0.000"
        beat_marker = " *" if adv > 0 else ""

        print(f"{epoch:>5} | {train_loss:>7.4f} {val_acc:>6.3f} | "
              f"{gnn_r:>7.4f} {gnn_s:>5.3f} | {gr_r:>7.4f} | "
              f"{beats:>5} {loses:>5} {ties_:>4} | "
              f"{adv_str} | {cur_lr:>.2e} | {elapsed:>5.1f}s{beat_marker}")

        logger.log({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_acc": round(val_acc, 4),
            "gnn_ratio": round(gnn_r, 4), "gnn_std": round(gnn_s, 4),
            "greedy_ratio": round(gr_r, 4),
            "gnn_beats_greedy": beats, "gnn_loses_greedy": loses, "ties": ties_,
            "advantage": round(adv, 4),
            "lr": round(cur_lr, 6),
            "time_sec": round(elapsed, 1),
        })

        if gnn_r > best_ratio:
            best_ratio = gnn_r
            best_epoch = epoch
            wait = 0
            best_path = Path(paths.save_path).parent / f"gnn_{graph_cfg.type}_best.pt"
            save_checkpoint(model, best_path)
        else:
            wait += 1

        if epoch > train_cfg.warmup_epochs:
            scheduler.step()

        if wait >= train_cfg.early_stop_wait:
            print(f"\nEarly stopping at epoch {epoch}. "
                  f"Best: ratio={best_ratio:.4f} @ epoch {best_epoch}")
            break

    save_checkpoint(model, paths.save_path)
    total_time = perf_counter() - train_start

    print(f"\n{'='*90}")
    print(f"TRAINING COMPLETE — {total_time:.1f}s")
    print(f"{'='*90}")
    print(f"  Best gnn ratio: {best_ratio:.4f} @ epoch {best_epoch}")
    print(f"  Greedy baseline: {val_greedy['avg_ratio']:.4f}")
    print(f"  Advantage: {best_ratio - val_greedy['avg_ratio']:+.4f}")
    if best_ratio > val_greedy["avg_ratio"]:
        print(f"  >>> gnn BEATS GREEDY on validation! <<<")
    else:
        gap = val_greedy["avg_ratio"] - best_ratio
        print(f"  gnn still behind Greedy by {gap:.4f}")

    print(f"\n{'='*90}")
    print("TEST SET EVALUATION (best model)")
    print(f"{'='*90}")

    best_model_path = Path(paths.save_path).parent / f"gnn_{graph_cfg.type}_best.pt"
    if best_model_path.exists():
        best_model = load_checkpoint(best_model_path, device=device, dropout=0.0)
        test_metrics = evaluate_with_greedy(
            best_model, test_loader, device, test_greedy,
            decode_strategy=decode_cfg.strategy,
            top_m=decode_cfg.top_m,
            beam_width=decode_cfg.beam_width,
        )

        print(f"\n  {'Solver':<10} {'Ratio':>8} {'Feasible':>10} {'Beats':>6}")
        print(f"  {'-'*40}")
        print(f"  {'DP':<10} {'1.0000':>8} {'100%':>10} {'—':>6}")
        print(f"  {'Greedy':<10} {test_greedy['avg_ratio']:>8.4f} "
              f"{test_greedy['feasible_rate']*100:>9.1f}% {'—':>6}")
        print(f"  {'gnn':<10} {test_metrics['gnn_avg_ratio']:>8.4f} "
              f"{test_metrics['gnn_feasible_rate']*100:>9.1f}% "
              f"{test_metrics['gnn_beats_greedy']:>6}")

        n_test = test_metrics["n_instances"]
        print(f"\n  gnn vs Greedy on test ({n_test} instances):")
        print(f"    gnn wins:    {test_metrics['gnn_beats_greedy']:>4} "
              f"({test_metrics['beats_rate']*100:.1f}%)")
        print(f"    Greedy wins: {test_metrics['gnn_loses_greedy']:>4} "
              f"({test_metrics['gnn_loses_greedy']/max(n_test,1)*100:.1f}%)")
        print(f"    Ties:        {test_metrics['gnn_ties_greedy']:>4}")
        print(f"    Advantage:   {test_metrics['advantage']:+.4f}")


if __name__ == "__main__":
    main()