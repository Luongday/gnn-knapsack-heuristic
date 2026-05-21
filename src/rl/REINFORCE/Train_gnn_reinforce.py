"""gnn + REINFORCE training — YAML config with full ablation support.
   Supports: no baseline, hard update (copy), Polyak EMA baseline.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch_geometric.loader import DataLoader

_HERE     = Path(__file__).resolve().parent
_GNN_ROOT = _HERE.parent
_REPO_PARENT = _HERE.parents[3]
for _p in [str(_REPO_PARENT), str(_GNN_ROOT), str(_HERE)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from GNNForKnapSack.src.gnn.Knapsack_GNN.dataset import GeneratedKnapsack01Dataset
from GNNForKnapSack.src.gnn.Knapsack_GNN.model import KnapsackGNN, save_checkpoint, load_checkpoint

try:
    from GNNForKnapSack.src.core.config_loader import load_config, ConfigDict, resolve_device
except ImportError:
    print("ERROR: config_loader.py not found.")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Helpers for reproducibility
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.benchmark = True


# ---------------------------------------------------------------------------
# Sampling and repair (giữ nguyên)
# ---------------------------------------------------------------------------
def sample_and_repair(
    logits:   torch.Tensor,
    weights:  torch.Tensor,
    values:   torch.Tensor,
    capacity: float,
    greedy:   bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    probs = torch.sigmoid(logits)

    if greedy:
        sampled = (probs > 0.5).float()
    else:
        dist = torch.distributions.Bernoulli(probs=probs)
        sampled = dist.sample()

    selected = sampled.detach().cpu().clone()
    w_cpu = weights.detach().cpu()
    v_cpu = values.detach().cpu()
    total_w = float((selected * w_cpu).sum())

    if total_w > capacity + 1e-6:
        sel_idx = torch.where(selected > 0.5)[0]
        if len(sel_idx) > 0:
            ratios = v_cpu[sel_idx] / (w_cpu[sel_idx] + 1e-8)
            order  = sel_idx[torch.argsort(ratios)]
            for i in order:
                if total_w <= capacity + 1e-6:
                    break
                selected[i] = 0
                total_w -= float(w_cpu[i])

    action_final = selected.to(logits.device)

    eps = 1e-8
    log_p_per_node = (
        action_final * torch.log(probs + eps)
        + (1 - action_final) * torch.log(1 - probs + eps)
    )
    log_prob = log_p_per_node.mean()

    value = float((selected * v_cpu).sum())
    return selected, log_prob, value


def reinforce_batch_loss(
    model:         KnapsackGNN,
    baseline:      Optional[KnapsackGNN],
    batch,
    device:        torch.device,
    baseline_type: str = "polyak",
    entropy_beta:  float = 0.001,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    batch = batch.to(device)
    logits_all = model(batch)

    # Compute baseline rewards
    with torch.no_grad():
        if baseline_type == "none":
            # No baseline: advantage = reward
            base_logits_all = None
        else:
            base_logits_all = baseline(batch)

    batch_vec = batch.batch
    n_graphs = int(batch_vec.max().item()) + 1

    log_probs: List[torch.Tensor] = []
    rewards:   List[float]         = []
    baselines: List[float]         = []
    entropies: List[torch.Tensor]  = []

    for g in range(n_graphs):
        mask  = batch_vec == g
        lg    = logits_all[mask]
        w_g   = batch.wts[mask]
        v_g   = batch.vals[mask]
        cap_g = float(batch.cap[g].item())

        _, log_p, reward = sample_and_repair(lg, w_g, v_g, cap_g, greedy=False)

        if base_logits_all is not None:
            base_lg = base_logits_all[mask]
            _, _, base_reward = sample_and_repair(base_lg, w_g, v_g, cap_g, greedy=True)
        else:
            base_reward = 0.0

        # entropy
        probs = torch.sigmoid(lg)
        ent = -(probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8)).mean()
        entropies.append(ent)

        log_probs.append(log_p)
        rewards.append(reward)
        baselines.append(base_reward)

    rewards_t   = torch.tensor(rewards,   device=device, dtype=torch.float32)
    baselines_t = torch.tensor(baselines, device=device, dtype=torch.float32)

    if baseline_type == "none":
        advantages = rewards_t
    else:
        advantages = rewards_t - baselines_t

    if advantages.numel() > 1:
        adv_std = advantages.std()
        if adv_std > 1e-6:
            advantages = (advantages - advantages.mean()) / (adv_std + 1e-8)
    else:
        adv_std = torch.tensor(0.0, device=device)

    log_probs_t = torch.stack(log_probs)
    entropy_t = torch.stack(entropies).mean()
    loss = -(advantages * log_probs_t).mean() - entropy_beta * entropy_t

    stats = {
        "avg_reward":   float(rewards_t.mean().item()),
        "avg_baseline": float(baselines_t.mean().item()),
        "adv_std":      float(adv_std.item()),
        "entropy":      float(entropy_t.item()),
    }
    return loss, stats


@torch.no_grad()
def evaluate_policy(
    model:  KnapsackGNN,
    loader: DataLoader,
    device: torch.device,
    greedy: bool = True,
) -> Dict[str, float]:
    model.eval()
    values = []
    ratios = []
    feasibles = 0
    n = 0

    for batch in loader:
        batch = batch.to(device)
        logits_all = model(batch)
        batch_vec = batch.batch
        n_graphs = int(batch_vec.max().item()) + 1

        for g in range(n_graphs):
            mask  = batch_vec == g
            lg    = logits_all[mask]
            w_g   = batch.wts[mask]
            v_g   = batch.vals[mask]
            cap_g = float(batch.cap[g].item())

            action, _, value = sample_and_repair(lg, w_g, v_g, cap_g, greedy=greedy)
            total_w = float((action * w_g.cpu()).sum())

            values.append(value)
            feasibles += int(total_w <= cap_g + 1e-6)
            n += 1

            dp_sol = batch.y[mask].view(-1).cpu()
            dp_val = float((dp_sol * v_g.cpu()).sum())
            if dp_val > 0:
                ratios.append(value / dp_val)

    model.train()
    return {
        "avg_value":      float(np.mean(values)) if values else 0.0,
        "avg_ratio_vs_dp": float(np.mean(ratios)) if ratios else 0.0,
        "feasibility":    feasibles / max(n, 1),
        "n":              n,
    }


# ---------------------------------------------------------------------------
# Baseline updates
# ---------------------------------------------------------------------------
def update_baseline_polyak(model: KnapsackGNN, baseline: KnapsackGNN, tau: float = 0.95) -> None:
    with torch.no_grad():
        for param_base, param_model in zip(baseline.parameters(), model.parameters()):
            param_base.data.mul_(tau).add_(param_model.data, alpha=1 - tau)


def update_baseline_hard(model: KnapsackGNN, baseline: KnapsackGNN) -> None:
    baseline.load_state_dict(model.state_dict())


# ---------------------------------------------------------------------------
# Training logger
# ---------------------------------------------------------------------------
class TrainingLogger:
    HEADER = ["epoch", "train_loss", "avg_reward", "avg_baseline", "adv_std", "entropy",
              "val_avg_value", "val_ratio", "val_feasibility",
              "baseline_updated", "lr", "time_sec"]

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.rows: List[dict] = []
        log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, row: dict) -> None:
        self.rows.append(row)
        with self.log_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.HEADER)
            writer.writeheader()
            writer.writerows(self.rows)


def load_datasets(dataset_cfg, graph_cfg):
    from torch.utils.data import Subset
    from GNNForKnapSack.src.gnn.Knapsack_GNN.dataset import split_dataset_by_instances

    def create_dataset(root_dir, k, graph_type, max_conflict_edges):
        return GeneratedKnapsack01Dataset.get_lazy(
            root_dir=root_dir,
            k=k,
            graph_type=graph_type,
            max_conflict_edges=max_conflict_edges,
        )

    train_dir = dataset_cfg.train_dir
    val_dir   = dataset_cfg.val_dir
    test_dir  = dataset_cfg.test_dir
    k         = dataset_cfg.k
    train_ratio = dataset_cfg.train_ratio
    val_ratio   = dataset_cfg.val_ratio

    if val_dir and test_dir:
        train_ds = create_dataset(train_dir, k, graph_cfg.type, graph_cfg.max_conflict_edges)
        val_ds   = create_dataset(val_dir,   k, graph_cfg.type, graph_cfg.max_conflict_edges)
        test_ds  = create_dataset(test_dir,  k, graph_cfg.type, graph_cfg.max_conflict_edges)
        print(f"Dataset: 3 SEPARATE dirs")
        print(f"  Train: {train_dir} ({len(train_ds)})")
        print(f"  Val:   {val_dir} ({len(val_ds)})")
        print(f"  Test:  {test_dir} ({len(test_ds)})")
        return train_ds, train_ds, val_ds, test_ds
    elif test_dir:
        tv_ds = create_dataset(train_dir, k, graph_cfg.type, graph_cfg.max_conflict_edges)
        te_ds = create_dataset(test_dir, k, graph_cfg.type, graph_cfg.max_conflict_edges)
        n_tv = len(tv_ds)
        n_train = max(1, int(n_tv * 0.9))
        train = Subset(tv_ds, list(range(n_train)))
        val   = Subset(tv_ds, list(range(n_train, n_tv)))
        test  = Subset(te_ds, list(range(len(te_ds))))
        print(f"Dataset: train+val from {train_dir}, test from {test_dir}")
        return tv_ds, train, val, test
    else:
        ds = create_dataset(train_dir, k, graph_cfg.type, graph_cfg.max_conflict_edges)
        train, val, test = split_dataset_by_instances(
            ds, train_ratio=train_ratio, val_ratio=val_ratio,
        )
        print(f"Dataset: single dir total={len(ds)} "
              f"train={len(train)} val={len(val)} test={len(test)}")
        return ds, train, val, test


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train KnapsackGNN with REINFORCE + ablation (none/hard/polyak baseline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--dataset_dir", type=str)
    parser.add_argument("--val_dir",     type=str)
    parser.add_argument("--test_dir",    type=str)
    parser.add_argument("--k",           type=int)
    parser.add_argument("--train_ratio", type=float)
    parser.add_argument("--val_ratio",   type=float)
    parser.add_argument("--out_dir",     type=Path)
    parser.add_argument("--epochs",      type=int)
    parser.add_argument("--batch_size",  type=int)
    parser.add_argument("--lr",          type=float)
    parser.add_argument("--hidden_dim",  type=int)
    parser.add_argument("--num_layers",  type=int)
    parser.add_argument("--dropout",     type=float)
    parser.add_argument("--conv_type",   choices=["gin", "sage", "hybrid"])
    parser.add_argument("--no_global_ctx", action="store_true")
    parser.add_argument("--baseline_update_every", type=int)
    parser.add_argument("--baseline_type", type=str, default=None,
                        choices=["none", "hard", "polyak"],
                        help="none: no baseline, hard: copy on improvement, polyak: EMA")
    parser.add_argument("--baseline_tau", type=float, default=None,
                        help="Polyak averaging factor (only for polyak)")
    parser.add_argument("--hard_update_threshold", type=float, default=None,
                        help="Min improvement to copy baseline in hard mode")
    parser.add_argument("--entropy_beta", type=float, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--pretrained",  type=str)
    parser.add_argument("--graph_type",  choices=["knn", "conflict_static", "random", "full"])
    parser.add_argument("--max_conflict_edges", type=int)
    parser.add_argument("--save_path",   type=str)
    parser.add_argument("--seed",        type=int)
    parser.add_argument("--grad_clip",   type=float)
    parser.add_argument("--device",      type=str, default=None)
    parser.add_argument("--resume",      type=str, default=None,
                        help="Resume training from checkpoint file")
    parser.add_argument("--early_stop_patience", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.config:
        config = load_config(args.config)
        cli_dict = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
        for key, value in cli_dict.items():
            if key in config.training:
                config.training[key] = value
            elif key in config.model:
                config.model[key] = value
            elif key in config.graph:
                config.graph[key] = value
            elif key in config.paths:
                config.paths[key] = value
            elif key in config.dataset:
                config.dataset[key] = value
            elif key == "seed":
                config.seed = value
            elif key in ["baseline_tau", "entropy_beta", "grad_accum", "early_stop_patience",
                         "baseline_type", "hard_update_threshold"]:
                if not hasattr(config.training, key):
                    setattr(config.training, key, value)
    else:
        config = ConfigDict({
            "dataset": {
                "train_dir": args.dataset_dir if args.dataset_dir else "data/cross_scale/train_small",
                "val_dir": args.val_dir,
                "test_dir": args.test_dir,
                "k": args.k if args.k else 16,
                "train_ratio": args.train_ratio if args.train_ratio else 0.8,
                "val_ratio": args.val_ratio if args.val_ratio else 0.1,
            },
            "model": {
                "hidden_dim": args.hidden_dim if args.hidden_dim else 128,
                "num_layers": args.num_layers if args.num_layers else 3,
                "dropout": args.dropout if args.dropout is not None else 0.1,
                "conv_type": args.conv_type if args.conv_type else "gin",
                "use_global_ctx": not args.no_global_ctx,
            },
            "training": {
                "epochs": args.epochs if args.epochs else 50,
                "batch_size": args.batch_size if args.batch_size else 16,
                "lr": args.lr if args.lr else 5e-5,
                "baseline_update_every": args.baseline_update_every if args.baseline_update_every else 1,
                "baseline_type": args.baseline_type or "polyak",
                "baseline_tau": args.baseline_tau if args.baseline_tau is not None else 0.95,
                "hard_update_threshold": args.hard_update_threshold if args.hard_update_threshold is not None else 0.001,
                "entropy_beta": args.entropy_beta if args.entropy_beta is not None else 0.001,
                "grad_accum": args.grad_accum if args.grad_accum is not None else 1,
                "grad_clip": args.grad_clip if args.grad_clip else 1.0,
                "pretrained": args.pretrained,
                "early_stop_patience": args.early_stop_patience if args.early_stop_patience is not None else 10,
            },
            "graph": {
                "type": args.graph_type if args.graph_type else "knn",
                "max_conflict_edges": args.max_conflict_edges,
            },
            "seed": args.seed if args.seed else 2025,
            "paths": {
                "out_dir": str(args.out_dir) if args.out_dir else "results/GNN_REINFORCE/",
                "save_path": args.save_path if args.save_path else "results/GNN_REINFORCE/gnn_reinforce.pt",
            },
        })

    dataset_cfg = config.dataset
    model_cfg   = config.model
    train_cfg   = config.training
    graph_cfg   = config.graph
    seed        = config.seed
    out_dir     = Path(config.paths.out_dir)
    save_path   = Path(config.paths.save_path)

    device_str = args.device if args.device is not None else config.get("device", "auto")
    device = resolve_device(device_str)

    set_seed(seed, deterministic=True)
    print(f"Device: {device} | Seed: {seed} (deterministic)")

    base_ds, train_set, val_set, test_set = load_datasets(dataset_cfg, graph_cfg)

    train_loader = DataLoader(train_set, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=train_cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(test_set,  batch_size=train_cfg.batch_size, shuffle=False)

    in_dim = base_ds[0].num_node_features

    model = KnapsackGNN(
        in_dim=in_dim, hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers, dropout=model_cfg.dropout,
        use_global_ctx=model_cfg.use_global_ctx,
        conv_type=model_cfg.conv_type,
    ).to(device)

    start_epoch = 1
    best_val_ratio = 0.0
    best_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 1) + 1
        best_val_ratio = checkpoint.get('best_val_ratio', 0.0)
        best_epoch = checkpoint.get('best_epoch', 0)
    elif train_cfg.pretrained:
        print(f"Warm starting from {train_cfg.pretrained}")
        pretrained = load_checkpoint(train_cfg.pretrained, device=device, dropout=model_cfg.dropout)
        model.load_state_dict(pretrained.state_dict(), strict=False)

    # Baseline network (still created, may not be used if baseline_type='none')
    baseline = KnapsackGNN(
        in_dim=in_dim, hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers, dropout=model_cfg.dropout,
        use_global_ctx=model_cfg.use_global_ctx,
        conv_type=model_cfg.conv_type,
    ).to(device)
    baseline.load_state_dict(model.state_dict())
    baseline.eval()
    for p in baseline.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )

    params = sum(p.numel() for p in model.parameters())
    print(f"Model: conv={model_cfg.conv_type} hidden={model_cfg.hidden_dim} "
          f"layers={model_cfg.num_layers} params={params:,}")
    print(f"Training: {train_cfg.epochs} epochs, batch={train_cfg.batch_size}, lr={train_cfg.lr}")
    print(f"Baseline: {train_cfg.baseline_type} | update_every={train_cfg.baseline_update_every}")
    if train_cfg.baseline_type == "polyak":
        print(f"  tau={train_cfg.baseline_tau}")
    elif train_cfg.baseline_type == "hard":
        print(f"  threshold={train_cfg.hard_update_threshold}")
    print(f"Entropy beta={train_cfg.entropy_beta}, grad_accum={train_cfg.grad_accum}")
    print(f"Early stop patience={train_cfg.early_stop_patience}")

    log_path = out_dir / f"gnnrl_{graph_cfg.type}_{train_cfg.baseline_type}_log.csv"
    logger = TrainingLogger(log_path)

    train_start = time.perf_counter()
    no_improve_epochs = 0
    best_val_ratio_sofar = best_val_ratio  # for hard baseline trigger

    print(f"\n{'Epoch':>5} | {'Loss':>8} | {'AvgRwd':>8} {'AvgBase':>8} {'AdvStd':>8} {'Entropy':>8} | "
          f"{'Val':>8} {'Ratio':>7} {'Feas':>6} | {'BaseUpdate':>8} | {'time':>6}")
    print("-" * 115)

    for epoch in range(start_epoch, train_cfg.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        epoch_loss   = 0.0
        n_batches    = 0
        epoch_reward = 0.0
        epoch_base   = 0.0
        epoch_adv_std = 0.0
        epoch_entropy = 0.0

        optimizer.zero_grad()
        for i, batch in enumerate(train_loader):
            loss, stats = reinforce_batch_loss(
                model, baseline if train_cfg.baseline_type != "none" else None,
                batch, device,
                baseline_type=train_cfg.baseline_type,
                entropy_beta=train_cfg.entropy_beta
            )
            loss = loss / train_cfg.grad_accum
            loss.backward()

            epoch_loss   += loss.item() * train_cfg.grad_accum
            epoch_reward += stats["avg_reward"]
            epoch_base   += stats["avg_baseline"]
            epoch_adv_std += stats["adv_std"]
            epoch_entropy += stats["entropy"]
            n_batches    += 1

            if (i + 1) % train_cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

        if n_batches % train_cfg.grad_accum != 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_rwd  = epoch_reward / max(n_batches, 1)
        avg_bse  = epoch_base / max(n_batches, 1)
        avg_adv_std = epoch_adv_std / max(n_batches, 1)
        avg_ent  = epoch_entropy / max(n_batches, 1)

        val_metrics = evaluate_policy(model, val_loader, device, greedy=True)

        # --- Update best model and early stopping (before baseline update) ---
        current_ratio = val_metrics["avg_ratio_vs_dp"]
        min_delta = getattr(train_cfg, 'min_delta', getattr(train_cfg, 'early_stop_min_delta', 0.001))
        just_improved = current_ratio > best_val_ratio + min_delta
        if just_improved:
            best_val_ratio = current_ratio
            best_epoch = epoch
            no_improve_epochs = 0
            save_checkpoint(model, save_path)
            # full checkpoint for resume
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_ratio': best_val_ratio,
                'best_epoch': best_epoch,
            }, save_path.with_suffix('.full.pt'))
        else:
            no_improve_epochs += 1

        # --- Baseline update (after we know if this epoch improved) ---
        baseline_updated = False
        if epoch % train_cfg.baseline_update_every == 0:
            if train_cfg.baseline_type == "hard":
                # Hard update: copy model only when this epoch set a new best
                if just_improved and current_ratio > best_val_ratio_sofar + train_cfg.hard_update_threshold - 1e-9:
                    update_baseline_hard(model, baseline)
                    baseline_updated = True
                    best_val_ratio_sofar = best_val_ratio
            elif train_cfg.baseline_type == "polyak":
                update_baseline_polyak(model, baseline, tau=train_cfg.baseline_tau)
                baseline_updated = True
            # "none" does nothing

        # Learning rate scheduling (based on validation ratio)
        scheduler.step(current_ratio)

        elapsed = time.perf_counter() - t0

        updated_str = "YES" if baseline_updated else "no"
        print(f"{epoch:>5} | {avg_loss:>+8.4f} | {avg_rwd:>8.1f} {avg_bse:>8.1f} {avg_adv_std:>8.4f} {avg_ent:>8.4f} | "
              f"{val_metrics['avg_value']:>8.1f} {current_ratio:>7.4f} "
              f"{val_metrics['feasibility']:>6.3f} | {updated_str:>8} | {elapsed:>5.1f}s")

        logger.log({
            "epoch":            epoch,
            "train_loss":       round(avg_loss, 4),
            "avg_reward":       round(avg_rwd, 2),
            "avg_baseline":     round(avg_bse, 2),
            "adv_std":          round(avg_adv_std, 4),
            "entropy":          round(avg_ent, 4),
            "val_avg_value":    round(val_metrics["avg_value"], 2),
            "val_ratio":        round(current_ratio, 4),
            "val_feasibility":  round(val_metrics["feasibility"], 4),
            "baseline_updated": int(baseline_updated),
            "lr":               optimizer.param_groups[0]["lr"],
            "time_sec":         round(elapsed, 1),
        })

        if no_improve_epochs >= train_cfg.early_stop_patience:
            print(f"Early stopping triggered after {epoch} epochs (no improvement for {no_improve_epochs} epochs)")
            break

    total_time = time.perf_counter() - train_start

    print(f"\n{'=' * 115}")
    print(f"TRAINING COMPLETE - {total_time:.1f}s")
    print(f"  Best val ratio: {best_val_ratio:.4f} @ epoch {best_epoch}")

    print(f"\n=== TEST SET EVALUATION ===")
    best_model_path = save_path
    if best_model_path.exists():
        best_model = load_checkpoint(best_model_path, device=device, dropout=0.0)
        test_metrics = evaluate_policy(best_model, test_loader, device, greedy=True)
        print(f"  avg_value:    {test_metrics['avg_value']:.2f}")
        print(f"  avg_ratio:    {test_metrics['avg_ratio_vs_dp']:.4f}")
        print(f"  feasibility:  {test_metrics['feasibility']:.4f}")
        print(f"  n instances:  {test_metrics['n']}")

    print(f"\nModel saved: {save_path}")
    print(f"Log saved:   {log_path}")


if __name__ == "__main__":
    main()