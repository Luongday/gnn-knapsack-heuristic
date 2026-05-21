"""S2V-DQN training for 0/1 Knapsack — YAML config support."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from torch_geometric.utils import scatter as pyg_scatter

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
_REPO_PARENT = _HERE.parents[3]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))

from GNNForKnapSack.src.core.instance_loader import load_instance, list_instances
from s2v_env import GraphKnapsackEnv
from s2v_model import S2VQNetwork, save_s2v_checkpoint
from s2v_replay import GraphReplayBuffer, GraphTransition
from GNNForKnapSack.src.core.rl_training_logger import RLTrainingLogger

try:
    from GNNForKnapSack.src.core.config_loader import load_config, ConfigDict, resolve_device
except ImportError:
    print("ERROR: config_loader.py not found.")
    sys.exit(1)


DEFAULTS = {
    "gamma":               0.99,
    "lr":                  3e-4,
    "batch_size":          64,
    "buffer_size":         50_000,
    "min_buffer_size":     1_000,
    "target_update_steps": 500,
    "eps_start":           1.0,
    "eps_end":             0.05,
    "eps_decay_steps":     30_000,
    "grad_clip":           1.0,
    "train_freq":          4,
}

LOG_EVERY = 1_000
VAL_EVERY = 5_000


def mark(msg: str):
    print(f"[S2V-DQN] {msg}", flush=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def epsilon_by_step(step: int, eps_start: float, eps_end: float, decay_steps: int) -> float:
    if step >= decay_steps:
        return eps_end
    t = step / max(1, decay_steps)
    return eps_start + t * (eps_end - eps_start)


def pick_action_eps_greedy(
    q_values:   torch.Tensor,
    valid_mask: np.ndarray,
    eps:        float,
    rng:        np.random.Generator,
) -> int:
    valid_idx = np.where(valid_mask > 0.5)[0]
    if len(valid_idx) == 0:
        return -1

    if rng.random() < eps:
        return int(rng.choice(valid_idx))

    q = q_values.detach().cpu().numpy().copy()
    q[valid_mask < 0.5] = -1e9
    return int(np.argmax(q))


def load_instances_for_training(dataset_dir: Path) -> List[dict]:
    files = list_instances(dataset_dir)
    instances = []
    for path in files:
        W, V, C = load_instance(path)
        npz = np.load(str(path), allow_pickle=True)
        dp_value = float(npz["dp_value"]) if "dp_value" in npz else 0.0
        instances.append({"weights": W, "values": V, "capacity": int(C),
                          "name": path.name, "dp_value": dp_value})
    return instances

def soft_update_target(online, target, tau: float) -> None:
    """Polyak averaging for S2V-DQN target network"""
    with torch.no_grad():
        for param_o, param_t in zip(online.parameters(), target.parameters()):
            param_t.data.mul_(tau).add_(param_o.data, alpha=1 - tau)

@torch.no_grad()
def evaluate_on_set(
    model: S2VQNetwork,
    instances: List[dict],
    device: torch.device,
    k: int = 16,
    limit: int = 50,
    graph_type: str = "knn",
    max_conflict_edges: int | None = None,
) -> dict:
    model.eval()
    values = []
    ratios = []
    feasibles = 0
    for inst in instances[:limit]:
        env = GraphKnapsackEnv(inst["weights"], inst["values"], inst["capacity"],
            k=k,
            graph_type=graph_type,
            max_conflict_edges=max_conflict_edges,
        )
        s = env.reset()
        done = False
        while not done:
            mask = env.valid_actions_mask()
            if mask.sum() == 0:
                break
            s_dev = s.to(device)
            q = model(s_dev)
            a = pick_action_eps_greedy(q, mask, 0.0, np.random.default_rng(0))
            if a < 0:
                break
            out = env.step(a)
            s = out.next_state
            done = out.done
        val = env.compute_solution_value()
        values.append(val)
        if env.compute_solution_weight() <= inst["capacity"] + 1e-6:
            feasibles += 1
        dp_val = inst.get("dp_value", 0.0)
        if dp_val > 0:
            ratios.append(val / dp_val)
    model.train()
    return {
        "avg_value":   float(np.mean(values)) if values else 0.0,
        "avg_ratio":   float(np.mean(ratios)) if ratios else 0.0,
        "feasibility": feasibles / max(len(values), 1),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train S2V-DQN (YAML config or CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file")
    parser.add_argument("--dataset_dir", type=Path)
    parser.add_argument("--val_dir",     type=Path)
    parser.add_argument("--test_dir",    type=Path)
    parser.add_argument("--out_dir",     type=Path)
    parser.add_argument("--device",      type=str,  default=None,
                        help="Device: auto, cpu, cuda (overrides config)")
    parser.add_argument("--hidden_dim",  type=int)
    parser.add_argument("--num_layers",  type=int)
    parser.add_argument("--k",           type=int)
    parser.add_argument("--train_steps", type=int)
    parser.add_argument("--lr",          type=float)
    parser.add_argument("--batch_size",  type=int)
    parser.add_argument("--seed",        type=int)
    parser.add_argument("--graph_type",  type=str, choices=["knn", "conflict_static", "conflict_dynamic", "random", "full"])
    parser.add_argument("--max_conflict_edges", type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Xây dựng cấu hình ---
    if args.config:
        config = load_config(args.config)
        cli_dict = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
        for key, value in cli_dict.items():
            if key in config.rl:
                config.rl[key] = value
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
    else:
        # Fallback
        config = ConfigDict({
            "dataset": {
                "train_dir": str(args.dataset_dir) if args.dataset_dir else "data/cross_scale/train_small",
                "val_dir": str(args.val_dir) if args.val_dir else None,
                "test_dir": str(args.test_dir) if args.test_dir else None,
            },
            "model": {
                "hidden_dim": args.hidden_dim if args.hidden_dim else 128,
                "num_layers": args.num_layers if args.num_layers else 3,
            },
            "rl": {
                "gamma": DEFAULTS["gamma"],
                "lr": args.lr if args.lr else DEFAULTS["lr"],
                "batch_size": args.batch_size if args.batch_size else DEFAULTS["batch_size"],
                "buffer_size": DEFAULTS["buffer_size"],
                "min_buffer_size": DEFAULTS["min_buffer_size"],
                "target_update_steps": DEFAULTS["target_update_steps"],
                "eps_start": DEFAULTS["eps_start"],
                "eps_end": DEFAULTS["eps_end"],
                "eps_decay_steps": DEFAULTS["eps_decay_steps"],
                "grad_clip": DEFAULTS["grad_clip"],
                "train_freq": DEFAULTS["train_freq"],
                "train_steps": args.train_steps if args.train_steps else 50_000,
            },
            "graph": {
                "type": args.graph_type if args.graph_type else "knn",
                "k": args.k if args.k else 16,
                "max_conflict_edges": args.max_conflict_edges,
            },
            "seed": args.seed if args.seed else 42,
            "paths": {
                "out_dir": str(args.out_dir) if args.out_dir else "results/S2V_DQN",
            },
        })

    dataset_cfg = config.dataset
    model_cfg   = config.model
    rl_cfg      = config.rl
    graph_cfg   = config.graph
    seed        = config.seed
    out_dir     = Path(config.paths.out_dir)
    target_tau = getattr(rl_cfg, 'target_tau', 0.0)

    set_seed(seed)
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_str = args.device if args.device is not None else config.get("device", "auto")
    device = resolve_device(device_str)
    mark(f"Device: {device}")

    mark(f"Loading train: {dataset_cfg.train_dir}")
    train_set = load_instances_for_training(Path(dataset_cfg.train_dir))

    val_set, test_set = [], []
    if dataset_cfg.val_dir:
        val_set = load_instances_for_training(Path(dataset_cfg.val_dir))
    if dataset_cfg.test_dir:
        test_set = load_instances_for_training(Path(dataset_cfg.test_dir))

    mark(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

    online = S2VQNetwork(
        in_dim=7,
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        dropout=0.1,
    ).to(device)

    target = S2VQNetwork(
        in_dim=7,
        hidden_dim=model_cfg.hidden_dim,
        num_layers=model_cfg.num_layers,
        dropout=0.1,
    ).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(online.parameters(), lr=rl_cfg.lr)
    buffer    = GraphReplayBuffer(rl_cfg.buffer_size, seed=seed)

    n_params = sum(p.numel() for p in online.parameters())
    mark(f"S2VQNetwork: hidden={model_cfg.hidden_dim} layers={model_cfg.num_layers} params={n_params:,} graph_type={graph_cfg.type}")

    logger = RLTrainingLogger(out_dir / f"training_{graph_cfg.type}_log.csv")

    step_count = 0
    updates = 0
    best_val_ratio = -float("inf")
    last_loss = None
    latest_val_value = None
    latest_val_ratio = None
    latest_val_feas = None
    start_time = time.perf_counter()

    while step_count < rl_cfg.train_steps:
        inst = train_set[int(rng.integers(0, len(train_set)))]
        env = GraphKnapsackEnv(inst["weights"], inst["values"], inst["capacity"],
                                k=graph_cfg.k, graph_type=graph_cfg.type,
                                max_conflict_edges=graph_cfg.max_conflict_edges)
        s = env.reset()
        done = False

        while not done and step_count < rl_cfg.train_steps:
            mask = env.valid_actions_mask()
            if mask.sum() == 0:
                break

            eps = epsilon_by_step(step_count, rl_cfg.eps_start, rl_cfg.eps_end, rl_cfg.eps_decay_steps)

            with torch.no_grad():
                s_dev = s.to(device)
                q = online(s_dev)

            action = pick_action_eps_greedy(q, mask, eps, rng)
            if action < 0:
                break

            out = env.step(action)
            s2 = out.next_state
            mask2 = env.valid_actions_mask()

            buffer.push(GraphTransition(
                s=s, a=int(action), r=float(out.reward),
                s2=s2, done=bool(out.done), valid_mask=mask2,
            ))

            s = s2
            done = out.done
            step_count += 1

            do_update = (
                len(buffer) >= rl_cfg.min_buffer_size
                and step_count % rl_cfg.train_freq == 0
            )

            if do_update:
                s_b, a_b, r_b, s2_b, d_b, mask_list = buffer.sample(rl_cfg.batch_size)

                s_b  = s_b.to(device)
                s2_b = s2_b.to(device)
                a_b  = a_b.to(device)
                r_b  = r_b.to(device)
                d_b  = d_b.to(device)

                q_all = online(s_b)
                ptr = s_b.ptr
                global_a = ptr[:-1] + a_b
                q_sa = q_all[global_a]

                with torch.no_grad():
                    q2_all = target(s2_b)
                    mask_full_np = np.concatenate(mask_list).astype(np.float32)
                    mask_full = torch.from_numpy(mask_full_np).to(device)
                    q2_masked = q2_all.masked_fill(mask_full < 0.5, float("-inf"))
                    max_q2 = pyg_scatter(
                        q2_masked, s2_b.batch,
                        dim=0, dim_size=rl_cfg.batch_size,
                        reduce="max",
                    )
                    max_q2 = torch.where(torch.isfinite(max_q2), max_q2, torch.zeros_like(max_q2))
                    y = r_b + (1.0 - d_b) * rl_cfg.gamma * max_q2

                loss = F.smooth_l1_loss(q_sa, y)
                last_loss = float(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), rl_cfg.grad_clip)
                optimizer.step()
                updates += 1

                if target_tau > 0.0:
                    soft_update_target(online, target, target_tau)
                elif updates % rl_cfg.target_update_steps == 0:
                    target.load_state_dict(online.state_dict())

            if step_count % VAL_EVERY == 0 and val_set:
                val_metrics = evaluate_on_set(online, val_set, device,
                                              k=graph_cfg.k, limit=50,
                                              graph_type=graph_cfg.type,
                                              max_conflict_edges=graph_cfg.max_conflict_edges)
                latest_val_value = val_metrics["avg_value"]
                latest_val_ratio = val_metrics["avg_ratio"]
                latest_val_feas  = val_metrics["feasibility"]
                mark(f"  [val] Step={step_count} | avg_value={latest_val_value:.2f} "
                     f"ratio={latest_val_ratio:.4f} feas={latest_val_feas:.3f}")
                if latest_val_ratio > best_val_ratio:
                    best_val_ratio = latest_val_ratio
                    save_s2v_checkpoint(online, out_dir / f"s2v_dqn_{graph_cfg.type}_best.pt")
                    mark(f"  [val] new best ratio={best_val_ratio:.4f} -> s2v_dqn_best.pt")

            if step_count % LOG_EVERY == 0 or step_count == 1:
                elapsed = time.perf_counter() - start_time
                loss_str = f"{last_loss:.4f}" if last_loss is not None else "N/A"
                mark(f"Step={step_count:>6} | eps={eps:.3f} | buffer={len(buffer):>5} "
                     f"loss={loss_str} | updates={updates:>5} | elapsed={elapsed/60:.1f} m")

                logger.log(
                    step=step_count,
                    updates=updates,
                    epsilon=eps,
                    loss=last_loss,
                    avg_value_val=latest_val_value,
                    avg_ratio_val=latest_val_ratio,
                    best_ratio_val=best_val_ratio if best_val_ratio > -float("inf") else None,
                    val_feasibility=latest_val_feas,
                    buffer_size=len(buffer),
                    elapsed_sec=elapsed,
                )

    train_time = time.perf_counter() - start_time
    save_s2v_checkpoint(online, out_dir / f"s2v_dqn_{graph_cfg.type}.pt")

    # Final evaluation on test set using best checkpoint
    test_ratio, test_feas = None, None
    best_ckpt = out_dir / f"s2v_dqn_{graph_cfg.type}_best.pt"
    if test_set and best_ckpt.exists():
        from s2v_model import S2VQNetwork as _S2V
        best_model = _S2V(in_dim=7, hidden_dim=model_cfg.hidden_dim,
                          num_layers=model_cfg.num_layers, dropout=0.0).to(device)
        best_model.load_state_dict(torch.load(best_ckpt, map_location=device)["model_state_dict"])
        test_metrics = evaluate_on_set(best_model, test_set, device,
                                       k=graph_cfg.k, limit=len(test_set),
                                       graph_type=graph_cfg.type,
                                       max_conflict_edges=graph_cfg.max_conflict_edges)
        test_ratio = test_metrics["avg_ratio"]
        test_feas  = test_metrics["feasibility"]
        mark(f"  [test] avg_ratio={test_ratio:.4f} | feasibility={test_feas:.3f}")

    meta = {
        "train_time_sec":  round(train_time, 2),
        "train_steps":     rl_cfg.train_steps,
        "updates":         updates,
        "n_train":         len(train_set),
        "n_val":           len(val_set),
        "n_test":          len(test_set),
        "seed":            seed,
        "hidden_dim":      model_cfg.hidden_dim,
        "num_layers":      model_cfg.num_layers,
        "best_val_ratio":  round(best_val_ratio, 4) if best_val_ratio > -float("inf") else None,
        "test_ratio":      round(test_ratio, 4) if test_ratio is not None else None,
        "test_feasibility": round(test_feas, 4) if test_feas is not None else None,
        "params":          n_params,
        "graph_type":      graph_cfg.type,
    }
    with (out_dir / "train_meta.json").open("w") as f:
        json.dump(meta, f, indent=2)

    mark(f"Training complete: {train_time:.1f}s, {updates} updates")
    if best_val_ratio > -float("inf"):
        mark(f"Best val ratio:  {best_val_ratio:.4f}")
    if test_ratio is not None:
        mark(f"Test ratio:      {test_ratio:.4f} | feasibility={test_feas:.3f}")
    mark(f"Model -> {out_dir / f's2v_dqn_{graph_cfg.type}.pt'}")
    mark(f"Log   -> {out_dir / f'training_{graph_cfg.type}_log.csv'}")


if __name__ == "__main__":
    main()