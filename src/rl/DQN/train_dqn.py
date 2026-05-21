"""Huấn luyện DQN cho bài toán 0/1 Knapsack — hỗ trợ YAML config."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# Add repo parent so 'GNNForKnapSack' package is importable
_REPO_PARENT = _HERE.parents[3]
if str(_REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(_REPO_PARENT))


from dqn_env import KnapsackEnv
from dqn_model import QNetwork
from dqn_replay import ReplayBuffer, Transition
from GNNForKnapSack.src.core.instance_loader import load_instance, list_instances
from GNNForKnapSack.src.core.rl_training_logger import RLTrainingLogger

# Import config loader
try:
    from GNNForKnapSack.src.core.config_loader import load_config, ConfigDict, resolve_device
except ImportError:
    print("ERROR: config_loader.py not found.")
    sys.exit(1)

CHECKPOINT_EVERY = 10_000
LOG_EVERY        = 1_000
VAL_EVERY        = 5_000


def mark(msg: str):
    print(f"[DQN-TRAIN] {msg}", flush=True)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def epsilon_by_step(step: int, cfg: dict) -> float:
    if step >= cfg.eps_decay_steps:
        return cfg.eps_end
    t = step / cfg.eps_decay_steps
    return cfg.eps_start + t * (cfg.eps_end - cfg.eps_start)


def pick_action(q_values: np.ndarray, valid_mask: np.ndarray,
                eps: float, rng: np.random.Generator) -> int:
    if rng.random() < eps:
        valid_actions = np.where(valid_mask > 0.5)[0]
        return int(rng.choice(valid_actions))
    q = q_values.copy()
    q[valid_mask < 0.5] = -1e9
    return int(np.argmax(q))


def load_instances_for_training(dataset_dir: Path):
    files = list_instances(dataset_dir)
    instances = []
    for path in files:
        W, V, C = load_instance(path)
        W = np.asarray(W, dtype=np.float32)
        V = np.asarray(V, dtype=np.float32)
        order = np.argsort(-(V / (W + 1e-8)))
        W, V = W[order], V[order]
        npz = np.load(str(path), allow_pickle=True)
        dp_value = float(npz["dp_value"]) if "dp_value" in npz else 0.0
        instances.append({
            "weights": W,
            "values": V,
            "capacity": int(C),
            "name": path.name,
            "dp_value": dp_value,
        })
    return instances

def split_instances(instances: list, seed: int, train_ratio: float, val_ratio: float):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(instances))
    rng.shuffle(idx)

    n = len(instances)
    n_train = max(1, int(round(n * train_ratio)))
    n_val   = max(1, int(round(n * val_ratio)))
    n_train = min(n_train, n - 2) if n >= 3 else n_train

    train = [instances[i] for i in idx[:n_train]]
    val   = [instances[i] for i in idx[n_train:n_train + n_val]]
    test  = [instances[i] for i in idx[n_train + n_val:]]
    return train, val, test

def soft_update_target(online: QNetwork, target: QNetwork, tau: float) -> None:
    """Polyak averaging: target = tau * target + (1 - tau) * online"""
    with torch.no_grad():
        for param_o, param_t in zip(online.parameters(), target.parameters()):
            param_t.data.mul_(tau).add_(param_o.data, alpha=1 - tau)


@torch.no_grad()
def evaluate_dqn_on_set(
    model: QNetwork,
    instances: list,
    device: torch.device,
    eps: float = 1e-8,
    limit: int = 50,
) -> dict:
    model.eval()
    values = []
    ratios = []
    feasibles = 0
    for inst in instances[:limit]:
        env = KnapsackEnv(inst["weights"], inst["values"],
                          inst["capacity"], eps=eps)
        s = env.reset()
        done = False
        while not done:
            mask = env.valid_actions_mask()
            q = model(torch.from_numpy(s).unsqueeze(0).to(device)).cpu().numpy()[0]
            q[mask < 0.5] = -1e9
            a = int(np.argmax(q))
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
        "n":           len(values),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Huấn luyện DQN cho 0/1 Knapsack (YAML config hoặc CLI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML configuration file")
    # Các tham số ghi đè
    parser.add_argument("--dataset_dir", type=Path)
    parser.add_argument("--val_dir", type=Path)
    parser.add_argument("--test_dir", type=Path)
    parser.add_argument("--out_dir", type=Path)
    parser.add_argument("--device", type=str, default=None,
                        help="Device: auto, cpu, cuda (overrides config)")
    parser.add_argument("--hidden_dim", type=int)
    parser.add_argument("--train_steps", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    # --- Xây dựng cấu hình ---
    if args.config:
        config = load_config(args.config)
        # Ghi đè bằng CLI
        cli_dict = {k: v for k, v in vars(args).items() if v is not None and k != "config"}
        for key, value in cli_dict.items():
            if key in config.rl:
                config.rl[key] = value
            elif key in config.model:
                config.model[key] = value
            elif key in config.paths:
                config.paths[key] = value
            elif key in config.dataset:
                config.dataset[key] = value
            elif key == "seed":
                config.seed = value
    else:
        # Fallback: xây dựng config từ DQNConfig và CLI
        from dqn_config import DQNConfig
        cfg = DQNConfig()
        config = ConfigDict({
            "dataset": {
                "train_dir": str(args.dataset_dir) if args.dataset_dir else "data/cross_scale/train_small",
                "val_dir": str(args.val_dir) if args.val_dir else None,
                "test_dir": str(args.test_dir) if args.test_dir else None,
                "train_ratio": cfg.train_ratio,
                "val_ratio": cfg.val_ratio,
            },
            "model": {
                "hidden_dim": args.hidden_dim if args.hidden_dim else 128,
            },
            "rl": {
                "gamma": cfg.gamma,
                "lr": args.lr if args.lr else cfg.lr,
                "batch_size": cfg.batch_size,
                "buffer_size": cfg.buffer_size,
                "min_buffer_size": cfg.min_buffer_size,
                "target_update_steps": cfg.target_update_steps,
                "eps_start": cfg.eps_start,
                "eps_end": cfg.eps_end,
                "eps_decay_steps": args.train_steps * 0.6 if args.train_steps else cfg.eps_decay_steps,
                "grad_clip_norm": cfg.grad_clip_norm,
                "train_steps": args.train_steps if args.train_steps else cfg.train_steps,
                "warmup_steps": cfg.warmup_steps,
            },
            "seed": args.seed if args.seed else cfg.seed,
            "paths": {
                "out_dir": str(args.out_dir) if args.out_dir else "results/DQN",
            },
        })

    # Trích xuất cấu hình
    dataset_cfg = config.dataset
    model_cfg   = config.model
    rl_cfg      = config.rl
    seed        = config.seed
    out_dir     = Path(config.paths.out_dir)
    target_tau = getattr(rl_cfg, 'target_tau', 0.0)

    set_seed(seed)
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    device_str = args.device if args.device is not None else config.get("device", "auto")
    device = resolve_device(device_str)
    mark(f"Device: {device}")

    # Load data
    mark(f"Loading instances from {dataset_cfg.train_dir}")
    all_train = load_instances_for_training(Path(dataset_cfg.train_dir))

    if dataset_cfg.val_dir and dataset_cfg.test_dir:
        train_set = all_train
        val_set   = load_instances_for_training(Path(dataset_cfg.val_dir))
        test_set  = load_instances_for_training(Path(dataset_cfg.test_dir))
        mark(f"Dataset: 3 SEPARATE dirs")
        mark(f"  Train: {dataset_cfg.train_dir} ({len(train_set)})")
        mark(f"  Val:   {dataset_cfg.val_dir} ({len(val_set)})")
        mark(f"  Test:  {dataset_cfg.test_dir} ({len(test_set)})")
    elif dataset_cfg.test_dir:
        n_tv = len(all_train)
        n_train = max(1, int(n_tv * 0.9))
        train_set = all_train[:n_train]
        val_set   = all_train[n_train:]
        test_set  = load_instances_for_training(Path(dataset_cfg.test_dir))
        mark(f"  Train+Val: {dataset_cfg.train_dir} ({n_tv} -> train={len(train_set)}, val={len(val_set)})")
        mark(f"  Test: {dataset_cfg.test_dir} ({len(test_set)} instances)")
    else:
        train_set, val_set, test_set = split_instances(
            all_train, seed, dataset_cfg.train_ratio, dataset_cfg.val_ratio
        )
        mark(f"Loaded {len(all_train)} -> train={len(train_set)} val={len(val_set)} test={len(test_set)}")

    # Init env
    sample_env = KnapsackEnv(train_set[0]["weights"],
                       train_set[0]["values"],
                       train_set[0]["capacity"], eps=1e-8)
    state_dim = sample_env.reset().shape[0]
    mark(f"State dimension: {state_dim}")

    online_net = QNetwork(state_dim, hidden_dim=model_cfg.hidden_dim).to(device)
    target_net = QNetwork(state_dim, hidden_dim=model_cfg.hidden_dim).to(device)
    target_net.load_state_dict(online_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(online_net.parameters(), lr=rl_cfg.lr)
    replay_buffer = ReplayBuffer(rl_cfg.buffer_size, seed=seed)

    params = sum(p.numel() for p in online_net.parameters())
    mark(f"QNetwork: state_dim={state_dim} hidden={model_cfg.hidden_dim} params={params:,}")
    mark(f"Training: {rl_cfg.train_steps} steps, lr={rl_cfg.lr}")

    logger = RLTrainingLogger(out_dir / f"training_dqn_log.csv")

    step = 0
    updates = 0
    best_val_ratio = -float("inf")
    last_loss = None
    latest_val_value = None
    latest_val_ratio = None
    latest_val_feas  = None
    start_time = time.perf_counter()

    while step < rl_cfg.train_steps:
        instance = random.choice(train_set)
        env = KnapsackEnv(instance["weights"], instance["values"],
                          instance["capacity"], eps=1e-8)
        state = env.reset()
        done = False

        while not done and step < rl_cfg.train_steps:
            valid_mask = env.valid_actions_mask()
            eps = epsilon_by_step(step, rl_cfg)

            with torch.no_grad():
                q_values = online_net(torch.from_numpy(state).unsqueeze(0).to(device))
                q_values = q_values.cpu().numpy()[0]

            action = pick_action(q_values, valid_mask, eps, rng)

            step_output = env.step(action)
            next_state = step_output.next_state
            reward = step_output.reward
            done = step_output.done
            next_mask = env.valid_actions_mask() if not done else np.array([1, 0], dtype=np.int64)

            replay_buffer.push(Transition(s=state, a=int(action), r=float(reward),
                                   s2=next_state, done=bool(done), mask2=next_mask))
            state = next_state
            step += 1

            if len(replay_buffer) >= rl_cfg.min_buffer_size:
                s_batch, a_batch, r_batch, s2_batch, done_batch, mask2_b = replay_buffer.sample(rl_cfg.batch_size)

                s_t  = torch.from_numpy(s_batch).to(device)
                a_t  = torch.from_numpy(a_batch).to(device)
                r_t  = torch.from_numpy(r_batch).to(device)
                s2_t = torch.from_numpy(s2_batch).to(device)
                done_t  = torch.from_numpy(done_batch).to(device)
                mask2_t = torch.from_numpy(mask2_b).to(device)

                current_q = online_net(s_t).gather(1, a_t.view(-1, 1)).squeeze(1)

                with torch.no_grad():
                    next_q = target_net(s2_t)
                    next_q = next_q + (mask2_t - 1.0) * 1e9
                    max_next_q = next_q.max(dim=1).values
                    target_q = r_t + (1.0 - done_t) * rl_cfg.gamma * max_next_q

                loss = F.smooth_l1_loss(current_q, target_q)
                last_loss = float(loss.item())

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online_net.parameters(), rl_cfg.grad_clip_norm)
                optimizer.step()
                updates += 1

                if target_tau > 0.0:
                    soft_update_target(online_net, target_net, target_tau)
                elif step % rl_cfg.target_update_steps == 0:
                    target_net.load_state_dict(online_net.state_dict())

            if step % VAL_EVERY == 0 and val_set:
                val_metrics = evaluate_dqn_on_set(online_net, val_set, device, limit=50)
                latest_val_value = val_metrics["avg_value"]
                latest_val_ratio = val_metrics["avg_ratio"]
                latest_val_feas  = val_metrics["feasibility"]
                mark(f"  [val] step={step} avg_value={latest_val_value:.2f} "
                     f"ratio={latest_val_ratio:.4f} feas={latest_val_feas:.3f}")
                if latest_val_ratio > best_val_ratio:
                    best_val_ratio = latest_val_ratio
                    torch.save({
                        "state_dim": state_dim,
                        "hidden_dim": model_cfg.hidden_dim,
                        "model_state": online_net.state_dict(),
                        "config": dict(config),
                    }, out_dir / "dqn_best.pt")
                    mark(f"  [val] new best ratio={best_val_ratio:.4f} -> dqn_best.pt")

            if step % LOG_EVERY == 0 or step == 1:
                elapsed = time.perf_counter() - start_time
                loss_str = f"{last_loss:.4f}" if last_loss is not None else "N/A"
                mark(f"Step={step:6d} | Eps={eps:.3f} "
                     f"Buffer={len(replay_buffer):6d} | Loss={loss_str} "
                     f"Updates {updates:5d} | Elapsed={elapsed/60:.1f} m")

                logger.log(
                    step=step,
                    updates=updates,
                    epsilon=eps,
                    loss=last_loss,
                    avg_value_val=latest_val_value,
                    avg_ratio_val=latest_val_ratio,
                    best_ratio_val=best_val_ratio if best_val_ratio > -float("inf") else None,
                    val_feasibility=latest_val_feas,
                    buffer_size=len(replay_buffer),
                    elapsed_sec=elapsed,
                )

    total_time = time.perf_counter() - start_time
    mark(f"Training complete! Time: {total_time / 60:.1f} min")

    final_path = out_dir / "dqn.pt"
    torch.save({
        "state_dim": state_dim,
        "hidden_dim": model_cfg.hidden_dim,
        "model_state": online_net.state_dict(),
        "config": dict(config),
        "train_steps": step,
    }, final_path)

    # Final evaluation on test set using best checkpoint
    test_ratio, test_feas = None, None
    best_ckpt = out_dir / "dqn_best.pt"
    if test_set and best_ckpt.exists():
        best_ckpt_data = torch.load(best_ckpt, map_location=device, weights_only=False)
        best_net = QNetwork(state_dim, hidden_dim=model_cfg.hidden_dim).to(device)
        best_net.load_state_dict(best_ckpt_data["model_state"])
        test_metrics = evaluate_dqn_on_set(best_net, test_set, device,
                                           limit=len(test_set))
        test_ratio = test_metrics["avg_ratio"]
        test_feas  = test_metrics["feasibility"]
        mark(f"  [test] avg_ratio={test_ratio:.4f} | feasibility={test_feas:.3f}")

    meta_path = out_dir / "train_meta.json"
    with meta_path.open("w") as f:
        json.dump({
            "train_time_sec":  round(total_time, 2),
            "train_steps":     rl_cfg.train_steps,
            "updates":         updates,
            "seed":            seed,
            "dataset_dir":     dataset_cfg.train_dir,
            "n_train":         len(train_set),
            "n_val":           len(val_set),
            "n_test":          len(test_set),
            "state_dim":       state_dim,
            "hidden_dim":      model_cfg.hidden_dim,
            "best_val_ratio":  round(best_val_ratio, 4) if best_val_ratio > -float("inf") else None,
            "test_ratio":      round(test_ratio, 4) if test_ratio is not None else None,
            "test_feasibility": round(test_feas, 4) if test_feas is not None else None,
        }, f, indent=2)

    mark(f"Model saved: {final_path}")
    if best_val_ratio > -float("inf"):
        mark(f"Best val ratio:  {best_val_ratio:.4f}")
    if test_ratio is not None:
        mark(f"Test ratio:      {test_ratio:.4f} | feasibility={test_feas:.3f}")


if __name__ == "__main__":
    main()