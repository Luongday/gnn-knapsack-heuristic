"""Configuration loader from YAML files with dot-notation access."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def resolve_device(device_str: str = "auto") -> "torch.device":
    """Resolve device string to torch.device.

    Values:
        "auto"  → CUDA if available, else CPU
        "cuda"  → force CUDA (raises if not available)
        "cpu"   → force CPU
    """
    import torch
    s = (device_str or "auto").lower().strip()
    if s == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(s)


class ConfigDict(dict):
    """Dictionary that supports dot-notation access."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def get(self, key, default=None):
        return super().get(key, default)


def _dict_to_config(obj: Any) -> Any:
    """Recursively convert dict to ConfigDict."""
    if isinstance(obj, dict):
        return ConfigDict({k: _dict_to_config(v) for k, v in obj.items()})
    elif isinstance(obj, list):
        return [_dict_to_config(item) for item in obj]
    else:
        return obj


def load_config(config_path: str | Path) -> ConfigDict:
    """Load YAML configuration file and return as ConfigDict."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return _dict_to_config(data)


def merge_configs(base_config: ConfigDict, override_config: ConfigDict) -> ConfigDict:
    """Merge two configs, override values from base with override."""
    result = ConfigDict(base_config.copy())
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


def save_config(config: ConfigDict, save_path: str | Path) -> None:
    """Save configuration back to YAML file."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        yaml.dump(dict(config), f, default_flow_style=False, sort_keys=False)


DECODE_DEFAULTS = {
    "strategy":   "greedy_prob",
    "top_m":      50,
    "beam_width": 5,
}

DECODE_STRATEGIES = ("greedy_prob", "greedy_ratio", "dp_subset",
                     "beam_search", "diversified_beam", "sample")


def get_decode_cfg(config: ConfigDict | dict | None) -> ConfigDict:
    """Read `decode` section from a config with backward-compatible defaults.

    Old configs without a `decode` block still work — defaults preserve the
    pre-refactor behavior (greedy_prob).
    """
    raw = {}
    if config is not None:
        section = config.get("decode") if hasattr(config, "get") else None
        if isinstance(section, dict):
            raw = dict(section)
    merged = {**DECODE_DEFAULTS, **raw}
    if merged["strategy"] not in DECODE_STRATEGIES:
        raise ValueError(
            f"Unknown decode.strategy={merged['strategy']!r}. "
            f"Allowed: {DECODE_STRATEGIES}"
        )
    return ConfigDict(merged)


def build_config_from_cli(args: argparse.Namespace, default_data_dir: Path, default_save_path: Path) -> ConfigDict:
    """Build ConfigDict from CLI arguments (backward compatibility)."""
    return ConfigDict({
        "dataset": {
            "source": getattr(args, "dataset_source", "generated"),
            "generated_dir": getattr(args, "generated_dir", str(default_data_dir)),
            "val_dir": getattr(args, "val_dir", None),
            "test_dir": getattr(args, "test_dir", None),
            "k": getattr(args, "k", 16),
            "train_ratio": getattr(args, "train_ratio", 0.8),
            "val_ratio": getattr(args, "val_ratio", 0.1),
        },
        "model": {
            "hidden_dim": getattr(args, "hidden_dim", 256),
            "num_layers": getattr(args, "num_layers", 4),
            "dropout": getattr(args, "dropout", 0.15),
            "conv_type": getattr(args, "conv_type", "gin"),
            "use_global_ctx": not getattr(args, "no_global_ctx", False),
        },
        "training": {
            "epochs": getattr(args, "epochs", 150),
            "batch_size": getattr(args, "batch_size", 16),
            "lr": getattr(args, "lr", 5e-4),
            "warmup_epochs": getattr(args, "warmup_epochs", 5),
            "early_stop_wait": getattr(args, "early_stop_wait", 60),
            "seed": getattr(args, "seed", 2025),
        },
        "graph": {
            "type": getattr(args, "graph_type", "knn"),
            "max_conflict_edges": getattr(args, "max_conflict_edges", None),
        },
        "paths": {
            "save_path": getattr(args, "save_path", str(default_save_path)),
            "log_dir": str(Path(getattr(args, "save_path", default_save_path)).parent),
        },
        "decode": {
            "strategy":   getattr(args, "decode_strategy", DECODE_DEFAULTS["strategy"]),
            "top_m":      getattr(args, "top_m",           DECODE_DEFAULTS["top_m"]),
            "beam_width": getattr(args, "beam_width",      DECODE_DEFAULTS["beam_width"]),
        },
    })
