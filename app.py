"""Streamlit demo app for GNN Knapsack Solver Benchmark."""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
for p in [str(ROOT.parent), str(ROOT), str(SRC),
          str(SRC / "solvers" / "DP"),
          str(SRC / "solvers" / "Greedy"),
          str(SRC / "solvers" / "GA"),
          str(SRC / "solvers" / "BB"),
          str(SRC / "gnn"),
          str(SRC / "rl" / "DQN"),
          str(SRC / "rl" / "S2V_DQN"),
          str(SRC / "rl" / "REINFORCE")]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# Solver-key display & sorting helpers (handle multi-variant schema)
# ---------------------------------------------------------------------------
_SOLVER_FAMILY_ORDER = ["dp", "bb", "ga", "greedy", "gnn", "gnn_dp", "gnnDP",
                        "dqn", "s2v", "reinforce"]
_SOLVER_FAMILY_LABEL = {
    "dp": "DP", "bb": "BB", "ga": "GA", "greedy": "Greedy",
    "gnn": "GNN", "gnn_dp": "GNN-DP", "gnnDP": "GNN-DP",
    "dqn": "DQN", "s2v": "S2V-DQN", "reinforce": "REINFORCE",
}

def _split_solver_key(key: str):
    """Return (family, variant_or_None) for a solver key like 's2v_knn'."""
    if key in _SOLVER_FAMILY_LABEL:
        return key, None
    # Two-token families (gnn_dp, gnnDP_*) take precedence over single-token gnn_*.
    for fam in ("gnnDP", "gnn_dp"):
        prefix = fam + "_"
        if key.startswith(prefix):
            return fam, key[len(prefix):]
    for fam in ("gnn", "s2v", "reinforce"):
        prefix = fam + "_"
        if key.startswith(prefix):
            return fam, key[len(prefix):]
    return key, None

def _solver_display_label(key: str) -> str:
    fam, variant = _split_solver_key(key)
    fam_label = _SOLVER_FAMILY_LABEL.get(fam, fam.upper())
    if variant is None or variant == "main":
        return fam_label
    return f"{fam_label} [{variant}]"

def _iter_summary_solvers(summary: dict):
    """Yield (key, dict) entries from summary.json sorted by family order."""
    items = [(k, v) for k, v in summary.items() if isinstance(v, dict)]
    def sort_key(kv):
        fam, variant = _split_solver_key(kv[0])
        try:
            fam_idx = _SOLVER_FAMILY_ORDER.index(fam)
        except ValueError:
            fam_idx = len(_SOLVER_FAMILY_ORDER)
        return (fam_idx, variant or "")
    return sorted(items, key=sort_key)


# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------
@st.cache_resource
def _load_classical_solvers():
    from GNNForKnapSack.src.solvers.DP.dp_baseline_eval          import solve_knapsack_dp
    from GNNForKnapSack.src.solvers.Greedy.greedy_baseline_eval  import solve_knapsack_greedy
    from GNNForKnapSack.src.solvers.GA.ga_baseline_eval          import solve_knapsack_ga
    from GNNForKnapSack.src.solvers.BB.bb_baseline_eval          import solve_knapsack_bb
    return solve_knapsack_dp, solve_knapsack_greedy, solve_knapsack_ga, solve_knapsack_bb

@st.cache_resource
def _load_gnn_model(model_path: str, graph_type: str):
    import torch
    from GNNForKnapSack.src.gnn.Knapsack_GNN.model         import load_checkpoint
    from GNNForKnapSack.src.gnn.Knapsack_GNN.Graph_builder import build_knapsack_graph_inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_checkpoint(Path(model_path), device=device, dropout=0.0)
    model.eval()
    return model, device, build_knapsack_graph_inference

@st.cache_resource
def _load_dqn_model(model_path: str):
    import torch
    from GNNForKnapSack.src.rl.DQN.dqn_env   import KnapsackEnv
    from GNNForKnapSack.src.rl.DQN.dqn_model import QNetwork
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dim  = ckpt.get("state_dim",  14)
    hidden_dim = ckpt.get("hidden_dim", 128)
    model = QNetwork(state_dim, hidden_dim)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, KnapsackEnv

# ---------------------------------------------------------------------------
# Instance generation
# ---------------------------------------------------------------------------
def generate_instance(n_items: int, seed: int, cap_pct: float):
    rng      = np.random.default_rng(seed)
    weights  = rng.integers(1, 1001, size=n_items, dtype=np.int32)
    mults    = rng.uniform(0.8, 1.3, size=n_items)
    values   = np.maximum((mults * weights).astype(np.int32), 1)
    capacity = max(int(cap_pct * weights.sum()), int(weights.max()))
    return weights, values, capacity

# ---------------------------------------------------------------------------
# Run solvers
# ---------------------------------------------------------------------------
def run_dp(weights, values, capacity):
    solve_dp, *_ = _load_classical_solvers()
    t0 = time.perf_counter()
    selected = solve_dp(weights, values, capacity)
    ms = (time.perf_counter() - t0) * 1000
    value  = int(values[selected].sum()) if selected else 0
    weight = int(weights[selected].sum()) if selected else 0
    return {"value": value, "weight": weight, "time_ms": ms,
            "selected": selected, "feasible": weight <= capacity}

def run_greedy(weights, values, capacity):
    _, solve_gr, *_ = _load_classical_solvers()
    t0 = time.perf_counter()
    selected = solve_gr(weights, values, capacity)
    ms = (time.perf_counter() - t0) * 1000
    value  = int(values[selected].sum()) if selected else 0
    weight = int(weights[selected].sum()) if selected else 0
    return {"value": value, "weight": weight, "time_ms": ms,
            "selected": selected, "feasible": weight <= capacity}

def run_ga(weights, values, capacity):
    _, _, solve_ga, _ = _load_classical_solvers()
    t0 = time.perf_counter()
    selected = solve_ga(weights, values, capacity)
    ms = (time.perf_counter() - t0) * 1000
    value  = int(values[selected].sum()) if selected else 0
    weight = int(weights[selected].sum()) if selected else 0
    return {"value": value, "weight": weight, "time_ms": ms,
            "selected": selected, "feasible": weight <= capacity}

def run_bb(weights, values, capacity, timeout: float = 10.0):
    _, _, _, solve_bb = _load_classical_solvers()
    t0 = time.perf_counter()
    selected, optimal, nodes = solve_bb(weights, values, capacity, timeout_sec=timeout)
    ms     = (time.perf_counter() - t0) * 1000
    value  = int(values[selected].sum()) if selected else 0
    weight = int(weights[selected].sum()) if selected else 0
    result = {"value": value, "weight": weight, "time_ms": ms,
              "selected": selected, "feasible": weight <= capacity}
    if not optimal:
        result["note"] = f"timeout — best-so-far ({nodes:,} nodes)"
    return result

def _no_torch():
    return {"value": None, "weight": None, "time_ms": 0,
            "selected": [], "feasible": False,
            "note": "torch chưa cài — chạy: pip install torch torch-geometric"}

# P1: GNN với decode_method (greedy | dp_subset | beam_search) + top_m / beam_width
def run_gnn(weights, values, capacity, model_path: str, graph_type: str,
            decode_method: str = "greedy", top_m: int = 30, beam_width: int = 5):
    try:
        import torch
        from GNNForKnapSack.src.core.decode_utils import decode as _decode
    except ImportError:
        return _no_torch()
    try:
        model, device, build_graph = _load_gnn_model(model_path, graph_type)
        W = torch.tensor(weights, dtype=torch.float32)
        V = torch.tensor(values,  dtype=torch.float32)
        C = float(capacity)
        t0    = time.perf_counter()
        graph = build_graph(weights.tolist(), values.tolist(), int(capacity),
                            graph_type=graph_type, k=16).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(graph)).cpu().squeeze()
        strategy = {"greedy": "greedy_prob", "dp_subset": "dp_subset",
                    "beam_search": "beam_search"}.get(decode_method, "greedy_prob")
        mask    = _decode(probs, W, C, strategy=strategy,
                          values=V, top_m=top_m, beam_width=beam_width)
        ms      = (time.perf_counter() - t0) * 1000
        sel_idx = [i for i, x in enumerate(mask.tolist()) if x == 1]
        value   = int(values[sel_idx].sum()) if sel_idx else 0
        weight  = int(weights[sel_idx].sum()) if sel_idx else 0
        return {"value": value, "weight": weight, "time_ms": ms,
                "selected": sel_idx, "feasible": weight <= capacity,
                "decode": decode_method, "top_m": top_m if decode_method == "dp_subset" else None}
    except Exception as e:
        return {"value": None, "weight": None, "time_ms": 0,
                "selected": [], "feasible": False, "note": str(e)}

def run_dqn(weights, values, capacity, model_path: str):
    try:
        import torch
        import numpy as _np
    except ImportError:
        return _no_torch()
    try:
        model, KnapsackEnv = _load_dqn_model(model_path)
        env   = KnapsackEnv(weights, values, capacity)
        state = env.reset()
        t0    = time.perf_counter()
        done  = False
        while not done:
            mask = env.valid_actions_mask()
            with torch.no_grad():
                q = model(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).cpu().numpy()[0]
            q[mask < 0.5] = -1e9
            action = int(_np.argmax(q))
            out = env.step(action)
            state, done = out.next_state, out.done
        ms      = (time.perf_counter() - t0) * 1000
        sel_idx = list(np.where(env.selection > 0)[0]) if hasattr(env, "selection") else []
        value   = int(values[sel_idx].sum()) if sel_idx else 0
        weight  = int(weights[sel_idx].sum()) if sel_idx else 0
        return {"value": value, "weight": weight, "time_ms": ms,
                "selected": sel_idx, "feasible": weight <= capacity}
    except Exception as e:
        return {"value": None, "weight": None, "time_ms": 0,
                "selected": [], "feasible": False, "note": str(e)}

@st.cache_resource
def _load_s2v_model(model_path: str):
    import torch
    import sys as _sys
    _s2v_dir = str(Path(__file__).resolve().parent / "src" / "rl" / "S2V_DQN")
    if _s2v_dir not in _sys.path:
        _sys.path.insert(0, _s2v_dir)
    from GNNForKnapSack.src.rl.S2V_DQN.s2v_model import load_s2v_checkpoint
    from GNNForKnapSack.src.rl.S2V_DQN.s2v_env   import GraphKnapsackEnv
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_s2v_checkpoint(Path(model_path), device=device)
    model.eval()
    return model, device, GraphKnapsackEnv

def run_s2v_dqn(weights, values, capacity, model_path: str,
                graph_type: str = "knn", k: int = 16):
    try:
        import torch
        import numpy as _np
    except ImportError:
        return _no_torch()
    try:
        model, device, GraphKnapsackEnv = _load_s2v_model(model_path)
        env  = GraphKnapsackEnv(weights, values, capacity,
                                k=min(k, max(len(weights) - 1, 1)),
                                graph_type=graph_type)
        s    = env.reset()
        t0   = time.perf_counter()
        done = False
        while not done:
            mask = env.valid_actions_mask()
            if mask.sum() == 0:
                break
            with torch.no_grad():
                q = model(s.to(device)).cpu().numpy()
            q[mask < 0.5] = -1e9
            action = int(_np.argmax(q))
            out  = env.step(action)
            s    = out.next_state
            done = out.done
        ms      = (time.perf_counter() - t0) * 1000
        sel_idx = [i for i, v in enumerate(env.get_selection().tolist()) if v == 1]
        value   = int(values[sel_idx].sum()) if sel_idx else 0
        weight  = int(weights[sel_idx].sum()) if sel_idx else 0
        return {"value": value, "weight": weight, "time_ms": ms,
                "selected": sel_idx, "feasible": weight <= capacity}
    except Exception as e:
        return {"value": None, "weight": None, "time_ms": 0,
                "selected": [], "feasible": False, "note": str(e)}

def run_reinforce(weights, values, capacity, model_path: str,
                  graph_type: str = "knn", decode_method: str = "greedy",
                  top_m: int = 30, beam_width: int = 5):
    try:
        import torch
        from GNNForKnapSack.src.core.decode_utils import decode as _decode
    except ImportError:
        return _no_torch()
    try:
        model, device, build_graph = _load_gnn_model(model_path, graph_type)
        W = torch.tensor(weights, dtype=torch.float32)
        V = torch.tensor(values,  dtype=torch.float32)
        C = float(capacity)
        t0    = time.perf_counter()
        graph = build_graph(weights.tolist(), values.tolist(), int(capacity),
                            graph_type=graph_type, k=min(16, max(len(weights) - 1, 1))).to(device)
        with torch.no_grad():
            probs = torch.sigmoid(model(graph)).cpu().squeeze()
        strategy = {"greedy": "greedy_prob", "dp_subset": "dp_subset",
                    "beam_search": "beam_search"}.get(decode_method, "greedy_prob")
        mask    = _decode(probs, W, C, strategy=strategy,
                          values=V, top_m=top_m, beam_width=beam_width)
        ms      = (time.perf_counter() - t0) * 1000
        sel_idx = [i for i, x in enumerate(mask.tolist()) if x == 1]
        value   = int(values[sel_idx].sum()) if sel_idx else 0
        weight  = int(weights[sel_idx].sum()) if sel_idx else 0
        return {"value": value, "weight": weight, "time_ms": ms,
                "selected": sel_idx, "feasible": weight <= capacity,
                "decode": decode_method}
    except Exception as e:
        return {"value": None, "weight": None, "time_ms": 0,
                "selected": [], "feasible": False, "note": str(e)}

def _make_reinforce_fn(baseline: str):
    def _fn(w, v, c, cfg):
        RF_DIR = Path(__file__).resolve().parent / "results" / "GNN_REINFORCE"
        model_path = str(
            RF_DIR / f"gnn_reinforce_knn_{baseline}"
                   / f"gnn_reinforce_knn_{baseline}.pt"
        )
        return run_reinforce(w, v, c, model_path,
                             graph_type="knn",
                             decode_method=cfg.get("gnn_decode", "greedy"),
                             top_m=cfg.get("gnn_top_m", 30),
                             beam_width=cfg.get("beam_width", 5))
    return _fn

SOLVER_COLOR = {
    "DP":           "#5F5E5A",
    "Greedy":       "#7F77DD",
    "GA":           "#1D9E75",
    "BB":           "#888888",
    "GNN-kNN":      "#D85A30",
    "GNN-Conflict": "#D4537E",
    "GNN-Random":   "#2C5F9E",
    "GNN-Full":     "#378ADD",
    "DQN":          "#F5A623",
    "S2V-DQN":      "#E8A33D",
    "REINFORCE":        "#E74C3C",
    "REINFORCE-hard":   "#E74C3C",
    "REINFORCE-none":   "#C0392B",
    "REINFORCE-polyak": "#922B21",
}

GNN_GRAPH_TYPE = {
    "GNN-kNN":      "knn",
    "GNN-Conflict": "conflict_static",
    "GNN-Random":   "random",
    "GNN-Full":     "full",
}

def _make_gnn_fn(variant_key):
    graph_type = GNN_GRAPH_TYPE[variant_key]
    cfg_key = f"gnn_model_{graph_type}"   # e.g. gnn_model_knn, gnn_model_conflict_static
    def _fn(w, v, c, cfg):
        model_path = cfg.get(cfg_key) or cfg.get("gnn_model_default", "")
        return run_gnn(w, v, c, model_path, graph_type,
                       decode_method=cfg.get("gnn_decode", "greedy"),
                       top_m=cfg.get("gnn_top_m", 30),
                       beam_width=cfg.get("beam_width", 5))
    return _fn

SOLVER_FN = {
    "DP":           lambda w, v, c, cfg: run_dp(w, v, c),
    "Greedy":       lambda w, v, c, cfg: run_greedy(w, v, c),
    "GA":           lambda w, v, c, cfg: run_ga(w, v, c),
    "BB":           lambda w, v, c, cfg: run_bb(w, v, c, timeout=cfg.get("bb_timeout", 10)),
    "GNN-kNN":      _make_gnn_fn("GNN-kNN"),
    "GNN-Conflict": _make_gnn_fn("GNN-Conflict"),
    "GNN-Random":   _make_gnn_fn("GNN-Random"),
    "GNN-Full":     _make_gnn_fn("GNN-Full"),
    "DQN":          lambda w, v, c, cfg: run_dqn(w, v, c, cfg["dqn_model"]),
    "S2V-DQN":      lambda w, v, c, cfg: run_s2v_dqn(
                        w, v, c, cfg["s2v_model"],
                        graph_type=cfg.get("s2v_graph_type", "knn")),
    "REINFORCE":    lambda w, v, c, cfg: run_reinforce(
                        w, v, c, cfg["reinforce_model"],
                        graph_type=cfg.get("reinforce_graph_type", "knn"),
                        decode_method=cfg.get("gnn_decode", "greedy"),
                        top_m=cfg.get("gnn_top_m", 30),
                        beam_width=cfg.get("beam_width", 5)),
    "REINFORCE-hard": _make_reinforce_fn("hard"),
    "REINFORCE-none": _make_reinforce_fn("none"),
    "REINFORCE-polyak": _make_reinforce_fn("polyak"),
}

# ---------------------------------------------------------------------------
# Common data utilities
# ---------------------------------------------------------------------------
def _load_csv_values(path) -> dict:
    """Load {instance_file: total_value} from eval CSV."""
    try:
        df   = pd.read_csv(path)
        vcol = "total_value" if "total_value" in df.columns else "value"
        if vcol not in df.columns or "instance_file" not in df.columns:
            return {}
        return dict(zip(df["instance_file"], pd.to_numeric(df[vcol], errors="coerce")))
    except Exception:
        return {}

def _compute_ratio(solver_csv, dp_csv):
    """(mean_ratio, n) for matched instances."""
    s = _load_csv_values(solver_csv)
    d = _load_csv_values(dp_csv)
    common = [k for k in s if k in d and d[k] and d[k] > 0 and s[k] is not None and not np.isnan(s[k])]
    if not common:
        return None, 0
    ratios = [s[k] / d[k] for k in common]
    return float(np.mean(ratios)), len(common)

def _load_avg_time(path) -> Optional[float]:
    """Avg inference_time_ms từ eval CSV."""
    try:
        df = pd.read_csv(path)
        col = next((c for c in ["inference_time_ms", "time_ms"] if c in df.columns), None)
        if col is None:
            return None
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        return float(vals.mean()) if len(vals) else None
    except Exception:
        return None

def _load_training_log(path) -> Optional[dict]:
    """Load training log CSV → {col: list}."""
    try:
        df = pd.read_csv(path)
        return {col: pd.to_numeric(df[col], errors="coerce").tolist() for col in df.columns}
    except Exception:
        return None

# P1: find data/ subdirs containing NPZ instances
def _find_data_dirs():
    data_root = ROOT / "data"
    if not data_root.exists():
        return []
    dirs = []
    for d in sorted(data_root.rglob("*")):
        if d.is_dir() and any(d.glob("instance_*.npz")):
            dirs.append((str(d.relative_to(ROOT)), d))
    return dirs

# P1 / P2: load + validate a single NPZ instance
def _load_npz(path: Path):
    arr = np.load(str(path), allow_pickle=True)
    def pick(*keys):
        for k in keys:
            if k in arr.files: return arr[k]
        return None
    w_raw = pick("weights","w","W")
    v_raw = pick("values", "v","V")
    c_raw = pick("capacity","cap","C")
    if w_raw is None or v_raw is None or c_raw is None:
        raise KeyError(
            f"NPZ thiếu key bắt buộc — "
            f"weights={'OK' if w_raw is not None else 'MISSING'}, "
            f"values={'OK' if v_raw is not None else 'MISSING'}, "
            f"capacity={'OK' if c_raw is not None else 'MISSING'}. "
            f"Keys hiện có: {list(arr.files)}"
        )
    w   = np.asarray(w_raw).reshape(-1).astype(np.int32)
    v   = np.asarray(v_raw).reshape(-1).astype(np.int32)
    c   = int(np.asarray(c_raw).reshape(()).item())
    sol = pick("solution","selected","y")
    opt = pick("dp_value","optimal_value","opt")
    sol = np.asarray(sol).reshape(-1).astype(int) if sol is not None else None
    opt = int(np.asarray(opt).reshape(()).item())  if opt is not None else None
    return w, v, c, sol, opt

def _validate_npz(path: Path) -> dict:
    """Mirror of Check_Data.check_instance."""
    try:
        w, v, c, sol, opt = _load_npz(path)
    except Exception as e:
        return {"file": path.name, "ok": False, "errors": [str(e)]}
    errors = []
    if sol is None:
        errors.append("MISSING solution key")
    else:
        tw = int((w * sol).sum()); tv = int((v * sol).sum())
        if tw > c:          errors.append(f"CAPACITY VIOLATED: {tw} > {c}")
        if opt and tv != opt: errors.append(f"VALUE MISMATCH: {tv} != {opt}")
        if not set(sol.tolist()).issubset({0,1}): errors.append("SOLUTION NOT BINARY")
    if (w <= 0).any():  errors.append("NON-POSITIVE WEIGHT")
    if (v <= 0).any():  errors.append("NON-POSITIVE VALUE")
    if c <= 0:          errors.append("NON-POSITIVE CAPACITY")
    n_sel = int(sol.sum()) if sol is not None else 0
    return {
        "file": path.name, "n": len(w), "capacity": c,
        "n_selected": n_sel,
        "fill_ratio": round(int((w*sol).sum())/c, 3) if sol is not None and c > 0 else 0,
        "opt_value": opt, "ok": len(errors) == 0, "errors": errors,
    }

# ---------------------------------------------------------------------------
# Result card  (native Streamlit — không dùng HTML)
# ---------------------------------------------------------------------------
SOLVER_ORDER = ["DP", "Greedy", "GA", "BB",
                "GNN-kNN", "GNN-Conflict", "GNN-Random", "GNN-Full",
                "DQN", "S2V-DQN", "REINFORCE"]


def _result_key(solver_name: str, cfg: dict) -> str:
    """Tạo key duy nhất cho mỗi lần chạy solver (bao gồm decode/graph_type)."""
    decode = cfg.get("gnn_decode", "greedy")
    if solver_name in ("GNN-kNN", "GNN-Conflict", "GNN-Random", "GNN-Full"):
        return f"{solver_name}[{decode}]"
    if solver_name == "S2V-DQN":
        return f"S2V-DQN[{cfg.get('s2v_graph_type', 'knn')}]"
    if solver_name == "REINFORCE":
        return f"REINFORCE[{cfg.get('reinforce_graph_type', 'knn')}·{decode}]"
    return solver_name


def _solver_color(key: str) -> str:
    base = key.split("[")[0]
    return SOLVER_COLOR.get(key) or SOLVER_COLOR.get(base, "#999")


def _sort_results_keys(keys) -> list:
    def _k(k):
        base = k.split("[")[0]
        try:
            return (SOLVER_ORDER.index(base), k)
        except ValueError:
            return (len(SOLVER_ORDER), k)
    return sorted(keys, key=_k)


def _result_card(col, solver_name: str, result: dict, dp_value: Optional[int]):
    color = _solver_color(solver_name)
    value = result.get("value")
    with col:
        with st.container(border=True):
            st.markdown(
                f"<span style='color:{color};font-weight:700;font-size:1.05em'>"
                f"▌ {solver_name}</span>",
                unsafe_allow_html=True,
            )
            if value is None:
                st.error(f"Lỗi: {result.get('note','unknown')[:100]}")
                return

            m1, m2 = st.columns(2)
            m1.metric("Value", f"{value:,}")
            m2.metric("Time", f"{result['time_ms']:.1f} ms")

            capacity = result.get("capacity", 0) or 0
            weight   = result.get("weight",   0) or 0
            fill     = weight / capacity if capacity > 0 else 0
            st.progress(min(fill, 1.0),
                        text=f"Weight {weight:,} / {capacity:,}  ({fill*100:.0f}%)")

            tags = [f"Items: {len(result['selected'])}"]
            if dp_value and dp_value > 0:
                tags.append(f"Ratio: {value/dp_value:.4f}")
            if "decode" in result:
                tags.append(f"Decode: {result['decode']}")
            if result.get("note"):
                tags.append(f"⚠ {result['note'][:50]}")
            badge = "✅ Feasible" if result["feasible"] else "❌ Infeasible"
            st.caption(f"{badge}  |  {'  ·  '.join(tags)}")

# ---------------------------------------------------------------------------
# TAB 1 — Dashboard
# ---------------------------------------------------------------------------
def tab_dashboard():
    st.header("Dashboard — Kết quả đã đánh giá")

    # --- Size selector: cho phép xem theo từng test set ---
    SIZE_TO_DIR = {
        "small (n≤50)": "compare_small",
        "n=100":        "compare_n100",
        "n=200":        "compare_n200",
        "All sizes (gộp)": "compare",
    }
    available_sizes = [
        label for label, sub in SIZE_TO_DIR.items()
        if (ROOT / "results" / sub / "summary.json").exists()
    ]
    if not available_sizes:
        st.warning("Chưa có `summary.json` ở bất kỳ thư mục `results/compare*/` nào. "
                   "Chạy `Merge_results.py` trước.")
        return

    size_choice = st.radio(
        "Tập kiểm tra",
        available_sizes,
        horizontal=True,
        key="_dash_size",
        help="Chọn theo n=size để xem cross-scale; 'All sizes' là kết quả đã merge gộp.",
    )
    sub_dir      = SIZE_TO_DIR[size_choice]
    summary_path = ROOT / "results" / sub_dir / "summary.json"
    merged_path  = ROOT / "results" / sub_dir / "merged_results.csv"

    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        rows = []
        for solver_key, d in _iter_summary_solvers(summary):
            rows.append({
                "Solver":       _solver_display_label(solver_key),
                "N":            d.get("count", 0),
                "Feasible%":    f"{d.get('feasible_rate',0)*100:.1f}%",
                "Avg Ratio":    f"{d['avg_ratio_vs_dp_feasible']:.4f}" if d.get("avg_ratio_vs_dp_feasible") else "N/A",
                "Std Ratio":    f"{d['std_ratio_vs_dp']:.4f}"          if d.get("std_ratio_vs_dp")          else "N/A",
                "Avg Time(ms)": f"{d['avg_time_ms']:.2f}"              if d.get("avg_time_ms")              else "N/A",
            })
        st.subheader(f"Bảng tổng hợp — {size_choice}")
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # --- Dynamic charts từ summary.json đang chọn ---
        chart_rows = [
            (_solver_display_label(k),
             d.get("avg_ratio_vs_dp_feasible"),
             d.get("avg_time_ms"))
            for k, d in _iter_summary_solvers(summary)
        ]
        chart_rows = [(n, r, t) for n, r, t in chart_rows
                      if r is not None and t is not None]

        if chart_rows:
            names  = [n for n, _, _ in chart_rows]
            ratios = [r for _, r, _ in chart_rows]
            times  = [t for _, _, t in chart_rows]

            st.subheader(f"Biểu đồ kết quả — {size_choice}")
            fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5))

            # Bar — Approximation Ratio
            barsL = axL.barh(names, ratios, color="#4A90E2", alpha=0.85)
            axL.axvline(1.0, color="#222", lw=1.2, linestyle="--",
                        label="DP optimal (1.0)")
            axL.set_xlim(min(0.85, min(ratios) - 0.02), 1.01)
            axL.set_xlabel("Approximation Ratio (vs DP)")
            axL.set_title("Chất lượng giải")
            axL.legend(fontsize=9, loc="lower right")
            axL.grid(axis="x", alpha=0.3)
            for bar, val in zip(barsL, ratios):
                axL.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                         f"{val:.4f}", va="center", fontsize=8)

            # Bar — Inference time (log scale)
            barsR = axR.barh(names, times, color="#D85A30", alpha=0.85)
            axR.set_xscale("log")
            axR.set_xlabel("Avg Inference Time (ms, log scale)")
            axR.set_title("Tốc độ inference")
            axR.grid(axis="x", alpha=0.3)
            for bar, val in zip(barsR, times):
                axR.text(val * 1.05, bar.get_y() + bar.get_height()/2,
                         f"{val:.2f}", va="center", fontsize=8)

            plt.suptitle(f"Solver comparison on {size_choice}", y=1.02)
            plt.tight_layout()
            st.pyplot(fig); plt.close()
    else:
        st.warning(f"Chưa có `{summary_path.relative_to(ROOT)}`. "
                   "Chạy `Merge_results.py` trước.")

    # --- Dynamic charts từ merged_results.csv (per-instance level) ---
    if merged_path.exists():
        df_m = pd.read_csv(merged_path)
        # Auto-detect ratio columns: tất cả cột *_ratio trừ dp_*
        ratio_cols = [
            (col[:-len("_ratio")], col)
            for col in df_m.columns
            if col.endswith("_ratio") and not col.startswith("dp_")
        ]
        # Greedy column (anchor) — dùng để vẽ scatter
        greedy_col = "greedy_ratio" if "greedy_ratio" in df_m.columns else None

        # Sort theo family order
        def _ratio_key(item):
            fam, var = _split_solver_key(item[0])
            try: idx = _SOLVER_FAMILY_ORDER.index(fam)
            except ValueError: idx = 99
            return (idx, var or "")
        ratio_cols.sort(key=_ratio_key)

        # Chart: Ratio distribution histogram per solver (replace chart 3) ---
        if ratio_cols:
            st.subheader(f"Phân phối Approximation Ratio — {size_choice}")
            n = len(ratio_cols); ncols = 4; nrows = (n + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3.2 * nrows))
            axes = axes.flatten() if n > 1 else [axes]
            for ax, (key, col) in zip(axes, ratio_cols):
                vals = pd.to_numeric(df_m[col], errors="coerce").dropna()
                if vals.empty:
                    ax.set_visible(False); continue
                color = _solver_color(key)
                if vals.std() < 1e-9:
                    ax.axvline(vals.iloc[0], color=color, lw=3)
                    ax.text(0.5, 0.5, f"All = {vals.iloc[0]:.4f}",
                            transform=ax.transAxes, ha="center", va="center", fontsize=10)
                    ax.set_xlim(0.85, 1.01); ax.set_ylim(0, 1)
                else:
                    ax.hist(vals, bins=20, color=color, alpha=0.85, edgecolor="white", lw=0.4)
                    med = vals.median()
                    ax.axvline(med, color="black", linestyle="--", lw=1.2,
                               label=f"med={med:.4f}")
                    ax.axvline(1.0, color="red", linestyle=":", lw=1.0, alpha=0.6)
                    ax.legend(fontsize=7, loc="upper left")
                    ax.set_xlim(max(0.7, vals.min() - 0.01), 1.01)
                ax.set_title(_solver_display_label(key), fontsize=10, color=color)
                ax.grid(axis="y", alpha=0.25)
            for ax in axes[len(ratio_cols):]:
                ax.set_visible(False)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # Chart: Win/Tie/Lose vs Greedy (replace chart 5) ---
        if greedy_col and ratio_cols:
            st.subheader(f"Head-to-head vs Greedy — {size_choice}")
            wtl_rows = []
            gv = pd.to_numeric(df_m[greedy_col], errors="coerce")
            for key, col in ratio_cols:
                if key == "greedy": continue
                sv = pd.to_numeric(df_m[col], errors="coerce")
                mask = sv.notna() & gv.notna()
                if mask.sum() == 0: continue
                wins  = int(((sv > gv) & mask).sum())
                ties  = int(((sv == gv) & mask).sum())
                loses = int(((sv < gv) & mask).sum())
                wtl_rows.append({
                    "Solver": _solver_display_label(key),
                    "Win": wins, "Tie": ties, "Lose": loses, "Total": int(mask.sum()),
                })
            if wtl_rows:
                df_wtl = pd.DataFrame(wtl_rows)
                fig, ax = plt.subplots(figsize=(12, max(3, 0.5 * len(df_wtl))))
                y = np.arange(len(df_wtl))
                ax.barh(y, df_wtl["Win"],  color="#1D9E75", label="Win > Greedy")
                ax.barh(y, df_wtl["Tie"],  left=df_wtl["Win"],
                        color="#888888", label="Tie = Greedy")
                ax.barh(y, df_wtl["Lose"], left=df_wtl["Win"]+df_wtl["Tie"],
                        color="#D85A30", label="Lose < Greedy")
                ax.set_yticks(y); ax.set_yticklabels(df_wtl["Solver"])
                ax.set_xlabel(f"Số instance (tổng {int(df_wtl['Total'].max())})")
                ax.legend(loc="lower right", fontsize=9)
                ax.invert_yaxis()
                plt.tight_layout(); st.pyplot(fig); plt.close()
                st.dataframe(df_wtl, use_container_width=True, hide_index=True)

    # --- Static PNG plots cũ (chỉ show nếu user đã rerun plot_results.py) ---
    plot_dir   = ROOT / "plots"
    plot_files = {
        "So sánh solver (ratio + time)": "solver_comparison.png",
        "Phân phối ratio":               "ratio_distribution.png",
        "Ratio theo kích thước":         "ratio_by_size.png",
        "GNN vs Greedy scatter":         "gnn_vs_greedy.png",
    }
    available_pngs = [(t, plot_dir / f) for t, f in plot_files.items()
                      if (plot_dir / f).exists()]
    if available_pngs:
        with st.expander("Biểu đồ PNG cũ (chỉ tham khảo — không cập nhật theo size)",
                          expanded=False):
            st.caption("⚠️ Các PNG này sinh từ schema cũ (`gnn_ratio` single column). "
                       "Khuyến nghị dùng dynamic chart bên trên.")
            for title, p in available_pngs:
                st.markdown(f"**{title}**")
                st.image(str(p))

    # P1: xem kết quả từng solver (raw CSV)
    st.subheader("Kết quả chi tiết từng solver")

    # --- P1a: Solvers đơn (DP, Greedy, GA, BB, GNN, DQN) ---
    SOLVER_CSVS = {
        "DP":     ROOT / "results" / "cross_scale"  / "dp_small.csv",
        "Greedy": ROOT / "results" / "cross_scale" / "greedy_small.csv",
        "GA":     ROOT / "results" / "cross_scale"  / "ga_small.csv",
        "BB":     ROOT / "results" / "BB"  / "bb_small.csv",
        "GNN-Greedy decode":    ROOT / "results" / "GNN" / "gnn_eval_results.csv",
        "GNN-DP decode": ROOT / "results" / "GNN" / "gnn_dp_eval_results.csv",
        "DQN":    ROOT / "results" / "DQN" / "dqn_on_small.csv",
    }
    available = {k: v for k, v in SOLVER_CSVS.items() if v.exists()}
    if available:
        chosen = st.selectbox("Chọn solver", list(available.keys()), key="_dash_solver_sel")
        df_raw = pd.read_csv(available[chosen])
        for col in ["total_value","total_weight","dp_value","ratio","inference_time_ms","feasible"]:
            if col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
        st.dataframe(df_raw, use_container_width=True)
        if "ratio" in df_raw.columns:
            r = df_raw["ratio"].dropna()
            sc1,sc2,sc3,sc4 = st.columns(4)
            sc1.metric("N",         len(r))
            sc2.metric("Avg ratio", f"{r.mean():.4f}")
            sc3.metric("Std",       f"{r.std():.4f}")
            sc4.metric("Min",       f"{r.min():.4f}")
    else:
        st.info("Chưa có file CSV kết quả từng solver.")

    # --- P1b: S2V-DQN và REINFORCE — chia theo 4 graph type ---
    GRAPH_TYPES   = ["knn", "conflict_static", "random", "full"]
    GRAPH_LABELS  = {"knn":"kNN", "conflict_static":"Conflict Static",
                     "random":"Random", "full":"Full"}
    GRAPH_COLORS  = {"knn":"#D85A30","conflict_static":"#D4537E",
                     "random":"#1D9E75","full":"#378ADD"}

    # File path conventions (per graph type)
    S2V_DIR  = ROOT / "results" / "S2V_DQN"
    RF_DIR   = ROOT / "results" / "GNN_REINFORCE"

    def _s2v_paths(gt: str):
        """Trả về (subfolder, train_meta, eval_csv) cho graph_type gt."""
        sub      = S2V_DIR / f"s2v_dqn_{gt.replace('_static','').replace('_','')}"
        # handle naming: knn→s2v_dqn_knn, conflict_static→s2v_dqn_conflict, random→..., full→...
        for candidate in [
            S2V_DIR / f"s2v_dqn_{gt}",
            S2V_DIR / f"s2v_dqn_{gt.split('_')[0]}",
        ]:
            if candidate.is_dir():
                sub = candidate
                break
        meta_f  = sub / "train_meta.json"
        eval_f  = sub / f"s2v_dqn_eval_results_{gt}.csv"
        # fallback 1: any eval csv in subfolder
        if not eval_f.exists():
            candidates = list(sub.glob("*eval_results*.csv")) if sub.exists() else []
            if candidates:
                eval_f = candidates[0]
        # fallback 2: cross_scale CSV (s2v_<short>_on_small.csv)
        if not eval_f.exists():
            short = gt.replace("_static", "")  # conflict_static -> conflict
            cs_csv = ROOT / "results" / "cross_scale" / f"s2v_{short}_on_small.csv"
            if cs_csv.exists():
                eval_f = cs_csv
        return sub, meta_f, eval_f

    def _rf_paths(gt: str):
        # Support both old (gnnrl_knn_training_log.csv) and new (gnnrl_knn_polyak_log.csv) naming
        log_candidates = (
            list(RF_DIR.glob(f"gnnrl_{gt}_*_log.csv")) +
            [RF_DIR / f"gnnrl_{gt}_training_log.csv"]
        )
        train_log = next((f for f in log_candidates if f.exists()), None)
        eval_f = RF_DIR / f"reinforce_{gt}_eval_results.csv"
        if gt == "knn" and not eval_f.exists():
            legacy = RF_DIR / "reinforce_eval_results.csv"
            if legacy.exists():
                eval_f = legacy
        # fallback: cross_scale CSV (reinforce_<baseline>_on_small.csv — only kNN trained)
        if not eval_f.exists() and gt == "knn":
            for baseline in ("polyak", "hard", "none"):
                cs_csv = ROOT / "results" / "cross_scale" / f"reinforce_{baseline}_on_small.csv"
                if cs_csv.exists():
                    eval_f = cs_csv
                    break
        return train_log, eval_f

    def _load_meta(meta_f: Path) -> dict:
        if meta_f.exists() and meta_f.suffix == ".json":
            try:
                return json.loads(meta_f.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _show_eval_csv(eval_f: Path):
        """Hiển thị stats + dataframe từ eval CSV."""
        df = pd.read_csv(eval_f)
        for col in ["total_value","total_weight","dp_value","ratio","inference_time_ms","feasible"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        r = df["ratio"].dropna() if "ratio" in df.columns else pd.Series([], dtype=float)
        feas = df["feasible"].dropna() if "feasible" in df.columns else pd.Series([], dtype=float)
        t    = df["inference_time_ms"].dropna() if "inference_time_ms" in df.columns else pd.Series([], dtype=float)
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("N instances",  len(df))
        c2.metric("Avg Ratio/DP", f"{r.mean():.4f}" if len(r) else "N/A")
        c3.metric("Std",          f"{r.std():.4f}"  if len(r) > 1 else "N/A")
        c4.metric("Feasible%",    f"{feas.mean()*100:.1f}%" if len(feas) else "N/A")
        c5.metric("Avg Time(ms)", f"{t.mean():.2f}" if len(t) else "N/A")
        with st.expander("Xem dữ liệu thô", expanded=False):
            st.dataframe(df, use_container_width=True, hide_index=True)

    # ---- S2V-DQN section ----
    st.markdown("---")
    st.subheader("S2V-DQN — kết quả theo Graph Type")

    # Status overview table
    s2v_status_rows = []
    for gt in GRAPH_TYPES:
        sub, meta_f, eval_f = _s2v_paths(gt)
        meta = _load_meta(meta_f)
        trained  = meta_f.exists()
        evaluated = eval_f.exists()
        s2v_status_rows.append({
            "Graph Type":    GRAPH_LABELS[gt],
            "Trạng thái":    "✅ Evaluated" if evaluated else ("🟡 Trained" if trained else "⬜ Chưa chạy"),
            "Best Val Ratio": f"{meta.get('best_val_ratio', 'N/A'):.4f}" if meta.get("best_val_ratio") else "N/A",
            "Test Ratio":    f"{meta.get('test_ratio', 'N/A'):.4f}" if meta.get("test_ratio") else "N/A",
            "Test Feas%":    f"{meta.get('test_feasibility', 0)*100:.1f}%" if meta.get("test_feasibility") is not None else "N/A",
            "Eval CSV":      eval_f.name if evaluated else "—",
        })
    st.dataframe(pd.DataFrame(s2v_status_rows), use_container_width=True, hide_index=True)

    # Tabs per graph type
    s2v_tabs = st.tabs([f"{GRAPH_LABELS[gt]}" for gt in GRAPH_TYPES])
    for tab, gt in zip(s2v_tabs, GRAPH_TYPES):
        with tab:
            sub, meta_f, eval_f = _s2v_paths(gt)
            meta    = _load_meta(meta_f)
            trained  = meta_f.exists()
            evaluated = eval_f.exists()
            color = GRAPH_COLORS[gt]
            st.markdown(
                f"<span style='color:{color};font-weight:700'>▌ S2V-DQN [{GRAPH_LABELS[gt]}]</span>",
                unsafe_allow_html=True,
            )
            if meta:
                m1,m2,m3,m4 = st.columns(4)
                m1.metric("Train steps",     meta.get("train_steps","N/A"))
                m2.metric("Best Val Ratio",  f"{meta['best_val_ratio']:.4f}" if meta.get("best_val_ratio") else "N/A")
                m3.metric("Test Ratio",      f"{meta['test_ratio']:.4f}"     if meta.get("test_ratio")     else "N/A")
                m4.metric("Test Feas%",      f"{meta['test_feasibility']*100:.1f}%" if meta.get("test_feasibility") is not None else "N/A")
            if evaluated:
                st.caption(f"Eval CSV: `{eval_f.relative_to(ROOT)}`")
                _show_eval_csv(eval_f)
            elif trained:
                st.info(f"Đã train nhưng chưa đánh giá. Chạy:\n"
                        f"```\npython src/rl/S2V_DQN/Evaluate_S2V_DQN.py "
                        f"--model_path results/S2V_DQN/s2v_dqn_{gt.split('_')[0]}/s2v_dqn_{gt}_best.pt "
                        f"--graph_type {gt} "
                        f"--out_csv results/S2V_DQN/s2v_dqn_{gt.split('_')[0]}/s2v_dqn_eval_results_{gt}.csv\n```")
            else:
                st.warning(f"Chưa train graph_type `{gt}`.")

    # ---- REINFORCE section ----
    st.markdown("---")
    st.subheader("REINFORCE — kết quả theo Graph Type")

    def _rf_log_meta(train_log: Optional[Path]) -> dict:
        """Extract best_val_ratio, baseline_type, n_epochs from training log."""
        if train_log is None or not train_log.exists():
            return {}
        try:
            df = pd.read_csv(train_log)
            df["val_ratio"] = pd.to_numeric(df.get("val_ratio", pd.Series(dtype=float)), errors="coerce")
            best_ratio = df["val_ratio"].max() if "val_ratio" in df.columns else None
            n_epochs   = len(df)
            # Extract baseline type from filename: gnnrl_{gt}_{baseline_type}_log.csv
            stem = train_log.stem  # e.g. gnnrl_knn_polyak_log
            parts = stem.split("_")
            baseline = parts[-2] if len(parts) >= 4 and parts[-1] == "log" else "polyak"
            return {"best_ratio": best_ratio, "n_epochs": n_epochs, "baseline": baseline}
        except Exception:
            return {}

    rf_status_rows = []
    for gt in GRAPH_TYPES:
        train_log, eval_f = _rf_paths(gt)
        trained   = train_log is not None
        evaluated = eval_f.exists()
        meta = _rf_log_meta(train_log)
        rf_status_rows.append({
            "Graph Type":    GRAPH_LABELS[gt],
            "Trạng thái":    "✅ Evaluated" if evaluated else ("🟡 Trained" if trained else "⬜ Chưa chạy"),
            "Baseline":      meta.get("baseline", "—"),
            "Epochs":        meta.get("n_epochs", "—"),
            "Best val_ratio": f"{meta['best_ratio']:.4f}" if meta.get("best_ratio") else "—",
            "Training log":  train_log.name if trained else "—",
            "Eval CSV":      eval_f.name if evaluated else "—",
        })
    st.dataframe(pd.DataFrame(rf_status_rows), use_container_width=True, hide_index=True)

    rf_tabs = st.tabs([f"{GRAPH_LABELS[gt]}" for gt in GRAPH_TYPES])
    for tab, gt in zip(rf_tabs, GRAPH_TYPES):
        with tab:
            train_log, eval_f = _rf_paths(gt)
            trained   = train_log is not None
            evaluated = eval_f.exists()
            color = GRAPH_COLORS[gt]
            st.markdown(
                f"<span style='color:{color};font-weight:700'>▌ REINFORCE [{GRAPH_LABELS[gt]}]</span>",
                unsafe_allow_html=True,
            )
            if evaluated:
                st.caption(f"Eval CSV: `{eval_f.relative_to(ROOT)}`")
                _show_eval_csv(eval_f)
            elif trained:
                st.info(f"Đã train nhưng chưa đánh giá. Chạy:\n"
                        f"```\npython src/rl/REINFORCE/Evaluate_reinforce.py "
                        f"--model_path results/GNN_REINFORCE/gnn_reinforce_{gt}/gnn_reinforce_{gt}.pt "
                        f"--graph_type {gt} "
                        f"--out_csv results/GNN_REINFORCE/reinforce_{gt}_eval_results.csv\n```")
            else:
                st.warning(f"Chưa train graph_type `{gt}`.")
            if trained:
                with st.expander("Training log", expanded=False):
                    try:
                        df_log = pd.read_csv(train_log)
                        for col in df_log.columns:
                            df_log[col] = pd.to_numeric(df_log[col], errors="coerce").fillna(df_log[col])

                        # Mini chart: val_ratio + avg_reward per epoch
                        chart_cols = [c for c in ["val_ratio", "avg_reward", "adv_std", "entropy"] if c in df_log.columns]
                        if "epoch" in df_log.columns and chart_cols:
                            fig, axes = plt.subplots(1, len(chart_cols), figsize=(4 * len(chart_cols), 3))
                            if len(chart_cols) == 1:
                                axes = [axes]
                            for ax, col in zip(axes, chart_cols):
                                ax.plot(df_log["epoch"], pd.to_numeric(df_log[col], errors="coerce"),
                                        linewidth=1.5, color=color)
                                ax.set_title(col, fontsize=9)
                                ax.set_xlabel("epoch", fontsize=8)
                                ax.grid(axis="y", alpha=0.3)
                            plt.tight_layout()
                            st.pyplot(fig, use_container_width=True)
                            plt.close()

                        st.caption(f"**Log file:** `{train_log.name}` — {len(df_log)} epochs")
                        st.dataframe(df_log.tail(10), use_container_width=True, hide_index=True)
                    except Exception as ex:
                        st.caption(f"Không đọc được `{train_log.name}`: {ex}")

    # P2: Merged results interactive table
    st.subheader("Bảng so sánh chi tiết (merged_results.csv)")
    if merged_path.exists():
        df_merged = pd.read_csv(merged_path)
        # Detect ratio columns dynamically — handles multi-variant schema
        # (gnn_knn_ratio, s2v_full_ratio, reinforce_hard_ratio, etc.).
        solver_ratio_cols = {}
        for col in df_merged.columns:
            if col.endswith("_ratio") and col not in ("dp_ratio",):
                solver_key = col[:-len("_ratio")]
                solver_ratio_cols[_solver_display_label(solver_key)] = col
        # Filter controls
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            n_min, n_max = int(df_merged["n_items"].min()), int(df_merged["n_items"].max())
            n_range = st.slider("n_items range", n_min, n_max, (n_min, n_max))
        with fc2:
            avail_solvers = [k for k, v in solver_ratio_cols.items() if v in df_merged.columns and df_merged[v].notna().any()]
            chosen_solver = st.selectbox("Filter: solver thắng", ["(tất cả)"] + avail_solvers)
        with fc3:
            ratio_min = st.slider("Ratio tối thiểu", 0.0, 1.0, 0.0, step=0.01)

        mask = (df_merged["n_items"] >= n_range[0]) & (df_merged["n_items"] <= n_range[1])
        if chosen_solver != "(tất cả)":
            col = solver_ratio_cols[chosen_solver]
            if col in df_merged.columns:
                mask &= df_merged[col] >= ratio_min
        df_view = df_merged[mask].copy()

        # Hiển thị chỉ các cột quan trọng
        display_cols = ["instance_file", "n_items", "capacity", "dp_value"]
        for k, col in solver_ratio_cols.items():
            if col in df_view.columns:
                display_cols.append(col)
        for col in display_cols:
            if col in df_view.columns:
                df_view[col] = pd.to_numeric(df_view[col], errors="coerce")

        st.caption(f"Hiển thị {len(df_view)}/{len(df_merged)} instance")
        st.dataframe(df_view[display_cols].rename(columns={v: k for k, v in solver_ratio_cols.items()}),
                     use_container_width=True, hide_index=True)

        # Quick summary stats
        st.caption("**Thống kê nhanh (trên tập đã lọc):**")
        stat_cols = st.columns(len(avail_solvers))
        for col_ui, solver_name in zip(stat_cols, avail_solvers):
            col = solver_ratio_cols[solver_name]
            vals = pd.to_numeric(df_view[col], errors="coerce").dropna()
            col_ui.metric(solver_name, f"{vals.mean():.4f}" if len(vals) else "N/A",
                          f"±{vals.std():.4f}" if len(vals) > 1 else "")
    else:
        st.info("Chưa có `merged_results.csv`. Chạy `Merge_results.py` trước.")

    # P3: Pisinger results
    st.subheader("Kết quả trên Pisinger Hard Instances")
    PISINGER_SETS = {
        "Pisinger (mixed)": (
            ROOT/"results"/"GNN_Pisinger_t3"/"dp_test.csv",
            ROOT/"results"/"GNN_Pisinger_t3"/"greedy_test.csv",
        ),
        "Pisinger Type 4": (
            ROOT/"results"/"GNN_Pisinger_t3"/"dp.csv",
            ROOT/"results"/"GNN_Pisinger_t3"/"greedy.csv",
        ),
        "Pisinger Type 5": (
            ROOT/"results"/"pisinger_t5"/"dp.csv",
            ROOT/"results"/"pisinger_t5"/"greedy.csv",
        ),
        "Pisinger Hard": (
            ROOT/"results"/"pisinger_hard"/"dp.csv",
            ROOT/"results"/"pisinger_hard"/"greedy.csv",
        ),
    }
    pis_rows = []
    for label, (dp_p, gr_p) in PISINGER_SETS.items():
        if not dp_p.exists(): continue
        dp_vals = _load_csv_values(dp_p)
        gr_vals = _load_csv_values(gr_p) if gr_p.exists() else {}
        n       = len(dp_vals)
        avg_dp  = float(np.mean(list(dp_vals.values()))) if dp_vals else 0
        row = {"Dataset": label, "N": n, "Avg DP value": f"{avg_dp:.1f}"}
        if gr_vals:
            r, _ = _compute_ratio(gr_p, dp_p)
            row["Greedy ratio"] = f"{r:.4f}" if r else "N/A"
        pis_rows.append(row)
    if pis_rows:
        st.dataframe(pd.DataFrame(pis_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Chưa có kết quả Pisinger trong `results/pisinger*/`.")

# ---------------------------------------------------------------------------
# Live Demo helpers
# ---------------------------------------------------------------------------
def _export_instance_json(weights, values, capacity) -> bytes:
    return json.dumps({
        "n": len(weights), "capacity": int(capacity),
        "weights": weights.tolist(), "values": values.tolist(),
    }, indent=2).encode()

def _export_instance_csv(weights, values) -> bytes:
    return pd.DataFrame({"weight": weights, "value": values}).to_csv(index=False).encode()

def _validate_instance(weights, values, capacity):
    warns, infos = [], []
    if (weights <= 0).any():  warns.append(f"{(weights<=0).sum()} item có weight ≤ 0")
    if (values  <= 0).any():  warns.append(f"{(values<=0).sum()}  item có value ≤ 0")
    if capacity <= 0:          warns.append("Capacity ≤ 0")
    too_heavy = int((weights > capacity).sum())
    if too_heavy: warns.append(f"{too_heavy} item nặng hơn capacity (không thể chọn)")
    if int(weights.sum()) <= capacity:
        infos.append("Tổng weight ≤ capacity — trivial instance (có thể chọn tất cả)")
    if len(weights) > 1:
        corr = float(np.corrcoef(weights, values)[0, 1])
        infos.append(f"Tương quan weight–value: {corr:.3f}")
    return warns, infos

def _live_comparison_chart(results: dict, capacity: int):
    valid_keys = [k for k in _sort_results_keys(results.keys())
                  if results[k].get("value") is not None]
    if not valid_keys:
        return

    valid   = {k: results[k] for k in valid_keys}
    names   = valid_keys
    values_ = [valid[n]["value"]   for n in names]
    times   = [valid[n]["time_ms"] for n in names]
    colors  = [_solver_color(n) for n in names]

    st.subheader("So sánh trực quan")

    # --- Chart 1: Value bar + Time bar ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].bar(names, values_, color=colors, edgecolor="white")
    axes[0].set_title("Value đạt được"); axes[0].set_ylabel("Value")
    axes[0].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    v_max = max(values_) if values_ else 1
    for i, v in enumerate(values_):
        axes[0].text(i, v + v_max * 0.01, f"{v:,}", ha="center", fontsize=8)
    axes[1].bar(names, times, color=colors, edgecolor="white")
    axes[1].set_yscale("log"); axes[1].set_title("Thời gian chạy (log scale)")
    axes[1].set_ylabel("ms"); axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    for i, t in enumerate(times):
        axes[1].text(i, t * 1.3, f"{t:.1f}", ha="center", fontsize=8)
    plt.tight_layout(); st.pyplot(fig); plt.close()

    # --- Chart 2: Ratio vs DP (nếu có DP) ---
    dp_val = valid.get("DP", {}).get("value")
    if dp_val and dp_val > 0:
        ratios = [valid[n]["value"] / dp_val for n in names]
        fig2, ax2 = plt.subplots(figsize=(max(6, len(names) * 0.9), 4))
        bars = ax2.barh(names, ratios, color=colors, edgecolor="white")
        ax2.axvline(1.0, color="#333", linewidth=1.2, linestyle="--", label="DP (optimal)")
        ax2.set_xlabel("Ratio vs DP (1.0 = optimal)")
        ax2.set_title("Chất lượng nghiệm so với DP Exact")
        ax2.set_xlim(0, max(1.05, max(ratios) * 1.05))
        for bar, ratio in zip(bars, ratios):
            ax2.text(ratio + 0.003, bar.get_y() + bar.get_height() / 2,
                     f"{ratio:.4f}", va="center", fontsize=8)
        ax2.legend(fontsize=8)
        plt.tight_layout(); st.pyplot(fig2); plt.close()

    # --- Chart 3: Scatter Value vs Time ---
    if len(names) >= 2:
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        max_v = max(values_)
        min_v = min(values_)
        max_t = max(times)
        for n, v, t, c in zip(names, values_, times, colors):
            ax3.scatter(t, v, color=c, s=100, zorder=4)
            ax3.annotate(n, (t, v), textcoords="offset points",
                         xytext=(5, 4), fontsize=7, color=c)
        ax3.set_xscale("log")
        ax3.set_xlabel("Thời gian (ms, log scale)")
        ax3.set_ylabel("Value đạt được")
        ax3.set_title("Hiệu quả: Value vs Time")
        ax3.grid(True, alpha=0.4, linestyle='--')
        ax3.text(0.02, 0.96, f"Best Value: {max_v:,.0f}",
                 transform=ax3.transAxes, fontsize=13, fontweight='bold',
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.95))
        plt.tight_layout(); st.pyplot(fig3); plt.close()

    # --- Chart 4: Fill rate (% capacity used) ---
    fills = [valid[n].get("weight", 0) / capacity * 100 if capacity > 0 else 0 for n in names]
    fig4, ax4 = plt.subplots(figsize=(max(6, len(names) * 0.9), 3.5))
    bars4 = ax4.bar(names, fills, color=colors, edgecolor="white")
    ax4.axhline(100, color="#c0392b", linewidth=1.2, linestyle="--", label="Capacity limit (100%)")
    ax4.set_ylim(0, 115)
    ax4.set_ylabel("Fill rate (%)")
    ax4.set_title("Tỷ lệ sử dụng Capacity")
    ax4.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    for bar, f in zip(bars4, fills):
        ax4.text(bar.get_x() + bar.get_width() / 2, f + 1.2,
                 f"{f:.1f}%", ha="center", fontsize=8)
    ax4.legend(fontsize=8)
    plt.tight_layout(); st.pyplot(fig4); plt.close()

    # --- Chart 5: Item selection heatmap (chỉ hiện khi n_items ≤ 120) ---
    all_selected = {n: set(int(i) for i in valid[n].get("selected", [])) for n in names}
    n_items_total = max((max(s) for s in all_selected.values() if s), default=-1) + 1
    if 0 < n_items_total <= 120 and len(names) >= 2:
        matrix = np.zeros((len(names), n_items_total), dtype=np.float32)
        for row_i, n in enumerate(names):
            for item_idx in all_selected[n]:
                if item_idx < n_items_total:
                    matrix[row_i, item_idx] = 1.0

        fig5_w = max(10, n_items_total * 0.18)
        fig5_h = max(2.5, len(names) * 0.5)
        fig5, ax5 = plt.subplots(figsize=(fig5_w, fig5_h))
        im = ax5.imshow(matrix, aspect="auto", cmap="Blues", vmin=0, vmax=1,
                        interpolation="nearest")
        ax5.set_yticks(range(len(names)))
        ax5.set_yticklabels(names, fontsize=8)
        ax5.set_xlabel("Item index")
        ax5.set_title("Item selection heatmap — item nào được chọn bởi solver nào")
        plt.colorbar(im, ax=ax5, fraction=0.02, pad=0.02).set_label("Được chọn", fontsize=7)
        plt.tight_layout(); st.pyplot(fig5); plt.close()

    # --- Bảng tổng hợp ---
    rows = []
    for name in names:
        r = valid[name]
        v = r["value"]
        row = {
            "Solver":    name,
            "Value":     f"{v:,}",
            "Weight":    f"{r.get('weight', 0):,}",
            "Fill %":    f"{r.get('weight', 0) / capacity * 100:.1f}%" if capacity > 0 else "—",
            "Items":     len(r["selected"]),
            "Time (ms)": f"{r['time_ms']:.2f}",
            "Feasible":  "✅" if r["feasible"] else "❌",
        }
        if dp_val and dp_val > 0:
            row["Ratio vs DP"] = f"{v / dp_val:.4f}"
        if r.get("note"):
            row["Ghi chú"] = r["note"][:40]
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# Results table + save
# ---------------------------------------------------------------------------
def _build_results_df(ran: list, results: dict, capacity: int,
                      dp_val: Optional[int]) -> pd.DataFrame:
    rows = []
    for name in ran:
        r     = results[name]
        ok    = r.get("value") is not None
        value = r.get("value")
        wt    = r.get("weight", 0) or 0
        row   = {
            "Solver":     name,
            "Trạng thái": "OK"   if ok else "Lỗi",
            "Value":      value  if ok else None,
            "Weight":     wt     if ok else None,
            "Fill %":     round(wt / capacity * 100, 1) if ok and capacity > 0 else None,
            "Items chọn": len(r["selected"]) if ok else None,
            "Time (ms)":  round(r["time_ms"], 2) if ok else None,
            "Feasible":   r["feasible"] if ok else None,
            "Ratio/DP":   round(value / dp_val, 4) if ok and dp_val and dp_val > 0 else None,
            "Ghi chú":    r.get("note", "") if not ok
                          else (f"decode={r['decode']}" + (f",top_m={r['top_m']}" if r.get("top_m") else "")
                                if name == "GNN" and "decode" in r
                                else r.get("note", "")),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _results_table(ran: list, results: dict, capacity: int, dp_val: Optional[int]):
    """Bảng tổng hợp — plain dataframe (không dùng pandas Styler để tránh trắng xóa)."""
    df = _build_results_df(ran, results, capacity, dp_val)

    disp = df.copy()
    disp["Trạng thái"] = disp["Trạng thái"].map({"OK": "✅ OK", "Lỗi": "❌ Lỗi"})
    disp["Value"]      = disp["Value"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
    disp["Weight"]     = disp["Weight"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
    disp["Fill %"]     = disp["Fill %"].apply(lambda x: f"{x:.1f}%" if pd.notna(x) else "—")
    disp["Items chọn"] = disp["Items chọn"].apply(lambda x: str(int(x)) if pd.notna(x) else "—")
    disp["Time (ms)"]  = disp["Time (ms)"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
    disp["Feasible"]   = disp["Feasible"].apply(
        lambda x: ("✅" if x else "❌") if pd.notna(x) else "—")
    disp["Ratio/DP"]   = disp["Ratio/DP"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "—")

    st.dataframe(disp, use_container_width=True, hide_index=True)

    torch_missing = [n for n in ran
                     if (results[n].get("note") or "").startswith("torch chưa cài")]
    if torch_missing:
        st.warning(f"**{', '.join(torch_missing)}** cần torch: "
                   f"`pip install torch torch-geometric`")


def _detailed_solver_table(ran: list, results: dict,
                            weights: np.ndarray, values: np.ndarray,
                            capacity: int, dp_val: Optional[int]):
    """Bảng chi tiết từng solver — hiển thị dạng tabs, mỗi tab là 1 solver."""
    if not ran:
        return

    tabs = st.tabs([f"{'✅' if results[n].get('value') is not None else '❌'} {n}"
                    for n in ran])

    for tab, name in zip(tabs, ran):
        with tab:
            r     = results[name]
            color = _solver_color(name)

            if r.get("value") is None:
                st.error(f"Lỗi: {r.get('note', 'unknown')}")
                continue

            value  = r["value"]
            weight = r.get("weight", 0) or 0
            sel    = r.get("selected", [])
            fill   = weight / capacity if capacity > 0 else 0

            # ---- Metrics row ----
            mc1, mc2, mc3, mc4, mc5 = st.columns(5)
            mc1.metric("Value",      f"{value:,}")
            mc2.metric("Weight",     f"{weight:,} / {capacity:,}")
            mc3.metric("Fill",       f"{fill*100:.1f}%")
            mc4.metric("Items chọn", len(sel))
            mc5.metric("Time",       f"{r['time_ms']:.2f} ms")

            col_a, col_b = st.columns([2, 1])
            col_a.progress(min(fill, 1.0))
            if dp_val and dp_val > 0:
                ratio = value / dp_val
                delta = f"{(ratio-1)*100:+.2f}%" if ratio < 1 else "đạt optimal"
                col_b.metric("Ratio vs DP", f"{ratio:.4f}", delta=delta)

            extra = []
            if r.get("note"):   extra.append(f"⚠ {r['note']}")
            if r.get("decode"): extra.append(f"decode={r['decode']}"
                                              + (f", top_m={r['top_m']}" if r.get("top_m") else ""))
            if extra:
                st.caption("  ·  ".join(extra))

            st.divider()

            # ---- Items table ----
            if not sel:
                st.info("Không có item nào được chọn.")
                continue

            sel_valid = [i for i in sel if i < len(weights)]
            rows = []
            for i in sel_valid:
                w_i = int(weights[i]); v_i = int(values[i])
                rows.append({
                    "Index":      i,
                    "Weight":     w_i,
                    "Value":      v_i,
                    "Ratio v/w":  round(v_i / max(w_i, 1e-8), 4),
                    "Chọn bởi":   name,
                })
            df_items = pd.DataFrame(rows)

            # Highlight: item nào có ratio v/w cao nhất
            st.markdown(
                f"**{len(sel_valid)} items được chọn** — "
                f"tổng weight `{int(weights[sel_valid].sum()):,}` / `{capacity:,}`, "
                f"tổng value `{int(values[sel_valid].sum()):,}`"
            )
            st.dataframe(
                df_items.drop(columns=["Chọn bởi"]),
                use_container_width=True, hide_index=True,
            )

            # So sánh với solver khác (delta value)
            others_ok = [n2 for n2 in ran
                         if n2 != name and results[n2].get("value") is not None]
            if others_ok:
                st.caption("So sánh với solver khác:")
                cmp_cols = st.columns(min(len(others_ok), 5))
                for col, other in zip(cmp_cols, others_ok):
                    ov   = results[other]["value"]
                    diff = value - ov
                    col.metric(other, f"{ov:,}", delta=f"{diff:+,}")


def _save_results_buttons(ran: list, results: dict, capacity: int, dp_val: Optional[int],
                          weights=None, values=None, key_prefix: str = "live"):
    """Nút lưu kết quả dưới dạng CSV và JSON."""
    df = _build_results_df(ran, results, capacity, dp_val)

    # CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")

    # JSON chi tiết — bao gồm instance data để tự đủ khi upload vào tab So sánh
    detail = {}
    if weights is not None and values is not None:
        detail["_instance"] = {
            "weights":  [int(x) for x in weights],
            "values":   [int(x) for x in values],
            "capacity": int(capacity),
        }
    for name in ran:
        r = results[name]
        detail[name] = {
            "value":       r.get("value"),
            "weight":      r.get("weight"),
            "capacity":    r.get("capacity", capacity),
            "fill_pct":    round(float(r.get("weight", 0) / capacity * 100), 1) if capacity > 0 else None,
            "n_selected":  len(r.get("selected", [])),
            "selected":    r.get("selected", []),
            "time_ms":     r.get("time_ms"),
            "feasible":    r.get("feasible"),
            "ratio_vs_dp": round(r["value"] / dp_val, 6)
                           if r.get("value") and dp_val and dp_val > 0 else None,
            "decode":      r.get("decode"),
            "top_m":       r.get("top_m"),
            "note":        r.get("note"),
        }
    def _np_default(o):
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")
    json_bytes = json.dumps(detail, indent=2, ensure_ascii=False, default=_np_default).encode("utf-8")

    sa, sb, sc, _ = st.columns([1, 1, 1, 3])
    sa.download_button("⬇ CSV", data=csv_bytes,
                       file_name="knapsack_results.csv", mime="text/csv",
                       use_container_width=True, key=f"{key_prefix}_dl_csv")
    sb.download_button("⬇ JSON", data=json_bytes,
                       file_name="knapsack_results.json", mime="application/json",
                       use_container_width=True, key=f"{key_prefix}_dl_json")
    if sc.button("💾 Lưu vào results/", use_container_width=True, key=f"{key_prefix}_btn_save_res"):
        out_dir = ROOT / "results" / "live_demo"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        (out_dir / f"results_{ts}.csv").write_bytes(csv_bytes)
        (out_dir / f"results_{ts}.json").write_bytes(json_bytes)
        st.success(f"Đã lưu vào `results/live_demo/results_{ts}.*`")


# ---------------------------------------------------------------------------
# TAB 2 — Live Demo
# ---------------------------------------------------------------------------
def tab_live_demo():
    st.header("Live Demo — Sinh dữ liệu & Chạy solver")

    with st.sidebar:
        st.subheader("Cấu hình instance")
        n_items = st.number_input("Số item (n)", min_value=1, max_value=1000, value=30, step=1)
        cap_pct = st.slider("Capacity (% tổng weight)", 10, 90, 50, step=5) / 100.0
        seed    = st.number_input("Seed", value=42, step=1)

        st.subheader("Chọn solver")
        sel_dp     = st.checkbox("DP (Exact)",    value=True)
        sel_greedy = st.checkbox("Greedy",        value=True)
        sel_ga     = st.checkbox("GA",            value=True)
        sel_bb     = st.checkbox("B&B",           value=False)
        st.caption("GNN variants:")
        sel_gnn_knn      = st.checkbox("GNN-kNN",      value=True)
        sel_gnn_conflict = st.checkbox("GNN-Conflict",  value=False)
        sel_gnn_random   = st.checkbox("GNN-Random",    value=False)
        sel_gnn_full     = st.checkbox("GNN-Full",      value=False)
        sel_dqn          = st.checkbox("DQN",           value=False)
        sel_s2v          = st.checkbox("S2V-DQN",       value=False)
        sel_reinforce    = st.checkbox("REINFORCE",     value=False)
        sel_rf_hard = st.checkbox("REINFORCE-hard", value=False, key="_sel_rf_hard")
        sel_rf_none = st.checkbox("REINFORCE-none", value=False, key="_sel_rf_none")
        sel_rf_polyak = st.checkbox("REINFORCE-polyak", value=False, key="_sel_rf_polyak")

        st.subheader("GNN / REINFORCE settings")
        gnn_decode = st.selectbox("Decode Method",
            ["greedy", "dp_subset", "beam_search"],
            format_func=lambda x: {
                "greedy": "Greedy (nhanh)",
                "dp_subset": "DP Subset (chính xác hơn)",
                "beam_search": "Beam Search",
            }[x])
        gnn_top_m   = 30
        beam_width  = 5
        if gnn_decode == "dp_subset":
            gnn_top_m = st.slider("top_m", 10, 700, 30, step=5)
        elif gnn_decode == "beam_search":
            beam_width = st.slider("beam_width", 2, 20, 5, step=1)

        gnn_dir = ROOT / "results" / "ablation"
        with st.expander("Model paths"):
            gnn_model_knn      = st.text_input("GNN-kNN model",
                value=str(gnn_dir / "gnn_knn" / "gnn_best.pt"), key="_mp_knn")
            gnn_model_conflict = st.text_input("GNN-Conflict model",
                value=str(gnn_dir / "gnn_conflict_static" /"gnn_best.pt"), key="_mp_conflict")
            gnn_model_random   = st.text_input("GNN-Random model",
                value=str(gnn_dir / "gnn_random" /"gnn_best.pt"), key="_mp_random")
            gnn_model_full     = st.text_input("GNN-Full model",
                value=str(gnn_dir / "gnn_full" /"gnn_best.pt"), key="_mp_full")

        st.subheader("Khác")
        dqn_model_path = st.text_input("DQN model",
            value=str(ROOT / "results" / "DQN" / "dqn_best.pt"))
        s2v_graph_type = st.selectbox("S2V graph type",
            ["knn", "conflict_static", "random", "full"], key="_s2v_gt")
        _s2v_default_paths = {
            "knn":             str(ROOT / "results" / "S2V_DQN" / "s2v_dqn_knn"      / "s2v_dqn_knn_best.pt"),
            "conflict_static": str(ROOT / "results" / "S2V_DQN" / "s2v_dqn_conflict" / "s2v_dqn_conflict_static_best.pt"),
            "random":          str(ROOT / "results" / "S2V_DQN" / "s2v_dqn_random"   / "s2v_dqn_random_best.pt"),
            "full":            str(ROOT / "results" / "S2V_DQN" / "s2v_dqn_full"     / "s2v_dqn_full_best.pt"),
        }
        s2v_model_path = st.text_input("S2V-DQN model",
            value=_s2v_default_paths[s2v_graph_type], key="_s2v_mp")
        # # REINFORCE: hiện chỉ có model train trên kNN graph (3 baseline variants)
        # reinforce_graph_type = st.selectbox("REINFORCE graph type",
        #     ["knn"], key="_rf_gt",
        #     help="Chỉ kNN có checkpoint sẵn. Các graph khác chưa được train.")
        # reinforce_baseline = st.selectbox("REINFORCE baseline",
        #     ["hard", "none", "polyak"], key="_rf_bl",
        #     help="hard / none / polyak = 3 phương án baseline trong REINFORCE training.")
        # RF_DIR = ROOT / "results" / "GNN_REINFORCE"
        # _rf_default_paths = {
        #     ("knn", "hard"):    str(RF_DIR / "gnn_reinforce_knn_hard"   / "gnn_reinforce_knn_hard.pt"),
        #     ("knn", "none"):    str(RF_DIR / "gnn_reinforce_knn_none"   / "gnn_reinforce_knn_none.pt"),
        #     ("knn", "polyak"):  str(RF_DIR / "gnn_reinforce_knn_polyak" / "gnn_reinforce_knn_polyak.pt"),
        # }
        # reinforce_model_path = st.text_input("REINFORCE model",
        #     value=_rf_default_paths[(reinforce_graph_type, reinforce_baseline)],
        #     key="_rf_mp")
        bb_timeout = st.slider("B&B timeout (s)", 1, 60, 10)

    selected_solvers = []
    if sel_dp:           selected_solvers.append("DP")
    if sel_greedy:       selected_solvers.append("Greedy")
    if sel_ga:           selected_solvers.append("GA")
    if sel_bb:           selected_solvers.append("BB")
    if sel_gnn_knn:      selected_solvers.append("GNN-kNN")
    if sel_gnn_conflict: selected_solvers.append("GNN-Conflict")
    if sel_gnn_random:   selected_solvers.append("GNN-Random")
    if sel_gnn_full:     selected_solvers.append("GNN-Full")
    if sel_dqn:          selected_solvers.append("DQN")
    if sel_s2v:          selected_solvers.append("S2V-DQN")
    if sel_reinforce:    selected_solvers.append("REINFORCE")
    if sel_rf_hard:   selected_solvers.append("REINFORCE-hard")
    if sel_rf_none:   selected_solvers.append("REINFORCE-none")
    if sel_rf_polyak: selected_solvers.append("REINFORCE-polyak")

    if not selected_solvers:
        st.warning("Chọn ít nhất 1 solver ở sidebar."); return

    cfg = {
        "gnn_model_default":        gnn_model_knn,
        "gnn_model_knn":            gnn_model_knn,
        "gnn_model_conflict_static":gnn_model_conflict,
        "gnn_model_random":         gnn_model_random,
        "gnn_model_full":           gnn_model_full,
        "gnn_decode":               gnn_decode,
        "gnn_top_m":                gnn_top_m,
        "beam_width":               beam_width,
        "dqn_model":                dqn_model_path,
        "s2v_model":                s2v_model_path,
        "s2v_graph_type":           s2v_graph_type,
        # "reinforce_model":          reinforce_model_path,
        # "reinforce_graph_type":     reinforce_graph_type,
        "bb_timeout":               bb_timeout,
    }

    # --- Nguồn instance ---
    using_ext = "_live_ext_weights" in st.session_state
    if using_ext:
        weights  = st.session_state["_live_ext_weights"]
        values   = st.session_state["_live_ext_values"]
        capacity = st.session_state["_live_ext_capacity"]
        n_show   = len(weights)
        st.info(f"Đang dùng instance từ file/dataset — n={n_show}, capacity={capacity:,}")
        if st.button("Quay lại instance sinh ngẫu nhiên"):
            for k in ["_live_ext_weights","_live_ext_values","_live_ext_capacity"]:
                st.session_state.pop(k, None)
            st.session_state.pop("_live_results", None)
            st.rerun()
    else:
        weights, values, capacity = generate_instance(int(n_items), int(seed), cap_pct)
        n_show = int(n_items)

    # Lưu instance hiện tại để tab So sánh có thể đọc
    st.session_state["_live_weights"]  = weights
    st.session_state["_live_values"]   = values
    st.session_state["_live_capacity"] = int(capacity)

    # --- Hiển thị instance ---
    st.subheader("Instance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số item",     n_show)
    c2.metric("Capacity",    f"{capacity:,}")
    c3.metric("Tổng weight", f"{int(weights.sum()):,}")
    c4.metric("Tổng value",  f"{int(values.sum()):,}")

    with st.expander("Xem chi tiết item"):
        df_items = pd.DataFrame({"weight": weights, "value": values,
                                  "ratio v/w": (values/weights).round(3)})
        st.dataframe(df_items, use_container_width=True)

    # P1: Sinh & Lưu data
    with st.expander("Sinh & Lưu data"):
        dl1, dl2 = st.columns(2)
        dl1.download_button("Tải về JSON",
            data=_export_instance_json(weights, values, capacity),
            file_name=f"knapsack_n{n_show}_s{seed}.json", mime="application/json",
            use_container_width=True)
        dl2.download_button("Tải về CSV",
            data=_export_instance_csv(weights, values),
            file_name=f"knapsack_n{n_show}_s{seed}.csv", mime="text/csv",
            use_container_width=True)
        st.divider()
        sv1, sv2 = st.columns([3, 1])
        save_dir  = sv1.text_input("Thư mục lưu",
            value=str(ROOT/"data"/"custom"), key="_save_dir")
        save_name = sv2.text_input("Tên file",
            value=f"knapsack_n{n_show}_s{seed}.json", key="_save_name")
        if st.button("Lưu vào disk", key="_btn_save_disk"):
            try:
                p = Path(save_dir) / save_name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(_export_instance_json(weights, values, capacity))
                st.success(f"Đã lưu: {p}")
            except Exception as e:
                st.error(f"Lỗi: {e}")

    # P1: Chọn từ dataset có sẵn
    with st.expander("Chọn instance từ dataset có sẵn"):
        data_dirs = _find_data_dirs()
        if not data_dirs:
            st.info("Không tìm thấy thư mục nào có NPZ trong `data/`.")
        else:
            dir_labels = [lbl for lbl, _ in data_dirs]
            chosen_lbl = st.selectbox("Dataset", dir_labels, key="_ds_dir")
            chosen_dir = dict(data_dirs)[chosen_lbl]
            npz_files  = sorted(chosen_dir.glob("instance_*.npz"))
            if not npz_files:
                st.warning("Không có instance_*.npz trong thư mục này.")
            else:
                file_names = [f.name for f in npz_files]
                chosen_file = st.selectbox(
                    f"Chọn file ({len(file_names)} instances)", file_names, key="_ds_file")
                if st.button("Nạp instance này", key="_btn_load_ds"):
                    try:
                        w, v, c, _, _ = _load_npz(chosen_dir / chosen_file)
                        st.session_state["_live_ext_weights"]  = w
                        st.session_state["_live_ext_values"]   = v
                        st.session_state["_live_ext_capacity"] = c
                        st.session_state.pop("_live_results", None)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Lỗi đọc NPZ: {e}")

    # P1: Kiểm tra data
    with st.expander("Kiểm tra data"):
        check_mode = st.radio("Chế độ", ["Upload file đơn", "Validate thư mục NPZ"],
                               horizontal=True, key="_check_mode")

        if check_mode == "Upload file đơn":
            uploaded = st.file_uploader("Chọn file (JSON / CSV / NPZ)", type=["json","csv","npz"],
                                        key="_uploader")
            if uploaded is not None:
                try:
                    if uploaded.name.endswith(".json"):
                        raw    = json.load(uploaded)
                        w_up   = np.array(raw["weights"],  dtype=np.int32)
                        v_up   = np.array(raw["values"],   dtype=np.int32)
                        cap_up = int(raw["capacity"])
                    elif uploaded.name.endswith(".csv"):
                        df_up  = pd.read_csv(uploaded)
                        w_up   = df_up["weight"].to_numpy(dtype=np.int32)
                        v_up   = df_up["value"].to_numpy(dtype=np.int32)
                        cap_up = st.number_input("Capacity (CSV không lưu capacity)",
                                                 min_value=1, value=int(w_up.sum()//2),
                                                 key="_csv_cap")
                    else:  # NPZ
                        import tempfile, os
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".npz") as tmp:
                            tmp.write(uploaded.read()); tmp_path = tmp.name
                        try:
                            w_up, v_up, cap_up, sol, opt = _load_npz(Path(tmp_path))
                        finally:
                            os.unlink(tmp_path)

                    n_up = len(w_up)
                    ic1,ic2,ic3,ic4 = st.columns(4)
                    ic1.metric("n",        n_up)
                    ic2.metric("Capacity", f"{cap_up:,}")
                    ic3.metric("Σ weight", f"{int(w_up.sum()):,}")
                    ic4.metric("Σ value",  f"{int(v_up.sum()):,}")

                    ratio_vw = v_up / np.maximum(w_up, 1)
                    stats_df = pd.DataFrame({
                        "": ["Min","Max","Mean","Std"],
                        "Weight":    [w_up.min(),  w_up.max(),  f"{w_up.mean():.1f}",  f"{w_up.std():.1f}"],
                        "Value":     [v_up.min(),  v_up.max(),  f"{v_up.mean():.1f}",  f"{v_up.std():.1f}"],
                        "Ratio v/w": [f"{ratio_vw.min():.3f}", f"{ratio_vw.max():.3f}",
                                      f"{ratio_vw.mean():.3f}", f"{ratio_vw.std():.3f}"],
                    })
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)

                    warns, infos = _validate_instance(w_up, v_up, cap_up)
                    for w in warns: st.warning(w)
                    for info in infos: st.info(info)
                    if not warns: st.success("Instance hợp lệ.")

                    fig, axes = plt.subplots(1, 3, figsize=(13, 3))
                    axes[0].hist(w_up,     bins=min(30,n_up), color="#7F77DD", edgecolor="white")
                    axes[0].set_title("Phân phối Weight")
                    axes[1].hist(v_up,     bins=min(30,n_up), color="#1D9E75", edgecolor="white")
                    axes[1].set_title("Phân phối Value")
                    axes[2].hist(ratio_vw, bins=min(30,n_up), color="#D85A30", edgecolor="white")
                    axes[2].set_title("Phân phối Ratio v/w")
                    for ax in axes: ax.grid(alpha=0.25)
                    plt.tight_layout(); st.pyplot(fig); plt.close()

                    if st.button("Dùng instance này để chạy solver", type="primary",
                                 key="_btn_load_ext"):
                        st.session_state["_live_ext_weights"]  = w_up
                        st.session_state["_live_ext_values"]   = v_up
                        st.session_state["_live_ext_capacity"] = cap_up
                        st.session_state.pop("_live_results", None)
                        st.rerun()
                except Exception as e:
                    st.error(f"Lỗi đọc file: {e}")

        else:  # Validate thư mục NPZ
            data_dirs = _find_data_dirs()
            all_dirs  = [("Nhập đường dẫn thủ công", None)] + data_dirs
            dir_lbl   = st.selectbox("Chọn thư mục", [l for l,_ in all_dirs], key="_val_dir")
            if dir_lbl == "Nhập đường dẫn thủ công":
                val_dir = Path(st.text_input("Đường dẫn", key="_val_dir_custom"))
            else:
                val_dir = dict(all_dirs)[dir_lbl]

            max_check = st.number_input("Kiểm tra tối đa N files (0=tất cả)", min_value=0, value=100)
            if st.button("Validate", type="primary", key="_btn_validate"):
                if not val_dir or not val_dir.exists():
                    st.error("Thư mục không tồn tại.")
                else:
                    files = sorted(val_dir.glob("instance_*.npz"))
                    if max_check > 0: files = files[:int(max_check)]
                    if not files:
                        st.warning("Không tìm thấy instance_*.npz")
                    else:
                        results_val = []
                        prog = st.progress(0)
                        for i, fp in enumerate(files):
                            results_val.append(_validate_npz(fp))
                            prog.progress((i+1)/len(files))
                        prog.empty()

                        n_ok  = sum(1 for r in results_val if r["ok"])
                        n_err = len(results_val) - n_ok
                        vc1,vc2,vc3 = st.columns(3)
                        vc1.metric("Tổng", len(results_val))
                        vc2.metric("OK", n_ok, delta=None)
                        vc3.metric("Lỗi", n_err, delta=None)

                        df_val = pd.DataFrame([
                            {"File": r["file"], "n": r.get("n","?"),
                             "Cap": r.get("capacity","?"),
                             "Fill": r.get("fill_ratio","?"),
                             "Opt": r.get("opt_value","?"),
                             "Status": "✅" if r["ok"] else "❌ "+"; ".join(r["errors"][:1])}
                            for r in results_val
                        ])
                        st.dataframe(df_val, use_container_width=True, hide_index=True)

    # --- Session state: reset khi instance thay đổi ---
    inst_key = "ext" if using_ext else f"{n_items}_{seed}_{cap_pct:.3f}"
    if st.session_state.get("_live_inst_key") != inst_key:
        st.session_state["_live_results"]  = {}
        st.session_state["_live_inst_key"] = inst_key
    if "_live_results" not in st.session_state:
        st.session_state["_live_results"] = {}
    results: dict = st.session_state["_live_results"]

    # --- Nút chạy ---
    st.subheader("Chạy solver")
    btn_cols = st.columns([2] + [1] * len(selected_solvers))

    if btn_cols[0].button("Chạy tất cả", type="primary", use_container_width=True):
        prog = st.progress(0, text="Bắt đầu...")
        acc  = dict(st.session_state["_live_results"])
        for i, name in enumerate(selected_solvers):
            rkey = _result_key(name, cfg)
            prog.progress((i + 1) / len(selected_solvers), text=f"Đang chạy {rkey}...")
            r = SOLVER_FN[name](weights, values, capacity, cfg)
            r["capacity"] = capacity
            acc[rkey] = r
        prog.empty()
        st.session_state["_live_results"] = acc
        st.rerun()

    for idx, name in enumerate(selected_solvers):
        rkey = _result_key(name, cfg)
        if btn_cols[idx + 1].button(f"Chạy {rkey}", use_container_width=True,
                                    key=f"_btn_{name}"):
            with st.spinner(f"Đang chạy {rkey}..."):
                r = SOLVER_FN[name](weights, values, capacity, cfg)
                r["capacity"] = capacity
            st.session_state["_live_results"] = {
                **st.session_state["_live_results"], rkey: r
            }
            st.rerun()

    # --- Hiển thị kết quả ---
    results = st.session_state["_live_results"]
    ran     = _sort_results_keys(list(results.keys()))
    if not ran:
        st.info("Nhấn nút solver để chạy."); return

    dp_val = results.get("DP", {}).get("value")

    # 1. Bảng tổng hợp
    st.subheader(f"Bảng kết quả ({len(ran)} solver)")
    _results_table(ran, results, capacity, dp_val)

    # Lưu bảng kết quả
    _save_results_buttons(ran, results, capacity, dp_val, weights=weights, values=values, key_prefix="live")

    # 2. Bảng chi tiết items từng solver
    with st.expander("Chi tiết items được chọn (từng solver)", expanded=True):
        _detailed_solver_table(ran, results, weights, values, capacity, dp_val)

    # 3. Cards tóm tắt
    with st.expander("Cards tóm tắt", expanded=False):
        ncols = min(len(ran), 3)
        for i in range(0, len(ran), ncols):
            group = ran[i:i + ncols]
            cols  = st.columns(ncols)
            for col, name in zip(cols, group):
                _result_card(col, name, results[name], dp_val)

    # 3. Biểu đồ
    with st.expander("Biểu đồ so sánh", expanded=True):
        _live_comparison_chart(results, capacity)

    # Nút xóa
    if st.button("🗑 Xóa tất cả kết quả", key="_btn_clear"):
        st.session_state["_live_results"] = {}
        st.rerun()

# ---------------------------------------------------------------------------
# TAB 3 — So sánh
# ---------------------------------------------------------------------------
def _compare_single_instance(results: dict, capacity: int,
                              weights=None, values=None, source_label: str = ""):
    """Hiển thị so sánh single-instance — tái sử dụng helpers từ Live Demo."""
    ran    = _sort_results_keys(list(results.keys()))
    dp_val = results.get("DP", {}).get("value")

    if source_label:
        st.caption(f"Nguồn: {source_label}  ·  {len(ran)} solver  ·  capacity={capacity:,}")

    # Bộ lọc solver để chọn subset cần so sánh
    chosen = st.multiselect(
        "Chọn solver cần so sánh (bỏ trống = tất cả)",
        options=ran, default=ran, key="_cmp_chosen",
        format_func=lambda k: k,
    )
    if not chosen:
        st.info("Chọn ít nhất 1 solver."); return
    results_sub = {k: results[k] for k in chosen}
    ran_sub     = chosen

    st.subheader("Bảng kết quả")
    _results_table(ran_sub, results_sub, capacity, dp_val)
    _save_results_buttons(ran_sub, results_sub, capacity, dp_val, weights=weights, values=values, key_prefix="cmp")

    if weights is not None and values is not None:
        with st.expander("Chi tiết items được chọn", expanded=False):
            _detailed_solver_table(ran_sub, results_sub,
                                   np.asarray(weights), np.asarray(values),
                                   capacity, dp_val)

    with st.expander("Biểu đồ so sánh", expanded=True):
        _live_comparison_chart(results_sub, capacity)


def tab_scatter():
    st.header("So sánh solver")

    mode = st.radio(
        "Chế độ so sánh",
        ["📡 Single instance (từ Live Demo hoặc file JSON)",
         "📊 Multi-instance (scatter — merged_results.csv)"],
        horizontal=True, key="_scatter_mode",
    )
    st.divider()

    # ------------------------------------------------------------------ #
    # MODE A — Single instance                                            #
    # ------------------------------------------------------------------ #
    if mode.startswith("📡"):
        source = st.radio(
            "Nguồn dữ liệu",
            ["Session Live Demo (đang chạy)", "Upload file JSON", "Chọn file từ results/live_demo/"],
            horizontal=True, key="_cmp_src",
        )

        results, capacity, weights, values, label = {}, 0, None, None, ""

        if source == "Session Live Demo (đang chạy)":
            live = st.session_state.get("_live_results", {})
            if not live:
                st.info("Chưa có kết quả. Chạy solver ở tab **Live Demo** trước."); return
            results  = live
            capacity = st.session_state.get("_live_capacity", 0)
            weights  = st.session_state.get("_live_weights")
            values   = st.session_state.get("_live_values")
            label    = f"Live Demo session — {st.session_state.get('_live_inst_key','')}"

        elif source == "Upload file JSON":
            uploaded = st.file_uploader(
                "Upload file JSON đã tải từ Live Demo", type=["json"],
                key="_cmp_upload",
            )
            if uploaded is None:
                st.info("Tải lên file JSON (⬇ JSON từ Live Demo → tab này)."); return
            try:
                raw = json.loads(uploaded.read().decode("utf-8"))
            except Exception as e:
                st.error(f"Lỗi đọc JSON: {e}"); return
            inst = raw.pop("_instance", None)
            if inst:
                weights  = np.array(inst["weights"],  dtype=np.int32)
                values   = np.array(inst["values"],   dtype=np.int32)
                capacity = int(inst["capacity"])
            else:
                capacity = next((v.get("capacity", 0) for v in raw.values() if isinstance(v, dict)), 0)
            # Chuyển JSON dict → results format
            results = {}
            for k, v in raw.items():
                if not isinstance(v, dict): continue
                results[k] = {
                    "value":    v.get("value"),
                    "weight":   v.get("weight", 0),
                    "capacity": v.get("capacity", capacity),
                    "selected": v.get("selected", []),
                    "time_ms":  v.get("time_ms", 0),
                    "feasible": v.get("feasible", True),
                    "decode":   v.get("decode"),
                    "top_m":    v.get("top_m"),
                    "note":     v.get("note"),
                }
            label = f"File: {uploaded.name}"

        else:  # Chọn file từ results/live_demo/
            live_dir = ROOT / "results" / "live_demo"
            json_files = sorted(live_dir.glob("results_*.json"), reverse=True) if live_dir.exists() else []
            if not json_files:
                st.warning(f"Chưa có file JSON trong `results/live_demo/`. Nhấn **Lưu vào results/** ở Live Demo trước."); return
            chosen_file = st.selectbox(
                f"Chọn file ({len(json_files)} files)",
                json_files,
                format_func=lambda p: p.name,
                key="_cmp_file",
            )
            try:
                raw = json.loads(chosen_file.read_text(encoding="utf-8"))
            except Exception as e:
                st.error(f"Lỗi đọc file: {e}"); return
            inst = raw.pop("_instance", None)
            if inst:
                weights  = np.array(inst["weights"],  dtype=np.int32)
                values   = np.array(inst["values"],   dtype=np.int32)
                capacity = int(inst["capacity"])
            else:
                capacity = next((v.get("capacity", 0) for v in raw.values() if isinstance(v, dict)), 0)
            results = {}
            for k, v in raw.items():
                if not isinstance(v, dict): continue
                results[k] = {
                    "value":    v.get("value"),
                    "weight":   v.get("weight", 0),
                    "capacity": v.get("capacity", capacity),
                    "selected": v.get("selected", []),
                    "time_ms":  v.get("time_ms", 0),
                    "feasible": v.get("feasible", True),
                    "decode":   v.get("decode"),
                    "top_m":    v.get("top_m"),
                    "note":     v.get("note"),
                }
            label = f"File: {chosen_file.name}"

        if results:
            _compare_single_instance(results, capacity, weights=weights, values=values,
                                     source_label=label)
        return

    # ------------------------------------------------------------------ #
    # MODE B — Multi-instance scatter (unchanged)                         #
    # ------------------------------------------------------------------ #
    merged_path = ROOT / "results" / "compare_small" / "merged_results.csv"
    if not merged_path.exists():
        st.warning("Chưa có `merged_results.csv`. Chạy Merge_results.py trước.")
        return

    df = pd.read_csv(merged_path)
    available = [s for s in ["dp","gnn","greedy","ga","bb","dqn","s2v","reinforce"]
                 if f"{s}_value" in df.columns]
    # dp dùng column dp_value — chỉ thêm nếu column thực sự tồn tại
    if not available:
        st.warning("merged_results.csv không có column *_value nào hợp lệ."); return
    solver_labels = {"dp":"DP","gnn":"GNN","greedy":"Greedy","ga":"GA","bb":"BB",
                     "dqn":"DQN","s2v":"S2V-DQN","reinforce":"REINFORCE"}
    # Map màu theo short key (SOLVER_COLOR dùng display name nên không dùng trực tiếp được)
    _SCATTER_COLOR = {
        "dp":       "#5F5E5A",
        "gnn":      "#D85A30",
        "greedy":   "#7F77DD",
        "ga":       "#1D9E75",
        "bb":       "#888888",
        "dqn":      "#F5A623",
        "s2v":      "#E8A33D",
        "reinforce":"#E74C3C",
    }

    col1, col2 = st.columns([1, 2])
    with col1:
        safe_default = [s for s in ["gnn","dp"] if s in available]
        chosen = st.multiselect("Chọn solver cần so sánh", options=available,
            default=safe_default, format_func=lambda x: solver_labels.get(x, x.upper()))
        plot_btn = st.button("Vẽ biểu đồ", type="primary")

    if not plot_btn or len(chosen) < 2:
        st.info("Chọn ít nhất 2 solver rồi nhấn Vẽ biểu đồ."); return

    def _get(s):
        col = "dp_value" if s == "dp" else f"{s}_value"
        return pd.to_numeric(df.get(col), errors="coerce") if col in df.columns else None

    if len(chosen) == 2:
        a, b   = chosen
        va, vb = _get(a), _get(b)
        if va is None or vb is None: st.error("Thiếu data."); return
        mask   = va.notna() & vb.notna()
        va, vb = va[mask], vb[mask]
        lo = min(va.min(), vb.min()) * 0.98
        hi = max(va.max(), vb.max()) * 1.01
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(va, vb, color=_SCATTER_COLOR.get(b,"#999"), alpha=0.55, s=22,
                   edgecolors="white", linewidths=0.3)
        ax.plot([lo,hi],[lo,hi],"--",color="black",lw=1.2,alpha=0.5,label="Equal")
        a_wins = int((va > vb+1e-3).sum()); b_wins = int((vb > va+1e-3).sum())
        ties   = int(mask.sum()) - a_wins - b_wins
        ax.text(0.04,0.96,
                f"{solver_labels.get(a,a)} better: {a_wins}\n"
                f"{solver_labels.get(b,b)} better: {b_wins}\nTies: {ties}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round",facecolor="wheat",alpha=0.75))
        ax.set_xlabel(f"{solver_labels.get(a,a)} value",fontsize=11)
        ax.set_ylabel(f"{solver_labels.get(b,b)} value",fontsize=11)
        ax.set_title(f"{solver_labels.get(a,a)} vs {solver_labels.get(b,b)}")
        ax.legend(); ax.grid(alpha=0.25)
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        plt.tight_layout(); st.pyplot(fig); plt.close()
    else:
        dp_val = _get("dp")
        if dp_val is None: st.error("Cần có DP."); return
        targets = [s for s in chosen if s != "dp"]
        ncols = min(3, len(targets))
        nrows = (len(targets)+ncols-1)//ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols,5*nrows), squeeze=False)
        for idx, s in enumerate(targets):
            ax = axes[idx//ncols][idx%ncols]
            vs = _get(s)
            if vs is None: ax.set_visible(False); continue
            mask = dp_val.notna() & vs.notna()
            x, y = dp_val[mask], vs[mask]
            lo = min(x.min(),y.min())*0.98; hi = max(x.max(),y.max())*1.01
            color = _SCATTER_COLOR.get(s,"#999")
            ax.scatter(x, y, color=color, alpha=0.55, s=20, edgecolors="white", linewidths=0.3)
            ax.plot([lo,hi],[lo,hi],"--k",lw=1.1,alpha=0.5)
            pct = (y >= x-1e-3).mean()*100
            ax.text(0.04,0.96,f"Optimal rate: {pct:.1f}%",
                    transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(boxstyle="round",facecolor="wheat",alpha=0.75))
            ax.set_xlabel("DP value",fontsize=10); ax.set_ylabel(f"{solver_labels.get(s,s)}",fontsize=10)
            ax.set_title(f"{solver_labels.get(s,s)} vs DP",fontsize=12,fontweight="bold",color=color)
            ax.grid(alpha=0.25); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        for idx in range(len(targets), nrows*ncols):
            axes[idx//ncols][idx%ncols].set_visible(False)
        fig.suptitle("Solver value vs DP (per instance)",fontsize=14)
        plt.tight_layout(); st.pyplot(fig); plt.close()

# ---------------------------------------------------------------------------
# TAB 4 — Ablation & Cross-scale  (P2)
# ---------------------------------------------------------------------------
VARIANT_META = {
    "knn":             {"label":"GNN-kNN",     "color":"#D85A30","marker":"o"},
    "conflict_static": {"label":"GNN-Conflict","color":"#D4537E","marker":"^"},
    "random":          {"label":"GNN-Random",  "color":"#1D9E75","marker":"s"},
    "full":            {"label":"GNN-Full",    "color":"#378ADD","marker":"D"},
}

def tab_ablation():
    st.header("Ablation Study & Cross-scale Generalization")
    BASE = ROOT / "results" / "cross_scale"

    if not BASE.exists():
        st.warning(f"Thư mục `{BASE.relative_to(ROOT)}` chưa có. Chạy cross_scale_eval.py trước.")
        return

    graph_keys   = ["knn","conflict_static","random","full"]
    graph_labels = ["kNN","Conflict","Random","Full"]
    test_keys    = ["small","n100","n200"]
    test_labels  = ["n≤50","n=100","n=200"]
    colors       = [VARIANT_META[g]["color"] for g in graph_keys]

    # --- Chart 1: Ablation (Greedy vs DP Subset decode, tested on small) ---
    st.subheader("Ablation: Graph Structure × Decode Method (test n≤50)")
    dp_small = BASE / "dp_small.csv"
    if dp_small.exists():
        greedy_r, dp_r = [], []
        for g in graph_keys:
            gr, _  = _compute_ratio(BASE/f"{g}_on_small_greedy.csv", dp_small)
            dr, _  = _compute_ratio(BASE/f"{g}_on_small_dp.csv",     dp_small)
            greedy_r.append(gr if gr else 0)
            dp_r.append(dr if dr else 0)
        greedy_base, _ = _compute_ratio(BASE/"greedy_small.csv", dp_small)
        greedy_base_t = _load_avg_time(BASE / f"greedy_small.csv")

        # --- Chart 1a: Quality (Approximation Ratio) ---
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(graph_labels)); w = 0.35
        b1 = ax.bar(x - w/2, greedy_r, w, label="Greedy decode",   color="#4A90E2", alpha=0.85)
        b2 = ax.bar(x + w/2, dp_r,     w, label="DP subset decode", color="#D85A30", alpha=0.85)
        ax.axhline(1.0, color="#222", linestyle="-", lw=1.5, label="DP optimal (1.0)")
        if greedy_base:
            ax.axhline(greedy_base, color="gray", linestyle="--", lw=1.5,
                       label=f"Greedy heuristic ({greedy_base:.4f})")
        ax.set_xticks(x); ax.set_xticklabels(graph_labels, fontsize=11)
        ax.set_ylabel("Approximation Ratio (vs DP optimal)", fontsize=11)
        ax.set_title("Ablation: Graph Structure × Decode Method — Quality")
        valid_vals = [v for v in greedy_r + dp_r if v > 0]
        if valid_vals:
            ax.set_ylim(max(0.95, min(valid_vals) - 0.01), 1.008)
        ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(list(b1) + list(b2), greedy_r + dp_r):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
                        f"{val:.4f}", ha="center", fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # --- Chart 1b: Speed across 3 test sizes, 2 panels ---
        test_keys_sp  = ["small", "n100", "n200"]
        test_lbls_sp  = ["n≤50", "n=100", "n=200"]
        greedy_base_ts = [_load_avg_time(BASE / f"greedy_{t}.csv") for t in test_keys_sp]
        sp_markers = ["s", "o", "D", "^"]
        has_time = any(
            _load_avg_time(BASE / f"{g}_on_{t}_greedy.csv")
            for g in graph_keys for t in test_keys_sp
        )
        if has_time:
            fig2, (axL, axR) = plt.subplots(1, 2, figsize=(13, 4), sharey=True)
            for ax2, suffix, title in [
                (axL, "greedy", "(a) Greedy Decode"),
                (axR, "dp",    "(b) DP Subset Decode"),
            ]:
                for g, lbl, c, m in zip(graph_keys, graph_labels, colors, sp_markers):
                    ys = [_load_avg_time(BASE / f"{g}_on_{t}_{suffix}.csv") or 0
                          for t in test_keys_sp]
                    ax2.plot(test_lbls_sp, ys, marker=m, color=c,
                             linewidth=2, markersize=9, label=f"GNN-{lbl}")
                valid_gh = [(tl, t) for tl, t in zip(test_lbls_sp, greedy_base_ts) if t]
                if valid_gh:
                    ax2.plot([tl for tl, _ in valid_gh], [t for _, t in valid_gh],
                             marker="x", color="gray", linewidth=1.5, linestyle=":",
                             markersize=9, label="Greedy heuristic")
                ax2.set_yscale("log")
                ax2.yaxis.set_major_formatter(
                    __import__("matplotlib").ticker.FuncFormatter(lambda x, _: f"{x:.3g}")
                )
                ax2.set_xlabel("Test set (problem size)", fontsize=10)
                ax2.set_title(title, fontsize=11)
                ax2.legend(fontsize=9, loc="upper left")
                ax2.grid(axis="y", alpha=0.3)
            axL.set_ylabel("Avg Inference Time (ms, log scale)", fontsize=10)
            plt.suptitle("Speed: Graph Structure × Decode Method across Problem Sizes",
                         fontsize=12, y=1.01)
            plt.tight_layout(); st.pyplot(fig2); plt.close()
        # Tính thời gian trung bình cho từng graph variant (chỉ tập small)
        greedy_t = [_load_avg_time(BASE / f"{g}_on_small_greedy.csv") for g in graph_keys]
        dp_t = [_load_avg_time(BASE / f"{g}_on_small_dp.csv") for g in graph_keys]
        # Ablation table
        abl_rows = []
        for g, lbl, gr, dr, gt, dt in zip(graph_keys, graph_labels,
                                           greedy_r, dp_r, greedy_t, dp_t):
            abl_rows.append({
                "Graph":             lbl,
                "Greedy ratio":      f"{gr:.4f}" if gr else "MISSING",
                "DP subset ratio":   f"{dr:.4f}" if dr else "MISSING",
                "Δ ratio":           f"{(dr-gr):+.4f}" if gr and dr else "—",
                "Greedy time (ms)":  f"{gt:.2f}" if gt else "—",
                "DP subset time (ms)":f"{dt:.2f}" if dt else "—",
                "Time ×":            f"{dt/gt:.1f}×" if gt and dt else "—",
            })
        if greedy_base:
            abl_rows.append({
                "Graph": "(Greedy heuristic)",
                "Greedy ratio": f"{greedy_base:.4f}", "DP subset ratio": "—",
                "Δ ratio": "—",
                "Greedy time (ms)": f"{greedy_base_t:.2f}" if greedy_base_t else "—",
                "DP subset time (ms)": "—", "Time ×": "—",
            })
        st.dataframe(pd.DataFrame(abl_rows), use_container_width=True, hide_index=True)
    else:
        st.info(f"Chưa có `{dp_small.name}` trong cross_scale/.")

    st.divider()

    # --- Chart 2: Cross-scale generalization ---
    st.subheader("Cross-scale Generalization (trained n≤50, tested on larger n)")
    dp_files = {t: BASE/f"dp_{t}.csv" for t in test_keys}
    has_crossscale = all(p.exists() for p in dp_files.values())
    if has_crossscale:
        S2V_CS_META = {
            "knn":            {"label": "S2V-kNN",     "color": "#9B59B6", "marker": "o", "ls": "--"},
            "conflict_static":{"label": "S2V-Conflict","color": "#7D3C98", "marker": "^", "ls": "--"},
            "random":         {"label": "S2V-Random",  "color": "#BB8FCE", "marker": "s", "ls": "--"},
            "full":           {"label": "S2V-Full",    "color": "#D2B4DE", "marker": "D", "ls": "--"},
        }
        RL_SINGLE_APP = {
            "dqn":       {"label": "DQN",      "color": "#F5A623", "marker": "p", "ls": "-."},
        }
        # REINFORCE has multiple baseline variants: reinforce_<baseline>_on_<scale>.csv
        REINFORCE_BASELINES = [
            ("hard",   {"label": "REINFORCE-hard",  "color": "#E74C3C", "marker": "*", "ls": "-."}),
            ("none",   {"label": "REINFORCE-none",  "color": "#C0392B", "marker": "X", "ls": "-."}),
            ("polyak", {"label": "REINFORCE-polyak","color": "#922B21", "marker": "P", "ls": "-."}),
        ]

        def _ratio_list(fmt, keys=test_keys):
            return [(_compute_ratio(BASE/fmt.format(t=t), dp_files[t])[0] or 0) for t in keys]

        greedy_ys = _ratio_list("greedy_{t}.csv")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharey=False)

        # Panel (a): GNN variants
        for g in graph_keys:
            ys = _ratio_list(f"{g}_on_{{t}}_greedy.csv")
            m  = VARIANT_META[g]
            ax1.plot(test_labels, ys, marker=m["marker"], color=m["color"],
                     label=m["label"], linewidth=2, markersize=9)
        ax1.plot(test_labels, greedy_ys, "--x", color="#7F77DD",
                 label="Greedy heuristic", linewidth=1.5, markersize=8)
        ax1.set_xlabel("Test set (problem size)", fontsize=11)
        ax1.set_ylabel("Approximation Ratio (vs DP)", fontsize=11)
        ax1.set_title("(a) GNN — Graph Structure Ablation", fontsize=11)
        ax1.legend(fontsize=9, loc="lower left"); ax1.grid(axis="y", alpha=0.3)

        # Panel (b): cross-method (GNN best + S2V×4 + DQN + REINFORCE + Greedy)
        gnn_best = [max(
            (_compute_ratio(BASE/f"{g}_on_{t}_greedy.csv", dp_files[t])[0] or 0)
            for g in graph_keys) for t in test_keys]
        ax2.plot(test_labels, gnn_best, marker="*", linestyle="-",
                 label="GNN (best)", color="#D85A30", linewidth=2.5, markersize=11)

        for gt, meta in S2V_CS_META.items():
            ys = []
            for t in test_keys:
                r, _ = _compute_ratio(BASE/f"s2v_{gt}_on_{t}.csv", dp_files[t])
                ys.append(r or 0)
            if any(y > 0 for y in ys):
                ax2.plot(test_labels, ys, marker=meta["marker"], color=meta["color"],
                         label=meta["label"], linewidth=1.5, markersize=7, linestyle=meta["ls"])

        for key, meta in RL_SINGLE_APP.items():
            ys = []
            for t in test_keys:
                r, _ = _compute_ratio(BASE/f"{key}_on_{t}.csv", dp_files[t])
                ys.append(r or 0)
            if any(y > 0 for y in ys):
                ax2.plot(test_labels, ys, marker=meta["marker"], color=meta["color"],
                         label=meta["label"], linewidth=2, markersize=9, linestyle=meta["ls"])

        # REINFORCE — iterate 3 baseline variants
        for baseline, meta in REINFORCE_BASELINES:
            ys = []
            for t in test_keys:
                r, _ = _compute_ratio(BASE/f"reinforce_{baseline}_on_{t}.csv", dp_files[t])
                ys.append(r or 0)
            if any(y > 0 for y in ys):
                ax2.plot(test_labels, ys, marker=meta["marker"], color=meta["color"],
                         label=meta["label"], linewidth=2, markersize=9, linestyle=meta["ls"])

        ax2.plot(test_labels, greedy_ys, "--x", color="#7F77DD",
                 label="Greedy heuristic", linewidth=1.5, markersize=8)
        ax2.set_xlabel("Test set (problem size)", fontsize=11)
        ax2.set_ylabel("Approximation Ratio (vs DP)", fontsize=11)
        ax2.set_title("(b) Cross-method Comparison", fontsize=11)
        ax2.legend(fontsize=9, loc="lower left", ncol=2); ax2.grid(axis="y", alpha=0.3)

        plt.suptitle("Cross-scale Generalization (trained on n≤50, tested on larger n)",
                     fontsize=13, y=1.01)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Cross-scale table
        cs_rows = []
        for g, lbl in zip(graph_keys, graph_labels):
            row = {"Model": f"GNN-{lbl}"}
            for t, tl in zip(test_keys, test_labels):
                r, _ = _compute_ratio(BASE/f"{g}_on_{t}_greedy.csv", dp_files[t])
                row[tl] = f"{r:.4f}" if r else "—"
            cs_rows.append(row)
        for gt, meta in S2V_CS_META.items():
            if any((BASE/f"s2v_{gt}_on_{t}.csv").exists() for t in test_keys):
                row = {"Model": meta["label"]}
                for t, tl in zip(test_keys, test_labels):
                    r, _ = _compute_ratio(BASE/f"s2v_{gt}_on_{t}.csv", dp_files[t])
                    row[tl] = f"{r:.4f}" if r else "—"
                cs_rows.append(row)
        for key, meta in RL_SINGLE_APP.items():
            if any((BASE/f"{key}_on_{t}.csv").exists() for t in test_keys):
                row = {"Model": meta["label"]}
                for t, tl in zip(test_keys, test_labels):
                    r, _ = _compute_ratio(BASE/f"{key}_on_{t}.csv", dp_files[t])
                    row[tl] = f"{r:.4f}" if r else "—"
                cs_rows.append(row)
        for baseline, meta in REINFORCE_BASELINES:
            if any((BASE/f"reinforce_{baseline}_on_{t}.csv").exists() for t in test_keys):
                row = {"Model": meta["label"]}
                for t, tl in zip(test_keys, test_labels):
                    r, _ = _compute_ratio(BASE/f"reinforce_{baseline}_on_{t}.csv", dp_files[t])
                    row[tl] = f"{r:.4f}" if r else "—"
                cs_rows.append(row)
        row = {"Model": "Greedy heuristic"}
        for t, tl in zip(test_keys, test_labels):
            r, _ = _compute_ratio(BASE/f"greedy_{t}.csv", dp_files[t])
            row[tl] = f"{r:.4f}" if r else "—"
        cs_rows.append(row)
        st.dataframe(pd.DataFrame(cs_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Thiếu một số file dp_small/n100/n200.csv trong cross_scale/.")

    st.divider()


    # ---------------------------------------------------------------------------
    # Pisinger T3 Ablation (4 graph variants × 2 decode methods)
    # ---------------------------------------------------------------------------
    st.subheader("Pisinger Type 3 — Ablation: Graph Structure × Decode Method")

    PIS_DIR = ROOT / "results" / "cross_pisinger"
    pis_variant_files = {
        "kNN":      (PIS_DIR / "pisinger_kNN_on_test.csv",
                     PIS_DIR / "pisinger_kNN_on_test_dp.csv"),
        "Full":     (PIS_DIR / "pisinger_full_on_test.csv",
                     PIS_DIR / "pisinger_full_on_test_dp.csv"),
        "Conflict": (PIS_DIR / "pisinger_conflict_static_on_test.csv",
                     PIS_DIR / "pisinger_conflict_static_on_test_dp.csv"),
        "Random":   (PIS_DIR / "pisinger_random_on_test.csv",
                     PIS_DIR / "pisinger_random_on_test_dp.csv"),
    }
    pis_colors = {"kNN": "#D85A30", "Full": "#378ADD", "Conflict": "#D4537E", "Random": "#1D9E75"}

    def _pis_load(path: Path) -> list:
        if not path.exists():
            return []
        try:
            with open(path, newline="", encoding="utf-8") as f:
                return list(csv.DictReader(f))
        except Exception:
            return []

    def _pis_ratio(rows: list) -> list:
        """Return per-instance ratio list using 'ratio' column or dp_value."""
        out = []
        for r in rows:
            try:
                if "ratio" in r and float(r["ratio"]) > 0:
                    out.append(float(r["ratio"]))
                elif "dp_value" in r and float(r["dp_value"]) > 0:
                    out.append(float(r["total_value"]) / float(r["dp_value"]))
            except (ValueError, ZeroDivisionError):
                pass
        return out

    # ---- Load all 8 files ----
    pis_data = {}
    any_pis_found = False
    for lbl, (gf, df_) in pis_variant_files.items():
        g_rows = _pis_load(gf)
        d_rows = _pis_load(df_)
        if g_rows or d_rows:
            any_pis_found = True
        pis_data[lbl] = {
            "greedy": _pis_ratio(g_rows),
            "dp":     _pis_ratio(d_rows),
        }

    if not any_pis_found:
        st.info(
            f"Chưa có CSV kết quả Pisinger T3 trong `{PIS_DIR.relative_to(ROOT)}/`. "
            "Đặt 8 file CSV kết quả vào thư mục đó rồi reload."
        )
    else:
        from statistics import mean as _mean, stdev as _stdev

        variant_labels = list(pis_data.keys())
        greedy_means = [_mean(pis_data[l]["greedy"]) if pis_data[l]["greedy"] else 0
                        for l in variant_labels]
        dp_means     = [_mean(pis_data[l]["dp"])     if pis_data[l]["dp"]     else 0
                        for l in variant_labels]

        # ---- Chart A: Grouped bar (Greedy decode vs DP subset, 4 variants) ----
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(variant_labels)); w = 0.35
        b1 = ax.bar(x - w/2, greedy_means, w, label="Greedy decode",   color="#4A90E2", alpha=0.88)
        b2 = ax.bar(x + w/2, dp_means,     w, label="DP subset decode", color="#D85A30", alpha=0.88)
        ax.axhline(1.0, color="#222", lw=1.3, linestyle="--", label="DP optimal (1.0)")
        all_vals = [v for v in greedy_means + dp_means if v > 0]
        if all_vals:
            ax.set_ylim(max(0.6, min(all_vals) - 0.03), 1.015)
        ax.set_xticks(x); ax.set_xticklabels(variant_labels, fontsize=11)
        ax.set_ylabel("Approximation Ratio (vs DP)", fontsize=11)
        ax.set_title("Pisinger T3 — Graph Structure × Decode Method")
        ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(list(b1) + list(b2), greedy_means + dp_means):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{val:.4f}", ha="center", fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # ---- Chart B: DP subset decode ratio distribution (kNN model) ----
        knn_dp_ratios = pis_data["kNN"]["dp"]
        knn_g_ratios  = pis_data["kNN"]["greedy"]
        if knn_dp_ratios:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4))

            # Left: histogram DP subset
            axes[0].hist(knn_dp_ratios, bins=30, color="#D85A30", alpha=0.8, edgecolor="white")
            if knn_g_ratios:
                axes[0].axvline(_mean(knn_g_ratios), color="#4A90E2", lw=2, linestyle="--",
                                label=f"Greedy decode mean ({_mean(knn_g_ratios):.4f})")
            axes[0].axvline(_mean(knn_dp_ratios), color="#D85A30", lw=2, linestyle="-",
                            label=f"DP subset mean ({_mean(knn_dp_ratios):.4f})")
            axes[0].set_xlabel("Approximation Ratio", fontsize=11)
            axes[0].set_ylabel("Count", fontsize=11)
            axes[0].set_title("DP subset decode — ratio distribution (GNN-kNN)", fontsize=11)
            axes[0].legend(fontsize=9); axes[0].grid(axis="y", alpha=0.3)

            # Right: bucket breakdown
            buckets = [("<0.50", 0, .5), ("0.50-0.70", .5, .7),
                       ("0.70-0.90", .7, .9), ("0.90-0.98", .9, .98), ("≥0.98", .98, 1.1)]
            bucket_lbls = [b[0] for b in buckets]
            bucket_cnts = [sum(1 for r in knn_dp_ratios if b[1] <= r < b[2]) for b in buckets]
            bar_colors  = ["#8B0000","#D85A30","#F5A623","#1D9E75","#378ADD"]
            bars = axes[1].bar(bucket_lbls, bucket_cnts, color=bar_colors, alpha=0.85)
            axes[1].set_xlabel("Ratio bucket", fontsize=11)
            axes[1].set_ylabel("Instance count", fontsize=11)
            axes[1].set_title(f"Phân phối ratio — DP subset decode (N={len(knn_dp_ratios)})", fontsize=11)
            axes[1].grid(axis="y", alpha=0.3)
            for bar, cnt in zip(bars, bucket_cnts):
                axes[1].text(bar.get_x() + bar.get_width() / 2,
                             bar.get_height() + 0.5, str(cnt),
                             ha="center", fontsize=10, fontweight="bold")
            plt.suptitle("DP Subset Decode — Failure Analysis trên Pisinger T3", y=1.01)
            plt.tight_layout(); st.pyplot(fig); plt.close()

        # ---- Table: Ablation summary ----
        abl_rows = []
        for lbl in variant_labels:
            gm = pis_data[lbl]["greedy"]
            dm = pis_data[lbl]["dp"]
            gm_mean = _mean(gm) if gm else None
            dm_mean = _mean(dm) if dm else None
            gm_std  = _stdev(gm) if len(gm) > 1 else 0
            dm_std  = _stdev(dm) if len(dm) > 1 else 0
            delta   = (dm_mean - gm_mean) if gm_mean and dm_mean else None
            lt95    = sum(1 for r in gm if r < 0.95) if gm else None
            lt98    = sum(1 for r in gm if r < 0.98) if gm else None
            abl_rows.append({
                "Graph":              lbl,
                "Greedy decode":      f"{gm_mean:.4f} ± {gm_std:.4f}" if gm_mean else "MISSING",
                "DP subset decode":   f"{dm_mean:.4f} ± {dm_std:.4f}" if dm_mean else "MISSING",
                "Δ (DP − Greedy)":    f"{delta:+.4f}" if delta is not None else "—",
                "< 0.95 (Greedy)":   str(lt95) if lt95 is not None else "—",
                "< 0.98 (Greedy)":   str(lt98) if lt98 is not None else "—",
            })
        st.dataframe(pd.DataFrame(abl_rows), use_container_width=True, hide_index=True)

        # ---- Key findings callout ----
        best_g = max((v for v in greedy_means if v > 0), default=0)
        best_d = max((v for v in dp_means     if v > 0), default=0)
        worst_d = min((v for v in dp_means    if v > 0), default=0)
        st.warning(
            f"**Pisinger T3 — Key findings:**  \n"
            f"• Tất cả GNN variants đạt Greedy decode ratio ~**{best_g:.4f}** — không vượt Greedy heuristic.  \n"
            f"• DP subset decode thất bại (mean ~**{worst_d:.4f}**): GNN probs không có signal rõ "
            f"trên T3 (v/w gần bằng nhau) → top-m filter ngẫu nhiên → DP trên subset sai.  \n"
            f"• Cấu trúc đồ thị **không** tạo ra sự khác biệt — range chỉ {max(greedy_means)-min(v for v in greedy_means if v>0):.4f}."
        )

    # ---- Pisinger Training Curves ----
    st.subheader("Pisinger T3 — Training Curves")
    PIS_LOG_MAP = {
        "knn":             ROOT / "results" / "GNN_pisinger" / "pisinger_kNN"             / "gnn_knn_training_log.csv",
        "conflict_static": ROOT / "results" / "GNN_pisinger" / "pisinger_conflict_static" / "gnn_conflict_static_training_log.csv",
        "random":          ROOT / "results" / "GNN_pisinger" / "pisinger_random"          / "gnn_random_training_log.csv",
        "full":            ROOT / "results" / "GNN_pisinger" / "pisinger_full"            / "gnn_full_training_log.csv",
    }
    pis_logs = {k: _load_training_log(p) for k, p in PIS_LOG_MAP.items()}
    has_pis_logs = any(v is not None for v in pis_logs.values())

    if not has_pis_logs:
        # Try alternative: single kNN log at results/pisinger/
        alt_paths = [
            ROOT / "results" / "pisinger" / "gnn_knn_training_log.csv",
            ROOT / "results" / "GNN_Pisinger_t3" / "gnn_knn_training_log.csv",
        ]
        for ap in alt_paths:
            d = _load_training_log(ap)
            if d:
                pis_logs = {"knn": d}
                has_pis_logs = True
                break

    if not has_pis_logs:
        st.info("Chưa có training log Pisinger. Đặt log CSV vào `results/GNN_Pisinger_t3/gnn_{variant}/training_log.csv`.")
    else:
        # Ratio vs epoch
        fig, ax = plt.subplots(figsize=(11, 5))
        first_pis = next((v for v in pis_logs.values() if v), None)
        for key, d in pis_logs.items():
            if d is None or "gnn_ratio" not in d:
                continue
            m = VARIANT_META.get(key, {"label": key, "color": "#888", "marker": "o"})
            epochs = d["epoch"]; ratios = d["gnn_ratio"]
            ax.plot(epochs, ratios, color=m["color"], lw=1, label=m["label"], marker='o', markersize=4)
            best_ep = epochs[ratios.index(max(ratios))]
            ax.scatter([best_ep], [max(ratios)], color=m["color"], marker="*", s=120, zorder=5)
            ax.annotate(f"{max(ratios):.4f}",
                        xy=(best_ep, max(ratios)),
                        xytext=(best_ep + 0.5, max(ratios) + 0.0005),
                        fontsize=8, color=m["color"])
        if first_pis and "greedy_ratio" in first_pis:
            g = first_pis["greedy_ratio"][0]
            ax.axhline(g, color="#888", linestyle="--", lw=1.4,
                       label=f"Greedy baseline ({g:.4f})")
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel("Approximation Ratio (vs DP)", fontsize=11)
        ax.set_title("Pisinger T3 — Training curve (Ratio per variant)")
        ax.set_ylim(0.973, 0.980)
        ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.75)
        ax.annotate("★ = best epoch", xy=(0.01, 0.02),
                    xycoords="axes fraction", fontsize=8, color="#555")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Loss + Val Acc
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for key, d in pis_logs.items():
            if d is None:
                continue
            m = VARIANT_META.get(key, {"label": key, "color": "#888"})
            if "train_loss" in d:
                axes[0].plot(d["epoch"], d["train_loss"], color=m["color"],
                             lw=2, label=m["label"])
            if "val_acc" in d:
                axes[1].plot(d["epoch"], d["val_acc"], color=m["color"],
                             lw=2, label=m["label"])
        axes[0].set_title("Pisinger T3 — Training Loss")
        axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
        axes[0].legend(fontsize=10); axes[0].grid(alpha=0.25)
        axes[1].set_title("Pisinger T3 — Validation Accuracy")
        axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
        axes[1].legend(fontsize=10); axes[1].grid(alpha=0.25)
        plt.suptitle("Pisinger T3 — Training dynamics", y=1.01)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.divider()

    # --- Pisinger Cross-Type Ablation: greedy_prob vs dp_subset decode ---
    st.subheader("Pisinger Cross-Type Evaluation — Decode Strategy Ablation")
    pis_summary = ROOT / "results" / "GNN_Pisinger_t3" / "cross_type" / "summary_all6.csv"
    if pis_summary.exists():
        df_pis = pd.read_csv(pis_summary)
        # Rút gọn label cho dễ đọc
        type_short = {
            "type_01_uncorrelated":          "T1 Uncorrelated",
            "type_02_weakly_correlated":      "T2 Weakly",
            "type_03_strongly_correlated":    "T3 Strongly",
            "type_04_inverse_strongly_corr":  "T4 Inverse-strong",
            "type_05_almost_strongly_corr":   "T5 Almost-strong",
            "type_06_subset_sum":             "T6 Subset-sum",
        }
        df_pis["type_short"] = df_pis["type"].map(type_short).fillna(df_pis["type"])

        st.caption("Đánh giá GNN-Full trên 6 type Pisinger với 2 decode strategy. "
                   "Cho thấy `dp_subset` decode tách GNN khỏi Greedy, đạt optimal trên mọi type.")

        # --- Chart A: Ratio bar grouped (greedy_prob vs dp_subset) ---
        types_order = list(type_short.values())
        df_g = df_pis[df_pis["strategy"] == "greedy_prob"].set_index("type_short")
        df_d = df_pis[df_pis["strategy"] == "dp_subset"].set_index("type_short")
        types_avail = [t for t in types_order if t in df_g.index or t in df_d.index]

        fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 5))
        x = np.arange(len(types_avail)); w = 0.32

        # Quality
        gnn_g = [df_g.loc[t, "gnn_avg_ratio"] if t in df_g.index else 0 for t in types_avail]
        gnn_d = [df_d.loc[t, "gnn_avg_ratio"] if t in df_d.index else 0 for t in types_avail]
        gr_b  = [df_g.loc[t, "greedy_avg_ratio"] if t in df_g.index else 0 for t in types_avail]
        b1 = axL.bar(x - w, gnn_g, w, color="#D85A30", label="GNN + greedy_prob")
        b2 = axL.bar(x,     gnn_d, w, color="#1D9E75", label="GNN + dp_subset")
        b3 = axL.bar(x + w, gr_b,  w, color="#7F77DD", label="Greedy heuristic")
        axL.axhline(1.0, color="#222", linestyle="--", lw=1.2, label="DP optimal (1.0)")
        axL.set_xticks(x); axL.set_xticklabels(types_avail, rotation=20, ha="right", fontsize=9)
        axL.set_ylabel("Approximation Ratio (vs DP)")
        axL.set_title("Quality across 6 Pisinger types")
        axL.set_ylim(0.7, 1.03); axL.legend(fontsize=9, loc="lower left")
        axL.grid(axis="y", alpha=0.3)
        for bar, val in zip(list(b1) + list(b2) + list(b3), gnn_g + gnn_d + gr_b):
            if val > 0:
                axL.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                         f"{val:.3f}", ha="center", fontsize=7)

        # Speed (log)
        gnn_gt = [df_g.loc[t, "avg_time_ms"] if t in df_g.index else 0 for t in types_avail]
        gnn_dt = [df_d.loc[t, "avg_time_ms"] if t in df_d.index else 0 for t in types_avail]
        axR.bar(x - w/2, gnn_gt, w, color="#D85A30", label="greedy_prob")
        axR.bar(x + w/2, gnn_dt, w, color="#1D9E75", label="dp_subset")
        axR.set_xticks(x); axR.set_xticklabels(types_avail, rotation=20, ha="right", fontsize=9)
        axR.set_ylabel("Avg Inference Time (ms, log)"); axR.set_yscale("log")
        axR.set_title("Speed tradeoff")
        axR.legend(fontsize=9); axR.grid(axis="y", alpha=0.3)

        plt.suptitle("Pisinger Cross-Type — Decode Strategy Ablation", y=1.02)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # --- Chart B: Win/Tie/Lose stacked per type per strategy ---
        fig, ax = plt.subplots(figsize=(13, 5))
        rows = []
        for _, r in df_pis.iterrows():
            label = f"{type_short.get(r['type'], r['type'])} [{r['strategy']}]"
            rows.append((label,
                         int(r["beats_greedy"]),
                         int(r["ties_greedy"]),
                         int(r["loses_greedy"])))
        labels = [r[0] for r in rows]
        wins   = [r[1] for r in rows]
        ties   = [r[2] for r in rows]
        loses  = [r[3] for r in rows]
        y = np.arange(len(rows))
        ax.barh(y, wins,  color="#1D9E75", label="Win > Greedy")
        ax.barh(y, ties,  left=wins,                color="#888888", label="Tie = Greedy")
        ax.barh(y, loses, left=[w+t for w,t in zip(wins,ties)],
                color="#D85A30", label="Lose < Greedy")
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel("Số instance")
        ax.set_title("Head-to-head vs Greedy: 6 type × 2 decode strategies")
        ax.legend(loc="lower right", fontsize=9)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # --- Bảng raw + key insight ---
        with st.expander("Bảng raw — `summary_all6.csv`", expanded=False):
            cols_show = ["type_short", "strategy", "n_instances",
                         "gnn_avg_ratio", "greedy_avg_ratio", "advantage",
                         "beats_greedy", "ties_greedy", "loses_greedy", "avg_time_ms"]
            cols_show = [c for c in cols_show if c in df_pis.columns]
            st.dataframe(df_pis[cols_show], use_container_width=True, hide_index=True)

        # Key insight callout
        st.success(
            "**Key finding:** Với `dp_subset` decode, GNN đạt **ratio = 1.0** trên TẤT CẢ 6 type "
            "(gồm Type 3 strongly-correlated mà ban đầu nghi ngờ). Với `greedy_prob` decode, "
            "GNN bị 'che' bởi greedy heuristic trên các type strongly/almost correlated. "
            "→ Vấn đề không phải ở model, mà ở **decode strategy** — đây là phát hiện chính của ablation."
        )
    else:
        st.info(f"Chưa có `{pis_summary.relative_to(ROOT)}`. "
                "Chạy script eval cross-type trên Pisinger trước.")

    st.divider()

    # --- Chart 3-5: Training curves 4 variants ---
    st.subheader("Training Curves — 4 Graph Variants")
    LOG_DIR = ROOT / "results" / "GNN"
    log_map = {
        "knn":             LOG_DIR / "gnn_knn_training_log.csv",
        "conflict_static": LOG_DIR / "gnn_conflict_static_training_log.csv",
        "random":          LOG_DIR / "gnn_random_training_log.csv",
        "full":            LOG_DIR / "gnn_full_training_log.csv",
    }
    logs = {k: _load_training_log(p) for k, p in log_map.items()}
    has_logs = any(v is not None for v in logs.values())

    if not has_logs:
        st.info("Chưa có training log CSV cho 4 variants trong `results/GNN/`.")
    else:
        # Ratio vs epoch
        fig, ax = plt.subplots(figsize=(11,5))
        first_d = next((v for v in logs.values() if v), None)
        for key, d in logs.items():
            if d is None or "gnn_ratio" not in d: continue
            m = VARIANT_META[key]
            epochs = d["epoch"]; ratios = d["gnn_ratio"]
            ax.plot(epochs, ratios, color=m["color"], lw=2, label=m["label"])
            best_ep = epochs[ratios.index(max(ratios))]
            ax.scatter([best_ep],[max(ratios)],color=m["color"],marker="*",s=120,zorder=5)
        if first_d and "greedy_ratio" in first_d:
            g = first_d["greedy_ratio"][0]
            ax.axhline(g, color="#888", linestyle="--", lw=1.4, label=f"Greedy ({g:.4f})")
        ax.set_xlabel("Epoch",fontsize=11); ax.set_ylabel("Approximation Ratio",fontsize=11)
        ax.set_title("Training curve — Ratio per graph variant"); ax.legend(fontsize=10)
        ax.grid(axis="y",alpha=0.25); ax.annotate("★ = best epoch",
            xy=(0.01,0.02),xycoords="axes fraction",fontsize=8,color="#555")
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Loss + Val Accuracy
        fig, axes = plt.subplots(1,2,figsize=(14,5))
        for key, d in logs.items():
            if d is None: continue
            m = VARIANT_META[key]
            if "train_loss" in d:
                axes[0].plot(d["epoch"],d["train_loss"],color=m["color"],lw=2,label=m["label"])
            if "val_acc" in d:
                axes[1].plot(d["epoch"],d["val_acc"],  color=m["color"],lw=2,label=m["label"])
        axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss (BCE)"); axes[0].legend(fontsize=10); axes[0].grid(alpha=0.25)
        axes[1].set_title("Validation Accuracy"); axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy"); axes[1].legend(fontsize=10); axes[1].grid(alpha=0.25)
        plt.suptitle("Training dynamics — all variants",y=1.01)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        # Head-to-head vs Greedy
        has_h2h = any(d and "gnn_beats_greedy" in d for d in logs.values())
        if has_h2h:
            fig, ax = plt.subplots(figsize=(11,4))
            for key, d in logs.items():
                if d is None or "gnn_beats_greedy" not in d: continue
                m = VARIANT_META[key]
                ax.plot(d["epoch"],d["gnn_beats_greedy"],color=m["color"],lw=2,label=m["label"])
            ax.axhline(100,color="#aaa",linestyle=":",lw=1.2,label="50% threshold")
            ax.set_xlabel("Epoch",fontsize=11); ax.set_ylabel("GNN wins (out of 200)",fontsize=11)
            ax.set_title("Head-to-head: GNN wins vs Greedy per epoch")
            ax.legend(fontsize=10); ax.grid(axis="y",alpha=0.25)
            plt.tight_layout(); st.pyplot(fig); plt.close()

# ---------------------------------------------------------------------------
# TAB 5 — Training Curves (DQN + GNN single model)
# ---------------------------------------------------------------------------
def _plot_rl_log(df: pd.DataFrame, title: str, color: str):
    """Vẽ training curve cho DQN / S2V-DQN từ RLTrainingLogger CSV."""
    for col in ["loss", "avg_value_val", "avg_ratio_val", "epsilon"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    has_ratio = "avg_ratio_val" in df.columns and df["avg_ratio_val"].notna().any()
    n_plots = 3 if has_ratio else 3
    fig, axes = plt.subplots(1, n_plots, figsize=(15, 4))

    # Plot 1: ratio (nếu có) hoặc avg_value
    if has_ratio:
        ratio_df = df.dropna(subset=["avg_ratio_val"])
        if not ratio_df.empty:
            axes[0].plot(ratio_df["step"], ratio_df["avg_ratio_val"], color=color, lw=2)
        axes[0].set_title(f"{title}: Val Ratio vs DP")
        axes[0].set_ylabel("Avg Ratio"); axes[0].set_xlabel("Step")
    else:
        val_df = df.dropna(subset=["avg_value_val"]) if "avg_value_val" in df.columns else pd.DataFrame()
        if not val_df.empty:
            axes[0].plot(val_df["step"], val_df["avg_value_val"], color=color, lw=2)
        axes[0].set_title(f"{title}: Val Avg Value")
        axes[0].set_ylabel("Avg Value"); axes[0].set_xlabel("Step")
    axes[0].grid(alpha=0.3)

    # Plot 2: loss
    loss_df = df.dropna(subset=["loss"]) if "loss" in df.columns else pd.DataFrame()
    if not loss_df.empty:
        lc = loss_df["loss"].clip(upper=loss_df["loss"].quantile(0.98))
        axes[1].plot(loss_df["step"], lc, color="#D85A30", lw=1.5)
    axes[1].set_title(f"{title}: Training Loss")
    axes[1].set_ylabel("Loss"); axes[1].set_xlabel("Step"); axes[1].grid(alpha=0.3)

    # Plot 3: epsilon
    if "epsilon" in df.columns:
        axes[2].plot(df["step"], df["epsilon"], color="#1D9E75", lw=2)
    axes[2].set_title("Epsilon-Greedy")
    axes[2].set_ylabel("Epsilon"); axes[2].set_xlabel("Step"); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    return fig


def tab_training():
    st.header("Training Curves — GNN, DQN, S2V-DQN, REINFORCE")

    col1, col2 = st.columns(2)
    with col1:
        gnn_log_dir = st.text_input("GNN log dir",
            value=str(ROOT / "results" / "GNN"))
    with col2:
        dqn_log = st.text_input("DQN training log",
            value=str(ROOT / "results" / "DQN" / "training_dqn_log.csv"))

    col3, col4 = st.columns(2)
    with col3:
        s2v_log = st.text_input("S2V-DQN training log",
            value=str(ROOT / "results" / "S2V_DQN" / "s2v_dqn_knn" / "training_knn_log.csv"))
    with col4:
        reinforce_log = st.text_input("REINFORCE training log",
            value=str(ROOT / "results" / "GNN_REINFORCE" / "gnnrl_knn_training_log.csv"))

    show_gnn       = st.checkbox("GNN (all variants)",    value=True)
    show_pis       = st.checkbox("Pisinger T3 training",  value=False)
    show_dqn       = st.checkbox("DQN",                   value=True)
    show_s2v       = st.checkbox("S2V-DQN",               value=True)
    show_reinforce = st.checkbox("REINFORCE",              value=True)

    if not st.button("Vẽ training curves", type="primary"): return

    if show_gnn:
        import subprocess
        plot_dir = ROOT / "plots"
        subprocess.run(
            [sys.executable, str(ROOT / "src" / "scripts" / "plot_results.py"),
             "--log_dir", gnn_log_dir, "--out_dir", str(plot_dir)],
            capture_output=True, text=True, cwd=str(ROOT)
        )
        for fname, title in [
            ("training_ratio_all_variants.png",    "Ratio theo epoch"),
            ("training_loss_acc_all_variants.png", "Loss & Val Accuracy"),
            ("training_h2h_all_variants.png",       "H2H vs Greedy"),
        ]:
            p = plot_dir / fname
            if p.exists():
                st.subheader(title); st.image(str(p))

    if show_pis:
        pis_log_map = {
            "knn":            ROOT / "results" / "pisinger" / "gnn_knn"      / "training_log.csv",
            "conflict_static":ROOT / "results" / "pisinger" / "gnn_conflict" / "training_log.csv",
            "random":         ROOT / "results" / "pisinger" / "gnn_random"   / "training_log.csv",
            "full":           ROOT / "results" / "pisinger" / "gnn_full"     / "training_log.csv",
        }
        # Fallback: single kNN log
        alt_pis = [
            ROOT / "results" / "pisinger" / "gnn_knn_training_log.csv",
            ROOT / "results" / "GNN_Pisinger_t3" / "gnn_knn_training_log.csv",
        ]
        pis_logs_t = {k: _load_training_log(p) for k, p in pis_log_map.items()}
        if not any(v for v in pis_logs_t.values()):
            for ap in alt_pis:
                d = _load_training_log(ap)
                if d:
                    pis_logs_t = {"knn": d}
                    break
        if not any(v for v in pis_logs_t.values()):
            st.warning("Không tìm thấy training log Pisinger T3.")
        else:
            st.subheader("Pisinger T3 — Training Ratio")
            fig, ax = plt.subplots(figsize=(11, 5))
            first_pd = next((v for v in pis_logs_t.values() if v), None)
            for key, d in pis_logs_t.items():
                if d is None or "gnn_ratio" not in d:
                    continue
                m = VARIANT_META.get(key, {"label": key, "color": "#888"})
                ratios = d["gnn_ratio"]; epochs = d["epoch"]
                ax.plot(epochs, ratios, color=m["color"], lw=2, label=m["label"])
                best_ep = epochs[ratios.index(max(ratios))]
                ax.scatter([best_ep], [max(ratios)], color=m["color"], marker="*", s=130, zorder=5)
            if first_pd and "greedy_ratio" in first_pd:
                g = first_pd["greedy_ratio"][0]
                ax.axhline(g, color="#888", linestyle="--", lw=1.4,
                           label=f"Greedy baseline ({g:.4f})")
            ax.set_xlabel("Epoch"); ax.set_ylabel("Approximation Ratio (vs DP)")
            ax.set_title("Pisinger T3 — Ratio per graph variant")
            ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.25)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            # Loss + Acc
            fig, axes2 = plt.subplots(1, 2, figsize=(13, 5))
            for key, d in pis_logs_t.items():
                if d is None:
                    continue
                m = VARIANT_META.get(key, {"label": key, "color": "#888"})
                if "train_loss" in d:
                    axes2[0].plot(d["epoch"], d["train_loss"], color=m["color"], lw=2, label=m["label"])
                if "val_acc" in d:
                    axes2[1].plot(d["epoch"], d["val_acc"],   color=m["color"], lw=2, label=m["label"])
            axes2[0].set_title("Pisinger T3 — Loss"); axes2[0].set_ylabel("BCE Loss")
            axes2[1].set_title("Pisinger T3 — Val Accuracy"); axes2[1].set_ylabel("Accuracy")
            for ax2 in axes2:
                ax2.set_xlabel("Epoch"); ax2.legend(fontsize=10); ax2.grid(alpha=0.25)
            plt.suptitle("Pisinger T3 — Training dynamics", y=1.01)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    if show_dqn:
        p = Path(dqn_log)
        if p.exists():
            fig = _plot_rl_log(pd.read_csv(p), "DQN", "#378ADD")
            st.subheader("DQN Training"); st.pyplot(fig); plt.close()
        else:
            st.warning(f"Không tìm thấy: {dqn_log}")

    if show_s2v:
        # Auto-discover all S2V variants in results/S2V_DQN/s2v_dqn_<gt>/training_<gt>_log.csv
        s2v_logs = sorted((ROOT / "results" / "S2V_DQN").glob("s2v_dqn_*/training_*_log.csv"))
        # Add user-specified path if not already in the list
        user_p = Path(s2v_log)
        if user_p.exists() and user_p not in s2v_logs:
            s2v_logs.insert(0, user_p)
        if not s2v_logs:
            st.warning(f"Không tìm thấy training log nào cho S2V-DQN (đã tìm: {s2v_log})")
        else:
            S2V_COLORS = {"knn":"#9B59B6","conflict":"#7D3C98","random":"#BB8FCE","full":"#D2B4DE"}
            for p in s2v_logs:
                # Extract variant from path like .../s2v_dqn_knn/training_knn_log.csv
                variant = p.parent.name.replace("s2v_dqn_", "") or p.stem.replace("training_","").replace("_log","")
                color = S2V_COLORS.get(variant, "#E8A33D")
                fig = _plot_rl_log(pd.read_csv(p), f"S2V-DQN [{variant}]", color)
                st.subheader(f"S2V-DQN [{variant}] Training"); st.pyplot(fig); plt.close()

    if show_reinforce:
        # Auto-discover REINFORCE baseline variants in results/GNN_REINFORCE/gnnrl_*_log.csv
        rf_logs = sorted((ROOT / "results" / "GNN_REINFORCE").glob("gnnrl_*_log.csv"))
        # Also check nested subdirs (gnn_reinforce_<gt>_<baseline>/gnnrl_<gt>_<baseline>_log.csv)
        rf_logs += sorted((ROOT / "results" / "GNN_REINFORCE").glob("gnn_reinforce_*/gnnrl_*_log.csv"))
        user_p = Path(reinforce_log)
        if user_p.exists() and user_p not in rf_logs:
            rf_logs.insert(0, user_p)
        # De-duplicate while preserving order
        seen = set()
        rf_logs = [p for p in rf_logs if not (p in seen or seen.add(p))]
        if not rf_logs:
            st.warning(f"Không tìm thấy training log nào cho REINFORCE (đã tìm: {reinforce_log})")
        else:
            RF_COLORS = {"hard":"#E74C3C","none":"#C0392B","polyak":"#922B21"}
            for p in rf_logs:
                # Extract baseline tag from filename: gnnrl_<gt>_<baseline>_log.csv → baseline
                parts = p.stem.split("_")
                # Pattern: gnnrl_<gt>_<baseline>_log  or  gnnrl_<gt>_training_log
                tag = parts[-2] if len(parts) >= 4 and parts[-1] == "log" else "default"
                color = RF_COLORS.get(tag, "#E74C3C")
                df_rf = pd.read_csv(p)
                for col in ["train_loss", "val_ratio", "val_feasibility"]:
                    if col in df_rf.columns:
                        df_rf[col] = pd.to_numeric(df_rf[col], errors="coerce")
                has_ratio = "val_ratio" in df_rf.columns and df_rf["val_ratio"].notna().any()
                fig, axes = plt.subplots(1, 3, figsize=(15, 4))
                if has_ratio:
                    r_df = df_rf.dropna(subset=["val_ratio"])
                    axes[0].plot(r_df["epoch"], r_df["val_ratio"], color=color, lw=2)
                axes[0].set_title(f"REINFORCE [{tag}]: Val Ratio vs DP")
                axes[0].set_ylabel("Avg Ratio"); axes[0].set_xlabel("Epoch"); axes[0].grid(alpha=0.3)
                if "train_loss" in df_rf.columns:
                    l_df = df_rf.dropna(subset=["train_loss"])
                    axes[1].plot(l_df["epoch"], l_df["train_loss"], color="#D85A30", lw=1.5)
                axes[1].set_title(f"REINFORCE [{tag}]: Train Loss")
                axes[1].set_ylabel("Loss"); axes[1].set_xlabel("Epoch"); axes[1].grid(alpha=0.3)
                if "val_feasibility" in df_rf.columns:
                    f_df = df_rf.dropna(subset=["val_feasibility"])
                    axes[2].plot(f_df["epoch"], f_df["val_feasibility"], color="#1D9E75", lw=2)
                axes[2].set_title(f"REINFORCE [{tag}]: Val Feasibility")
                axes[2].set_ylabel("Feasibility"); axes[2].set_xlabel("Epoch"); axes[2].grid(alpha=0.3)
                plt.tight_layout()
                st.subheader(f"REINFORCE [{tag}] Training ({p.name})"); st.pyplot(fig); plt.close()

# ---------------------------------------------------------------------------
# TAB 6 — Phân tích kết quả  (P3)
# ---------------------------------------------------------------------------
def tab_analysis():
    st.header("Phân tích kết quả — Bảng so sánh chi tiết")
    BASE = ROOT / "results" / "cross_scale"

    if not BASE.exists():
        st.warning("Chưa có `results/cross_scale/`. Chạy cross_scale_eval.py trước."); return

    graph_keys   = ["knn","conflict_static","random","full"]
    graph_labels = ["kNN","Conflict","Random","Full"]
    test_keys    = ["small","n100","n200"]
    test_labels  = ["n≤50","n=100","n=200"]

    # Table 1: Ablation
    st.subheader("Table 1 — Ablation: Graph × Decode (test n≤50)")
    dp_small = BASE/"dp_small.csv"
    if dp_small.exists():
        rows = []
        for g, lbl in zip(graph_keys, graph_labels):
            gd, n1 = _compute_ratio(BASE/f"{g}_on_small_greedy.csv", dp_small)
            dd, n2 = _compute_ratio(BASE/f"{g}_on_small_dp.csv",     dp_small)
            rows.append({"Graph": lbl,
                         "Greedy decode":    f"{gd:.4f}" if gd else "MISSING",
                         "DP Subset decode": f"{dd:.4f}" if dd else "MISSING",
                         "Delta": f"{(dd-gd):+.4f}" if gd and dd else "—"})
        gb, _ = _compute_ratio(BASE/"greedy_small.csv", dp_small)
        rows.append({"Graph":"(Greedy heuristic)",
                     "Greedy decode": f"{gb:.4f}" if gb else "MISSING",
                     "DP Subset decode":"—","Delta":"—"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Thiếu dp_small.csv")

    st.divider()

    # Table 2: Cross-scale
    st.subheader("Table 2 — Cross-scale (Greedy decode, trained n≤50)")
    dp_files = {t: BASE/f"dp_{t}.csv" for t in test_keys}
    if all(p.exists() for p in dp_files.values()):
        RL_META_ANALYSIS = {
            "dqn": "DQN", "s2v": "S2V-DQN", "reinforce": "REINFORCE"
        }
        rows = []
        for g, lbl in zip(graph_keys, graph_labels):
            row = {"Model": f"GNN-{lbl}"}
            for t, tl in zip(test_keys, test_labels):
                r, n = _compute_ratio(BASE/f"{g}_on_{t}_greedy.csv", dp_files[t])
                row[tl] = f"{r:.4f}" if r else "MISSING"
            rows.append(row)
        # for key, lbl in RL_META_ANALYSIS.items():
        #     if any((BASE/f"{key}_on_{t}.csv").exists() for t in test_keys):
        #         row = {"Model": lbl}
        #         for t, tl in zip(test_keys, test_labels):
        #             r, n = _compute_ratio(BASE/f"{key}_on_{t}.csv", dp_files[t])
        #             row[tl] = f"{r:.4f}" if r else "MISSING"
        #         rows.append(row)
        # row = {"Model": "(Greedy heuristic)"}

        # ---- DQN (một dòng) ----
        dqn_has = False
        dqn_row = {"Model": "DQN"}
        for t, tl in zip(test_keys, test_labels):
            cand = BASE / f"dqn_on_{t}.csv"
            if not cand.exists():
                cand = BASE / f"dqn_{t}.csv"
            if cand.exists():
                r, n = _compute_ratio(cand, dp_files[t])
                dqn_row[tl] = f"{r:.4f}" if r else "MISSING"
                dqn_has = True
            else:
                dqn_row[tl] = "MISSING"
        if dqn_has:
            rows.append(dqn_row)

        # ---- S2V-DQN với 4 graph type ----
        s2v_graphs = [
            ("knn", "S2V-kNN"),
            ("conflict_static", "S2V-Conflict"),
            ("random", "S2V-Random"),
            ("full", "S2V-Full"),
        ]
        for gt, display in s2v_graphs:
            has = False
            row = {"Model": display}
            for t, tl in zip(test_keys, test_labels):
                cand = BASE / f"s2v_{gt}_on_{t}.csv"
                if not cand.exists():
                    cand = BASE / f"s2v_{gt}_{t}.csv"
                if cand.exists():
                    r, n = _compute_ratio(cand, dp_files[t])
                    row[tl] = f"{r:.4f}" if r else "MISSING"
                    has = True
                else:
                    row[tl] = "MISSING"
            if has:
                rows.append(row)

        # ---- REINFORCE (chỉ knn) ----
        baseline_keys = ['none', 'hard', 'polyak']
        for baseline in baseline_keys:
            has_data = False
            row = {"Model": f"REINFORCE ({baseline})"}  # hiển thị rõ baseline
            for t, tl in zip(test_keys, test_labels):
                # Thử các pattern tên file
                cand = BASE / f"reinforce_{baseline}_knn_on_{t}.csv"
                if not cand.exists():
                    cand = BASE / f"reinforce_{baseline}_on_{t}.csv"
                if not cand.exists():
                    cand = BASE / f"reinforce_{baseline}_{t}.csv"
                if cand.exists():
                    r, n = _compute_ratio(cand, dp_files[t])
                    row[tl] = f"{r:.4f}" if r else "MISSING"
                    has_data = True
                else:
                    row[tl] = "MISSING"
            if has_data:
                rows.append(row)

        row = {"Model": "(Greedy heuristic)"}
        for t, tl in zip(test_keys, test_labels):
            r, n = _compute_ratio(BASE/f"greedy_{t}.csv", dp_files[t])
            row[tl] = f"{r:.4f}" if r else "MISSING"
        rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("Thiếu dp_small/n100/n200.csv")

    st.divider()

    # Table 3: Decode method comparison (kNN best model)
    st.subheader("Table 3 — Decode Method Comparison (GNN-kNN)")
    if all(p.exists() for p in dp_files.values()):
        rows = []
        for t, tl in zip(test_keys, test_labels):
            gd, _ = _compute_ratio(BASE/f"knn_on_{t}_greedy.csv", dp_files[t])
            dd, _ = _compute_ratio(BASE/f"knn_on_{t}_dp.csv",     dp_files[t])
            rows.append({
                "Test set": tl,
                "Greedy decode":    f"{gd:.4f}" if gd else "MISSING",
                "DP Subset decode": f"{dd:.4f}" if dd else "MISSING",
                "Delta":            f"{(dd-gd):+.4f}" if gd and dd else "—",
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # Head-to-head
    st.subheader("Head-to-head: GNN variants vs Greedy (test n≤50)")
    if dp_small.exists():
        greedy_vals = _load_csv_values(BASE/"greedy_small.csv")
        h2h_rows = []
        for g, lbl in zip(graph_keys, graph_labels):
            gnn_vals = _load_csv_values(BASE/f"{g}_on_small_greedy.csv")
            common = set(gnn_vals) & set(greedy_vals)
            if not common: continue
            beats = sum(1 for k in common if gnn_vals[k] > greedy_vals[k]+1e-6)
            ties  = sum(1 for k in common if abs(gnn_vals[k]-greedy_vals[k]) < 1e-6)
            loses = len(common)-beats-ties
            h2h_rows.append({"Graph":lbl,"Wins":beats,"Ties":ties,"Losses":loses,
                              "Win rate": f"{beats/len(common)*100:.1f}%"})
        if h2h_rows:
            st.dataframe(pd.DataFrame(h2h_rows), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------
# TAB 7 — Gộp kết quả
# ---------------------------------------------------------------------------
def tab_merge():
    st.header("Gộp kết quả — Merge Results")
    st.caption("Chọn file CSV của từng solver, nhấn **Merge & Lưu** để tạo bảng so sánh tổng hợp.")

    # Import merge function
    try:
        import importlib.util, sys as _sys
        _spec = importlib.util.spec_from_file_location(
            "Merge_results",
            str(ROOT / "src" / "scripts" / "Merge_results.py")
        )
        _mod = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        load_csv_fn = _mod.load_csv
        merge_fn    = _mod.merge
    except Exception as e:
        st.error(f"Không load được Merge_results.py: {e}")
        return

    # Default paths
    DEFAULT_PATHS = {
        "dp":       ROOT / "results" / "DP"           / "dp_results.csv",
        "gnn":      ROOT / "results" / "GNN"          / "gnn_eval_results.csv",
        "gnn_dp":   ROOT / "results" / "GNN"          / "gnn_dp_eval_results.csv",
        "greedy":   ROOT / "results" / "Greedy"       / "greedy_eval_results.csv",
        "ga":       ROOT / "results" / "GA"           / "ga_eval_results.csv",
        "bb":       ROOT / "results" / "BB"           / "bb_results.csv",
        "dqn":      ROOT / "results" / "DQN"          / "dqn_on_small.csv",
        "s2v":      ROOT / "results" / "S2V_DQN"      / "s2v_dqn_eval_results.csv",
        "reinforce":ROOT / "results" / "GNN_REINFORCE"/ "reinforce_eval_results.csv",
    }
    LABELS = {
        "dp":"DP (Ground Truth)","gnn":"GNN (greedy decode)",
        "gnn_dp":"GNN-DP (DP-subset decode)","greedy":"Greedy",
        "ga":"GA","bb":"Branch & Bound","dqn":"DQN",
        "s2v":"S2V-DQN","reinforce":"REINFORCE",
    }

    # --- Path inputs ---
    st.subheader("1. Chọn file CSV")
    st.caption("Bỏ trống nếu solver chưa có kết quả.")

    user_paths = {}
    cols_left, cols_right = st.columns(2)
    solver_list = list(DEFAULT_PATHS.keys())
    for i, solver in enumerate(solver_list):
        col = cols_left if i % 2 == 0 else cols_right
        default = str(DEFAULT_PATHS[solver])
        exists  = DEFAULT_PATHS[solver].exists()
        icon    = "✅" if exists else "❌"
        with col:
            val = st.text_input(
                f"{icon} {LABELS[solver]}",
                value=default if exists else "",
                key=f"merge_path_{solver}",
                placeholder="(bỏ trống = bỏ qua solver này)",
            )
            user_paths[solver] = Path(val.strip()) if val.strip() else None

    st.divider()

    # --- Output dir ---
    st.subheader("2. Thư mục lưu kết quả")
    out_dir_str = st.text_input(
        "Output directory",
        value=str(ROOT / "results" / "compare"),
        key="merge_out_dir",
    )
    out_dir = Path(out_dir_str.strip())

    st.divider()

    # --- Preview ---
    st.subheader("3. Kiểm tra trước khi merge")
    preview_rows = []
    for solver, path in user_paths.items():
        if path and path.exists():
            try:
                with path.open(newline="", encoding="utf-8") as f:
                    n = sum(1 for _ in f) - 1
                preview_rows.append({"Solver": LABELS[solver], "File": path.name,
                                     "Rows": n, "Status": "✅ Sẵn sàng"})
            except Exception as e:
                preview_rows.append({"Solver": LABELS[solver], "File": path.name,
                                     "Rows": "?", "Status": f"⚠ {e}"})
        elif path:
            preview_rows.append({"Solver": LABELS[solver], "File": path.name,
                                 "Rows": "—", "Status": "❌ Không tìm thấy"})
        else:
            preview_rows.append({"Solver": LABELS[solver], "File": "—",
                                 "Rows": "—", "Status": "⏭ Bỏ qua"})
    st.dataframe(pd.DataFrame(preview_rows), use_container_width=True, hide_index=True)

    # Check DP
    dp_ok = user_paths.get("dp") and user_paths["dp"].exists()
    if not dp_ok:
        st.warning("DP là ground truth bắt buộc. Cần có `dp_results.csv` để tính ratio.")

    st.divider()

    # --- Merge button ---
    st.subheader("4. Thực hiện Merge")
    if st.button("Merge & Lưu", type="primary", disabled=not dp_ok):
        with st.spinner("Đang merge..."):
            try:
                def _load(solver):
                    p = user_paths.get(solver)
                    if p and p.exists():
                        return load_csv_fn(p, solver.upper())
                    return None

                # Wrap single-CSV variant solvers as one-key dicts (variant name = "main").
                def _wrap(rows):
                    return {"main": rows} if rows else None

                merge_fn(
                    dp_rows            = _load("dp"),
                    greedy_rows        = _load("greedy"),
                    ga_rows            = _load("ga"),
                    bb_rows            = _load("bb"),
                    dqn_rows           = _load("dqn"),
                    gnn_variants       = _wrap(_load("gnn")),
                    gnn_dp_variants    = _wrap(_load("gnn_dp")),
                    s2v_variants       = _wrap(_load("s2v")),
                    reinforce_variants = _wrap(_load("reinforce")),
                    out_dir            = out_dir,
                )
                st.success(f"Merge thành công! Kết quả lưu tại: `{out_dir}`")

                # Show summary
                summary_path = out_dir / "summary.json"
                if summary_path.exists():
                    with open(summary_path) as f:
                        summary = json.load(f)
                    srows = []
                    for solver_key, d in _iter_summary_solvers(summary):
                        srows.append({
                            "Solver":       _solver_display_label(solver_key),
                            "N":            d.get("count", 0),
                            "Feasible%":    f"{d.get('feasible_rate',0)*100:.1f}%",
                            "Avg Ratio":    f"{d['avg_ratio_vs_dp_feasible']:.4f}" if d.get("avg_ratio_vs_dp_feasible") else "N/A",
                            "Std":          f"{d['std_ratio_vs_dp']:.4f}"          if d.get("std_ratio_vs_dp")          else "N/A",
                            "Avg Time(ms)": f"{d['avg_time_ms']:.2f}"              if d.get("avg_time_ms")              else "N/A",
                        })
                    st.subheader("Kết quả Merge")
                    st.dataframe(pd.DataFrame(srows), use_container_width=True, hide_index=True)

                # Download buttons
                merged_csv = out_dir / "merged_results.csv"
                if merged_csv.exists():
                    with open(merged_csv, "rb") as f:
                        st.download_button(
                            "Download merged_results.csv",
                            data=f.read(),
                            file_name="merged_results.csv",
                            mime="text/csv",
                        )
                if summary_path.exists():
                    with open(summary_path, "rb") as f:
                        st.download_button(
                            "Download summary.json",
                            data=f.read(),
                            file_name="summary.json",
                            mime="application/json",
                        )

            except Exception as e:
                st.error(f"Lỗi khi merge: {e}")
                import traceback
                st.code(traceback.format_exc())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Knapsack Solver Benchmark",
    page_icon="🎒",
    layout="wide",
)

# Tắt hiệu ứng tối màn hình khi Streamlit rerun
st.markdown("""
<style>
.stale                          { opacity: 1 !important; transition: none !important; }
[data-stale="true"]             { opacity: 1 !important; }
[data-testid="stStatusWidget"]  { display: none !important; }
</style>
""", unsafe_allow_html=True)

st.title("🎒 Knapsack Solver Benchmark")
st.caption("So sánh GNN, RL và các thuật toán cổ điển cho bài toán 0/1 Knapsack")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Dashboard",
    "⚡ Live Demo",
    "🔍 So sánh",
    "📐 Ablation & Cross-scale",
    "📈 Training Curves",
    "📋 Phân tích",
    "🔀 Gộp kết quả",
])

with tab1: tab_dashboard()
with tab2: tab_live_demo()
with tab3: tab_scatter()
with tab4: tab_ablation()
with tab5: tab_training()
with tab6: tab_analysis()
with tab7: tab_merge()
