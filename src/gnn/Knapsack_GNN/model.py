"""KnapsackGNN model"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, GINConv, global_add_pool
from torch_geometric.utils import softmax as scatter_softmax

DEFAULT_IN_DIM     = 7
DEFAULT_HIDDEN_DIM = 256
DEFAULT_NUM_LAYERS = 4
DEFAULT_DROPOUT    = 0.15
DEFAULT_CONV_TYPE  = "gin"

# Arch tags recognised as "current" — no strict=False warning
VALID_ARCHS = {"gin_layernorm", "gin_v4_layernorm", "gin_v5_dropout"}


# ---------------------------------------------------------------------------
# Conv layer factories
# ---------------------------------------------------------------------------

def _make_gin_conv(in_dim: int, out_dim: int, dropout: float = DEFAULT_DROPOUT) -> GINConv:
    """GINConv with LayerNorm + Dropout inside the MLP."""
    mlp = nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(out_dim, out_dim),
    )
    return GINConv(nn=mlp, train_eps=True)


def _make_sage_conv(in_dim: int, out_dim: int) -> SAGEConv:
    return SAGEConv(in_dim, out_dim)


def _make_conv(
    conv_type: str,
    in_dim:    int,
    out_dim:   int,
    dropout:   float = DEFAULT_DROPOUT,
):
    if conv_type == "gin":
        return _make_gin_conv(in_dim, out_dim, dropout)
    elif conv_type == "sage":
        return _make_sage_conv(in_dim, out_dim)
    else:
        raise ValueError(f"Unknown conv_type: {conv_type}")


# ---------------------------------------------------------------------------
# Global context injection — VECTORIZED
# ---------------------------------------------------------------------------

class GlobalContextInjection(nn.Module):
    """Attention-weighted global pooling broadcast to each node."""

    def __init__(self, hidden_dim: int, dropout_attn: float = 0.1):
        super().__init__()
        self.attn_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(p=dropout_attn),   # ← v5: added (lighter p)
            nn.Linear(hidden_dim // 2, 1),
        )
        self.context_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        x:     torch.Tensor,   # [N, H]
        batch: torch.Tensor,   # [N]
    ) -> torch.Tensor:         # [N, H]
        # Attention scores → softmax within each graph
        attn_scores  = self.attn_gate(x)                           # [N, 1]
        attn_weights = scatter_softmax(attn_scores, batch, dim=0)  # [N, 1]

        # Weighted sum per graph
        weighted  = x * attn_weights           # [N, H]
        graph_emb = global_add_pool(weighted, batch)  # [G, H]

        # Project and broadcast back to nodes
        context      = self.context_proj(graph_emb)  # [G, H]
        node_context = context[batch]                # [N, H]
        return node_context


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class KnapsackGNN(nn.Module):
    def __init__(
        self,
        in_dim:         int   = DEFAULT_IN_DIM,
        hidden_dim:     int   = DEFAULT_HIDDEN_DIM,
        num_layers:     int   = DEFAULT_NUM_LAYERS,
        dropout:        float = DEFAULT_DROPOUT,
        use_residual:   bool  = True,
        use_global_ctx: bool  = True,
        conv_type:      str   = DEFAULT_CONV_TYPE,
    ):
        super().__init__()
        self.in_dim         = in_dim
        self.hidden_dim     = hidden_dim
        self.num_layers     = num_layers
        self.dropout_p      = dropout
        self.use_residual   = use_residual
        self.use_global_ctx = use_global_ctx
        self.conv_type      = conv_type

        # ── Message passing layers ────────────────────────────────────────
        self.convs = nn.ModuleList()
        if conv_type == "hybrid":
            # First n-1 layers: GIN; last layer: SAGE
            self.convs.append(_make_gin_conv(in_dim, hidden_dim, dropout))
            for _ in range(1, num_layers - 1):
                self.convs.append(_make_gin_conv(hidden_dim, hidden_dim, dropout))
            if num_layers > 1:
                self.convs.append(_make_sage_conv(hidden_dim, hidden_dim))
        else:
            self.convs.append(_make_conv(conv_type, in_dim, hidden_dim, dropout))
            for _ in range(num_layers - 1):
                self.convs.append(_make_conv(conv_type, hidden_dim, hidden_dim, dropout))

        # LayerNorm between layers — stable across mixed graph sizes
        self.norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(p=dropout)

        # Residual projection for the first layer (in_dim → hidden_dim)
        self.res_proj = (
            nn.Linear(in_dim, hidden_dim, bias=False) if use_residual else None
        )

        # ── Global context injection ──────────────────────────────────────
        # dropout_attn is lighter (0.1) — see GlobalContextInjection docstring
        dropout_attn = min(dropout, 0.1)
        if use_global_ctx:
            self.global_ctx = GlobalContextInjection(hidden_dim, dropout_attn)
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            )
        else:
            self.global_ctx = None
            self.fusion     = None

        # ── Output head ───────────────────────────────────────────────────
        self.lin = nn.Linear(hidden_dim, 1)

    # ------------------------------------------------------------------
    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index

        if x.size(1) != self.in_dim:
            raise ValueError(
                f"KnapsackGNN expects {self.in_dim} input features, "
                f"got {x.size(1)}. Check Graph_builder in_dim."
            )

        # Infer batch vector (single graph → all zeros)
        if hasattr(data, "batch") and data.batch is not None:
            batch_vec = data.batch
        else:
            batch_vec = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        # ── Message passing with residual connections ─────────────────────
        for i, conv in enumerate(self.convs):
            x_in = x
            x    = conv(x, edge_index)
            x    = self.norms[i](x)
            x    = torch.relu(x)
            x    = self.dropout(x)

            if self.use_residual:
                if i == 0 and self.res_proj is not None:
                    x = x + self.res_proj(x_in)
                elif i > 0:
                    x = x + x_in

        # ── Global context injection ──────────────────────────────────────
        if self.use_global_ctx and self.global_ctx is not None:
            ctx = self.global_ctx(x, batch_vec)              # [N, H]
            x   = self.fusion(torch.cat([x, ctx], dim=1))    # [N, 2H] → [N, H]

        logits = self.lin(x).squeeze(-1)   # [N]
        return logits


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

CKPT_KEY_STATE      = "model_state"
CKPT_KEY_IN_DIM     = "in_dim"
CKPT_KEY_HIDDEN     = "hidden_dim"
CKPT_KEY_LAYERS     = "num_layers"
CKPT_KEY_DROPOUT    = "dropout"
CKPT_KEY_RESIDUAL   = "use_residual"
CKPT_KEY_GLOBAL_CTX = "use_global_ctx"
CKPT_KEY_CONV_TYPE  = "conv_type"
CKPT_KEY_ARCH       = "arch"

CURRENT_ARCH = "gin_v5_dropout"


def save_checkpoint(model: KnapsackGNN, path: Path | str) -> None:
    """Save model weights and all hyperparameters needed to reconstruct it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            CKPT_KEY_STATE:      model.state_dict(),
            CKPT_KEY_IN_DIM:     model.in_dim,
            CKPT_KEY_HIDDEN:     model.hidden_dim,
            CKPT_KEY_LAYERS:     model.num_layers,
            CKPT_KEY_DROPOUT:    model.dropout_p,
            CKPT_KEY_RESIDUAL:   model.use_residual,
            CKPT_KEY_GLOBAL_CTX: model.use_global_ctx,
            CKPT_KEY_CONV_TYPE:  model.conv_type,
            CKPT_KEY_ARCH:       CURRENT_ARCH,
        },
        path,
    )
    print(f"[save_checkpoint] Saved {CURRENT_ARCH} → {path}")


def load_checkpoint(
    path:    Path | str,
    device:  torch.device,
    dropout: Optional[float] = None,
) -> KnapsackGNN:
    """Load a checkpoint and return an eval-mode KnapsackGNN.

    Args:
        path:    Path to the .pt file.
        device:  Target device.
        dropout: Override the saved dropout value (useful for test-time
                 ablations without retraining).
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)

    if isinstance(ckpt, dict) and CKPT_KEY_STATE in ckpt:
        in_dim       = ckpt.get(CKPT_KEY_IN_DIM,     DEFAULT_IN_DIM)
        hidden_dim   = ckpt.get(CKPT_KEY_HIDDEN,     DEFAULT_HIDDEN_DIM)
        num_layers   = ckpt.get(CKPT_KEY_LAYERS,     DEFAULT_NUM_LAYERS)
        dp           = ckpt.get(CKPT_KEY_DROPOUT,    DEFAULT_DROPOUT)
        use_residual = ckpt.get(CKPT_KEY_RESIDUAL,   True)
        use_global   = ckpt.get(CKPT_KEY_GLOBAL_CTX, True)
        conv_type    = ckpt.get(CKPT_KEY_CONV_TYPE,  DEFAULT_CONV_TYPE)
        arch         = ckpt.get(CKPT_KEY_ARCH,       "unknown")
        state_dict   = ckpt[CKPT_KEY_STATE]

        # ── v5 fix: correct arch validation set ──────────────────────────
        if arch not in VALID_ARCHS:
            print(
                f"[load_checkpoint] Old arch '{arch}' — loading with strict=False. "
                f"BatchNorm stats will be dropped; LayerNorm starts fresh."
            )
    else:
        # Legacy: raw state_dict with no metadata
        state_dict   = ckpt
        in_dim       = DEFAULT_IN_DIM
        hidden_dim   = DEFAULT_HIDDEN_DIM
        num_layers   = DEFAULT_NUM_LAYERS
        dp           = DEFAULT_DROPOUT
        use_residual = True
        use_global   = True
        conv_type    = DEFAULT_CONV_TYPE
        arch         = "unknown"
        print("[load_checkpoint] Legacy raw state_dict — using defaults.")

    # Allow caller to override dropout (e.g. set to 0.0 at test time)
    if dropout is not None:
        dp = dropout

    model = KnapsackGNN(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dp,
        use_residual=use_residual,
        use_global_ctx=use_global,
        conv_type=conv_type,
    ).to(device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[load_checkpoint] Missing keys  : {len(missing)}"
              f"  (expected when upgrading v4 → v5 due to new Dropout layers)")
    if unexpected:
        print(f"[load_checkpoint] Unexpected keys: {len(unexpected)}")

    model.eval()
    print(f"[load_checkpoint] Loaded arch='{arch}' | "
          f"hidden={hidden_dim} | layers={num_layers} | dropout={dp}")
    return model