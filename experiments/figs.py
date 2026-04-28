"""
Figures for the APA paper.

Each figure is produced by a top-level ``fig_*`` function and saved to
``experiments/figs/``. Run via the CLI:

    uv run python -m experiments.figs user_weights_grid
    uv run python -m experiments.figs all
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from apa.config import MODELS_DIR

FIGS_DIR = Path(__file__).parent / "figs"
FIGS_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Loading helpers
# =============================================================================

def _load_prism_W(K: int = 8) -> tuple[np.ndarray, list[str]]:
    """Load PRISM seen-user W matrix and corresponding user IDs."""
    W = torch.load(MODELS_DIR / f"W_seen_K{K}.pt", map_location="cpu", weights_only=False)
    mapping_path = MODELS_DIR / "user_to_idx.json"
    if mapping_path.exists():
        user_to_idx = json.loads(mapping_path.read_text())
        idx_to_user = {v: k for k, v in user_to_idx.items()}
        ids = [idx_to_user.get(i, f"prism_user_{i}") for i in range(W.shape[0])]
    else:
        ids = [f"prism_user_{i}" for i in range(W.shape[0])]
    return W.float().numpy(), ids


def _load_adapted_W(path: Path) -> dict[str, np.ndarray]:
    """Return {user_id: w_vector} from a W_adapted_*.pt checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return {uid: data["w"].float().numpy() for uid, data in ckpt["users"].items()}


# =============================================================================
# Figure 1: user-weights grid
# =============================================================================

def fig_user_weights_grid(
    K: int = 8,
    adapted_path: Path | None = None,
    save: bool = True,
) -> Path:
    """
    Render a 6 x K grid of user weight vectors.

    Rows 0-1: first two PRISM users (blue palette).
    Rows 2-3: first two C016 historical users (purple palette).
    Rows 4-5: first two C020 historical users (pink palette).

    Color intensity scales with weight magnitude on a shared global
    vmin/vmax so that magnitudes are comparable across users.
    """
    adapted_path = adapted_path or (MODELS_DIR / "W_adapted_hist_C016_C020_filtered.pt")

    W_prism, prism_ids = _load_prism_W(K=K)
    adapted = _load_adapted_W(adapted_path)

    prism_rows = [(prism_ids[i], W_prism[i]) for i in range(2)]
    c016_rows = [(f"hist_C016_{i:02d}", adapted[f"hist_C016_{i:02d}"]) for i in range(2)]
    c020_rows = [(f"hist_C020_{i:02d}", adapted[f"hist_C020_{i:02d}"]) for i in range(2)]

    groups = [
        ("PRISM", prism_rows, "Blues"),
        ("C016",  c016_rows,  "Purples"),
        ("C020",  c020_rows,  "RdPu"),
    ]

    # Per-row normalization on |w|: weight magnitudes vary by orders of
    # magnitude across PRISM (tiny continuous) vs. adapted (near one-hot)
    # users, so a shared scale collapses one group to white. Each row is
    # mapped to [0, 1] by its own max |w|.
    norm = Normalize(vmin=0.0, vmax=1.0)

    n_rows = sum(len(rows) for _, rows, _ in groups)
    n_cols = K

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
    })

    fig, ax = plt.subplots(figsize=(0.7 * n_cols + 2.5, 0.6 * n_rows + 1.2))

    cell_colors = np.zeros((n_rows, n_cols, 4))
    row_labels: list[str] = []
    row_idx = 0
    group_spans: list[tuple[str, int, int, str]] = []  # (label, start, end, cmap)
    for label, rows, cmap_name in groups:
        cmap = mpl.colormaps[cmap_name]
        start = row_idx
        for uid, w in rows:
            scale = float(np.max(np.abs(w))) or 1.0
            cell_colors[row_idx] = cmap(norm(np.abs(w) / scale))
            row_labels.append(uid)
            row_idx += 1
        group_spans.append((label, start, row_idx, cmap_name))

    ax.imshow(cell_colors, aspect="equal", interpolation="nearest")

    # Cell borders
    for r in range(n_rows):
        for c in range(n_cols):
            ax.add_patch(plt.Rectangle(
                (c - 0.5, r - 0.5), 1, 1,
                fill=False, edgecolor="white", linewidth=1.2,
            ))

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([f"$b_{{{i+1}}}$" for i in range(n_cols)])
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Basis function")
    ax.tick_params(axis="both", which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Group brackets on the left
    trans = ax.get_yaxis_transform()
    bracket_x = -0.18
    label_x = -0.22
    for label, start, end, cmap_name in group_spans:
        y0, y1 = start - 0.4, end - 1 + 0.4
        ax.plot([bracket_x, bracket_x], [y0, y1],
                transform=trans, clip_on=False,
                color=mpl.colormaps[cmap_name](0.75), linewidth=2.5)
        ax.text(label_x, (y0 + y1) / 2, label,
                transform=trans, ha="right", va="center",
                fontsize=11, fontweight="bold",
                color=mpl.colormaps[cmap_name](0.85))

    # Shared colorbar (neutral grey) showing the per-row normalized scale
    sm = ScalarMappable(norm=norm, cmap="Greys")
    cbar = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.04, shrink=0.85)
    cbar.set_label(r"Relative weight magnitude  $|w_k| / \max_k |w_k|$",
                   rotation=90, labelpad=8)
    cbar.outline.set_visible(False)

    ax.set_title("User weight vectors over LoRe basis", pad=12)
    fig.tight_layout()

    out = FIGS_DIR / "user_weights_grid.pdf"
    if save:
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=200)
        plt.close(fig)
    return out


# =============================================================================
# CLI
# =============================================================================

FIGURES = {
    "user_weights_grid": fig_user_weights_grid,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper figures.")
    parser.add_argument("name", choices=list(FIGURES) + ["all"])
    parser.add_argument("--K", type=int, default=8)
    args = parser.parse_args()

    names = list(FIGURES) if args.name == "all" else [args.name]
    for name in names:
        out = FIGURES[name](K=args.K)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
