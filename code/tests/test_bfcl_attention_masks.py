from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bfcl_attention_qwen3 import make_keep_mask


def test_global_ov_mask_picks_highest_channels() -> None:
    head_scores = torch.zeros((2, 2))
    ov_scores = torch.tensor(
        [
            [0.1, 0.9, 0.2, 0.3],
            [0.4, 0.8, 0.7, 0.6],
        ],
        dtype=torch.float32,
    )

    keep, info = make_keep_mask(
        head_scores=head_scores,
        ov_scores=ov_scores,
        unit="ov",
        topk=3,
        random_seed=None,
    )

    assert info["mask_strategy"] == "global"
    assert info["kept_ov_channels"] == 3
    assert keep.tolist() == [[False, True, False, False], [False, True, True, False]]


def test_layer_balanced_ov_mask_respects_layer_floor() -> None:
    head_scores = torch.zeros((2, 2))
    ov_scores = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [10.0, 9.0, 8.0, 7.0],
        ],
        dtype=torch.float32,
    )

    keep, info = make_keep_mask(
        head_scores=head_scores,
        ov_scores=ov_scores,
        unit="ov",
        topk=3,
        random_seed=None,
        mask_strategy="layer-balanced",
        layer_floor=1,
    )

    assert info["layer_floor"] == 1
    assert keep[0].sum().item() == 1
    assert keep[1].sum().item() == 2
    assert keep[0, 3]
    assert keep[1, 0]
    assert keep[1, 1]


def test_head_scaffold_ov_restricts_channel_candidates_to_scaffold_heads() -> None:
    head_scores = torch.tensor([[5.0, 1.0], [0.5, 0.4]], dtype=torch.float32)
    ov_scores = torch.tensor(
        [
            [0.1, 0.2, 100.0, 99.0],
            [98.0, 97.0, 96.0, 95.0],
        ],
        dtype=torch.float32,
    )

    keep, info = make_keep_mask(
        head_scores=head_scores,
        ov_scores=ov_scores,
        unit="ov",
        topk=2,
        random_seed=None,
        mask_strategy="head-scaffold-ov",
        head_scaffold_topk=1,
    )

    assert info["head_scaffold_kept_heads"] == 1
    assert info["head_scaffold_candidate_ov_channels"] == 2
    assert keep.tolist() == [[True, True, False, False], [False, False, False, False]]

