from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bfcl_attention_qwen3 import (
    install_attention_projection_hooks,
    make_boundary_swap_keep_mask,
    make_keep_mask,
    qk_keep_masks_from_attention_keep,
)


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


def test_boundary_swap_replaces_lowest_selected_batch_with_next_outside_batch() -> None:
    ov_scores = torch.tensor([[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)

    keep, info = make_boundary_swap_keep_mask(
        ov_scores=ov_scores,
        topk=4,
        swap_batch_size=2,
        remove_batch=0,
        add_batch=0,
    )

    assert keep.tolist() == [[True, True, False, False, True, True, False, False]]
    assert info["removed_rank_start"] == 2
    assert info["removed_rank_end_exclusive"] == 4
    assert info["added_rank_start"] == 4
    assert info["added_rank_end_exclusive"] == 6


def test_boundary_swap_can_probe_deeper_outside_batches() -> None:
    ov_scores = torch.tensor([[8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]], dtype=torch.float32)

    keep, info = make_boundary_swap_keep_mask(
        ov_scores=ov_scores,
        topk=4,
        swap_batch_size=2,
        remove_batch=0,
        add_batch=1,
    )

    assert keep.tolist() == [[True, True, False, False, False, False, True, True]]
    assert info["added_rank_start"] == 6
    assert info["added_rank_end_exclusive"] == 8


def test_qk_keep_maps_query_heads_to_grouped_kv_heads() -> None:
    keep = torch.zeros((1, 8), dtype=torch.bool)
    keep[0, 2:4] = True

    q_keep, k_keep, info = qk_keep_masks_from_attention_keep(
        keep,
        n_heads=4,
        n_kv_heads=2,
        head_dim=2,
    )

    assert q_keep.tolist() == keep.tolist()
    assert k_keep.tolist() == [[True, True, False, False]]
    assert info["q_heads_kept"] == 1
    assert info["kv_heads_kept"] == 1
    assert info["kv_group_size"] == 2


def test_kv_group_head_mask_keeps_whole_query_groups() -> None:
    head_scores = torch.tensor([[1.0, 2.0, 10.0, 20.0]], dtype=torch.float32)
    ov_scores = torch.zeros((1, 8), dtype=torch.float32)

    keep, info = make_keep_mask(
        head_scores=head_scores,
        ov_scores=ov_scores,
        unit="head",
        topk=2,
        random_seed=None,
        mask_strategy="kv-group",
        n_kv_heads=2,
    )

    assert keep.tolist() == [[False, False, False, False, True, True, True, True]]
    assert info["kept_kv_groups"] == 1
    assert info["kept_heads"] == 2
    assert info["kept_ov_channels"] == 4


def test_qkv_ov_projection_hooks_zero_qkv_and_ov_sites() -> None:
    class Attention:
        def __init__(self) -> None:
            self.q_proj = torch.nn.Identity()
            self.k_proj = torch.nn.Identity()
            self.v_proj = torch.nn.Identity()
            self.o_proj = torch.nn.Identity()

    class Layer:
        def __init__(self) -> None:
            self.self_attn = Attention()

    class Config:
        num_hidden_layers = 1
        num_attention_heads = 4
        num_key_value_heads = 2
        hidden_size = 8
        head_dim = 2

    class Model:
        def __init__(self) -> None:
            self.config = Config()
            self.layers = [Layer()]

    model = Model()
    keep = torch.tensor([[True, True, False, False, False, False, False, False]])
    hooks = install_attention_projection_hooks(model, keep, projection_sites="qkv-ov", ablation="zero", means=None)
    try:
        attn = model.layers[0].self_attn
        assert attn.q_proj(torch.ones(1, 1, 8)).tolist() == [[[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
        assert attn.k_proj(torch.ones(1, 1, 4)).tolist() == [[[1.0, 1.0, 0.0, 0.0]]]
        assert attn.v_proj(torch.ones(1, 1, 4)).tolist() == [[[1.0, 1.0, 0.0, 0.0]]]
        assert attn.o_proj(torch.ones(1, 1, 8)).tolist() == [[[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]]
    finally:
        for hook in hooks:
            hook.remove()
