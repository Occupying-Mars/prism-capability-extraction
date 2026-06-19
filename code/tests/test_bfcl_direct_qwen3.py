from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bfcl_direct_qwen3 import canonical


def test_canonical_sorts_mixed_type_sets_without_crashing() -> None:
    assert canonical({1, "x"}) == ["x", 1]
