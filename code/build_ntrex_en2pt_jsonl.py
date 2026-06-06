"""Dump NTREX-128 en->pt to a jsonl for our gen+score pipeline.
Truncates to 1012 rows to match the FLORES devtest size for apples-to-apples."""
import json, sys
from pathlib import Path
from datasets import load_dataset

ds = load_dataset("mteb/NTREX", split="test")
print(f"NTREX-128 columns sample: {list(ds.column_names)[:8]} ...")
print(f"NTREX-128 size: {len(ds)} rows")

# Find en + pt column names
src_col = next((c for c in ds.column_names if c == "eng" or c == "eng_Latn" or c == "en"), None)
tgt_col = next((c for c in ds.column_names if c == "por" or c == "por_Latn" or c == "pt" or c == "por-BR" or c == "por_BR" or c == "por-PT"), None)
print(f"  src column: {src_col}")
print(f"  tgt column: {tgt_col}")

assert src_col and tgt_col, f"Could not auto-detect en/pt columns. Available: {ds.column_names}"

pairs = []
for row in ds:
    s, t = row.get(src_col), row.get(tgt_col)
    if not s or not t:
        continue
    pairs.append((s.strip(), t.strip()))
    if len(pairs) >= 1012:
        break

out = Path("data/ntrex_en2pt.jsonl")
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as f:
    for i, (en, pt) in enumerate(pairs):
        f.write(json.dumps({"id": i, "en": en, "pt": pt,
                            "category": "ntrex_test", "tag": "heldout"},
                           ensure_ascii=False) + "\n")
print(f"wrote {len(pairs)} pairs -> {out}")
