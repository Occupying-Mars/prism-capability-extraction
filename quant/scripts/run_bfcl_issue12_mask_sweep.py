#!/usr/bin/env python3
"""Run BFCL issue #12 candidate mask evals across visible GPUs.

Each candidate is evaluated by `code/scripts/bfcl_direct_qwen3.py eval-mask`
with a single visible GPU. This keeps the intervention path identical to prior
BFCL receipts while allowing multiple masks to run concurrently on an 8xB200
node.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-jsonl", type=Path, required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--pairs", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--python", default="python")
    p.add_argument("--repo-root", type=Path, default=Path("."))
    p.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    p.add_argument("--model", default="Qwen/Qwen3-8B")
    p.add_argument("--adapter", type=Path)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--limit", type=int)
    p.add_argument("--candidate-id", action="append", default=[])
    p.add_argument("--kind", action="append", default=[])
    p.add_argument("--max-candidates", type=int)
    p.add_argument("--no-candidates", action="store_true")
    p.add_argument("--include-full-anchor", action="store_true")
    p.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.out_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    devices = [item.strip() for item in args.devices.split(",") if item.strip()]
    if not devices:
        raise ValueError("no devices specified")

    candidates = read_jsonl(args.candidate_jsonl)
    if args.candidate_id:
        wanted = set(args.candidate_id)
        candidates = [row for row in candidates if row["candidate_id"] in wanted]
    if args.kind:
        kinds = set(args.kind)
        candidates = [row for row in candidates if row.get("kind") in kinds]
    if args.no_candidates:
        candidates = []
    elif args.max_candidates is not None:
        if args.max_candidates < 0:
            raise ValueError("--max-candidates must be non-negative")
        candidates = candidates[: args.max_candidates]

    jobs: list[dict[str, Any]] = []
    if args.include_full_anchor:
        jobs.append(
            {
                "candidate_id": "full_unmasked",
                "cmd_extra": [],
            }
        )
    for row in candidates:
        jobs.append(
            {
                "candidate_id": row["candidate_id"],
                "cmd_extra": [
                    "--attribution",
                    str(args.candidate_root / row["mask_path"]),
                    "--topk",
                    str(row["topk_for_eval"]),
                ],
            }
        )

    script = args.repo_root / "code" / "scripts" / "bfcl_direct_qwen3.py"
    running: dict[subprocess.Popen, dict[str, Any]] = {}
    finished: list[dict[str, Any]] = []
    queue = list(jobs)
    started_at = time.time()

    def start_job(job: dict[str, Any], device: str) -> subprocess.Popen:
        cid = job["candidate_id"]
        output = args.out_dir / f"{cid}.jsonl"
        if args.skip_existing and output.exists() and output.stat().st_size > 0:
            job["skipped_existing"] = True
            finished.append(job)
            return None  # type: ignore[return-value]
        cmd = [
            args.python,
            str(script),
            "eval-mask",
            "--pairs",
            str(args.pairs),
            "--output",
            str(output),
            "--model",
            args.model,
            "--dtype",
            args.dtype,
            "--device-map",
            "auto",
            "--batch-size",
            str(args.batch_size),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--bfcl-canonicalization-prompt",
            "--normalized",
        ]
        if args.adapter:
            cmd += ["--adapter", str(args.adapter)]
        if args.limit is not None:
            cmd += ["--limit", str(args.limit)]
        cmd += job["cmd_extra"]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = device
        log_path = log_dir / f"{cid}.log"
        log_f = log_path.open("w")
        proc = subprocess.Popen(cmd, cwd=args.repo_root, env=env, stdout=log_f, stderr=subprocess.STDOUT)
        job.update(
            {
                "device": device,
                "pid": proc.pid,
                "output": str(output),
                "log": str(log_path),
                "started_at": time.time(),
                "cmd": cmd,
            }
        )
        print(f"[start] {cid} device={device} pid={proc.pid}", flush=True)
        return proc

    free_devices = list(devices)
    while queue or running:
        while queue and free_devices:
            job = queue.pop(0)
            device = free_devices.pop(0)
            proc = start_job(job, device)
            if proc is None:
                free_devices.append(device)
            else:
                running[proc] = job
        time.sleep(5)
        for proc in list(running):
            ret = proc.poll()
            if ret is None:
                continue
            job = running.pop(proc)
            job["returncode"] = ret
            job["elapsed_s"] = time.time() - job["started_at"]
            finished.append(job)
            free_devices.append(job["device"])
            print(f"[done] {job['candidate_id']} rc={ret} elapsed={job['elapsed_s']:.1f}s", flush=True)
            if ret != 0:
                print(f"[fail] {job['candidate_id']} log={job['log']}", flush=True)

    summary = {
        "jobs": finished,
        "job_count": len(finished),
        "failed": [job for job in finished if job.get("returncode", 0) not in (0, None)],
        "elapsed_s": time.time() - started_at,
    }
    (args.out_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"job_count": summary["job_count"], "failed": len(summary["failed"]), "elapsed_s": summary["elapsed_s"]}, indent=2))
    if summary["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
