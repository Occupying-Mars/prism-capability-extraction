"""Small shared W&B wiring for experiment scripts.

The helper intentionally keeps W&B as telemetry plus tiny receipt artifacts.
Large run artifacts should stay in the normal run directory or HF upload flow.
"""

from __future__ import annotations

import argparse
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class WandbSettings:
    mode: str
    project: str
    entity: str | None
    group: str | None
    job_type: str
    name: str | None
    tags: list[str]
    api_key_present: bool
    env_file: Path | None

    def public_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "project": self.project,
            "entity": self.entity,
            "group": self.group,
            "job_type": self.job_type,
            "name": self.name,
            "tags": self.tags,
            "api_key_present": self.api_key_present,
            "env_file": str(self.env_file) if self.env_file else None,
        }


def find_env_file(start: Path | None = None) -> Path | None:
    current = (start or Path.cwd()).resolve()
    search_roots = [current] if current.is_dir() else [current.parent]
    search_roots.extend(search_roots[0].parents)
    for root in search_roots:
        candidate = root / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_env_file(path: Path | None = None) -> tuple[dict[str, str], Path | None]:
    env_path = path or find_env_file()
    if env_path is None:
        return {}, None
    values: dict[str, str] = {}
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if value and value[0] in {"'", '"'}:
            try:
                value = shlex.split(value)[0]
            except ValueError:
                value = value.strip("'\"")
        else:
            value = value.split(" #", 1)[0].strip()
        values[key] = value
    return values, env_path


def add_wandb_args(
    parser: argparse.ArgumentParser,
    *,
    default_project: str = "prism-bfcl",
    default_group: str | None = None,
    default_job_type: str = "train",
    default_tags: str = "",
) -> None:
    parser.add_argument("--wandb-mode", choices=["auto", "online", "offline", "disabled"], default="auto")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-project", default=default_project)
    parser.add_argument("--wandb-group", default=default_group)
    parser.add_argument("--wandb-job-type", default=default_job_type)
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-tags", default=default_tags)
    parser.add_argument("--wandb-env-file", type=Path, help="Optional .env path. Defaults to nearest .env.")


def _first(name: str, dotenv: dict[str, str]) -> str | None:
    # Project convention: .env is the canonical local source, shell env is fallback.
    return dotenv.get(name) or os.environ.get(name)


def _split_tags(value: str | None) -> list[str]:
    if not value:
        return []
    return [tag.strip() for tag in value.split(",") if tag.strip()]


def wandb_settings_from_args(
    args: argparse.Namespace,
    *,
    default_project: str = "prism-bfcl",
    default_group: str | None = None,
    default_job_type: str = "train",
    default_name: str | None = None,
    default_tags: str = "",
) -> WandbSettings:
    dotenv, env_file = load_env_file(getattr(args, "wandb_env_file", None))
    api_key = _first("WANDB_API_KEY", dotenv)
    if api_key and not os.environ.get("WANDB_API_KEY"):
        os.environ["WANDB_API_KEY"] = api_key

    mode = getattr(args, "wandb_mode", "auto")
    if mode == "auto":
        mode = _first("WANDB_MODE", dotenv) or ("online" if api_key else "disabled")
    if mode == "online" and not api_key:
        print("[wandb] no WANDB_API_KEY in .env or environment; disabling", flush=True)
        mode = "disabled"

    project = getattr(args, "wandb_project", None) or _first("WANDB_PROJECT", dotenv) or default_project
    entity = getattr(args, "wandb_entity", None) or _first("WANDB_ENTITY", dotenv)
    group = getattr(args, "wandb_group", None) or _first("WANDB_GROUP", dotenv) or default_group
    job_type = getattr(args, "wandb_job_type", None) or _first("WANDB_JOB_TYPE", dotenv) or default_job_type
    name = getattr(args, "wandb_name", None) or _first("WANDB_NAME", dotenv) or default_name
    tags = _split_tags(getattr(args, "wandb_tags", None) or _first("WANDB_TAGS", dotenv) or default_tags)

    return WandbSettings(
        mode=mode,
        project=project,
        entity=entity,
        group=group,
        job_type=job_type,
        name=name,
        tags=tags,
        api_key_present=bool(api_key),
        env_file=env_file,
    )


def init_wandb_run(
    args: argparse.Namespace,
    *,
    config: dict[str, Any],
    default_project: str = "prism-bfcl",
    default_group: str | None = None,
    default_job_type: str = "train",
    default_name: str | None = None,
    default_tags: str = "",
):
    settings = wandb_settings_from_args(
        args,
        default_project=default_project,
        default_group=default_group,
        default_job_type=default_job_type,
        default_name=default_name,
        default_tags=default_tags,
    )
    config.setdefault("wandb", settings.public_dict())
    if settings.mode == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("W&B logging requested but wandb is not installed") from exc

    print(
        f"[wandb] mode={settings.mode} project={settings.project} "
        f"entity={settings.entity or '-'} group={settings.group or '-'}",
        flush=True,
    )
    return wandb.init(
        project=settings.project,
        entity=settings.entity,
        group=settings.group,
        job_type=settings.job_type,
        name=settings.name,
        mode=settings.mode,
        tags=settings.tags,
        config=config,
    )


def log_wandb_receipts(run, *, name: str, files: Iterable[Path], artifact_type: str = "run_receipts") -> None:
    if run is None:
        return
    import wandb

    artifact = wandb.Artifact(name, type=artifact_type)
    added = False
    for path in files:
        p = Path(path)
        if p.is_file():
            artifact.add_file(str(p))
            added = True
    if added:
        run.log_artifact(artifact)
