"""
Config loading.

- Several YAML files can be layered: later files override earlier ones
  (`--config configs/default.yaml --config configs/experiments/alphapose.yaml`).
- Single values can be overridden from the CLI: `-o pose.type=rtmpose`.
- Relative paths are resolved against the project root, so configs work on any
  machine without editing absolute paths.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable, List, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"

# Config keys whose string values are treated as filesystem paths.
PATH_KEYS = {"cfg", "ckpt", "weights", "video_dir", "out_root", "alphapose_root", "videos"}


def deep_update(dst: dict, src: dict) -> dict:
    for k, v in (src or {}).items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)
        else:
            dst[k] = copy.deepcopy(v)
    return dst


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_yaml(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def apply_override(cfg: dict, expr: str) -> None:
    """Apply 'a.b.c=value' (value parsed as YAML, so numbers/bools/lists work)."""
    if "=" not in expr:
        raise ValueError(f"Override must look like key.path=value, got '{expr}'")
    key, raw = expr.split("=", 1)
    node = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(raw)


def _looks_like_path(s: str) -> bool:
    return ("/" in s) or ("\\" in s) or s.startswith(".")


def resolve_path(value: str, base: Path = PROJECT_ROOT) -> str:
    """Resolve a relative path against `base`. Bare names (e.g. 'yolov8m.pt',
    'imagenet') are left alone unless such a file exists under `base`."""
    p = Path(value).expanduser()
    if p.is_absolute():
        return str(p)
    candidate = (base / p).resolve()
    if _looks_like_path(value) or candidate.exists():
        return str(candidate)
    return value


def _resolve_paths(node: Any) -> Any:
    if isinstance(node, dict):
        for k, v in node.items():
            if k in PATH_KEYS and isinstance(v, str) and v:
                node[k] = resolve_path(v)
            elif k in PATH_KEYS and isinstance(v, list):
                node[k] = [resolve_path(x) if isinstance(x, str) else x for x in v]
            else:
                _resolve_paths(v)
    elif isinstance(node, list):
        for x in node:
            _resolve_paths(x)
    return node


def load_config(paths: Optional[Iterable[str]] = None, overrides: Optional[List[str]] = None,
                include_default: bool = True) -> dict:
    cfg: dict = {}
    files = [Path(p) for p in (paths or [])]
    if include_default and DEFAULT_CONFIG not in [f.resolve() for f in files]:
        files.insert(0, DEFAULT_CONFIG)
    for f in files:
        deep_update(cfg, load_yaml(f))
    for o in overrides or []:
        apply_override(cfg, o)
    return _resolve_paths(cfg)


def component_cfg(section: dict) -> dict:
    """Flatten a stage section into builder kwargs.

    A section holds `type`, optional shared scalar keys, and one settings block
    per backend; only the block named by `type` is used:

        pose:
          type: alphapose          # <- the toggle
          batch_size: 64           # shared
          alphapose: {cfg: ..., ckpt: ...}
          rtmpose:   {cfg: ..., ckpt: ...}
    -> {"type": "alphapose", "batch_size": 64, "cfg": ..., "ckpt": ...}
    """
    section = dict(section or {})
    t = section.get("type")
    if t is None:
        raise ValueError(f"Config section needs a 'type': {section!r}")
    shared = {k: v for k, v in section.items() if k != "type" and not isinstance(v, dict)}
    block = section.get(str(t), section.get(str(t).lower(), {})) or {}
    return {"type": t, **shared, **block}


def run_name(cfg: dict) -> str:
    """'auto' -> <pose>_<reid>_<tracker>, e.g. 'alphapose_osnet_strongsort'."""
    name = str(cfg.get("run_name", "auto"))
    if name != "auto":
        return name
    parts = [str(cfg.get(s, {}).get("type", "none")).lower() for s in ("pose", "reid", "tracker")]
    return "_".join(parts)
