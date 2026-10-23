"""
Command line:

  python -m posepipe list
  python -m posepipe show   [--config ...] [-o key=value ...]
  python -m posepipe run    [--config ...] [-o key=value ...] [--out DIR]
  python -m posepipe stitch --root OUT_DIR [--config ...]
  python -m posepipe evaluate --root OUT_DIR
  python -m posepipe plots    --root OUT_DIR [--config-label NAME]
  python -m posepipe study    [--config ...] [--pose alphapose] --variant both --experiments all
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml


def _add_cfg_args(p):
    p.add_argument("--config", "-c", action="append", default=[],
                   help="YAML file(s) layered over configs/default.yaml (repeatable)")
    p.add_argument("-o", "--set", dest="overrides", action="append", default=[],
                   help="Override a key, e.g. -o pose.type=alphapose (repeatable)")


def _load(args):
    from .config import load_config
    return load_config(args.config, args.overrides)


def cmd_list(_args):
    from .detection import DETECTORS
    from .pose import POSE
    from .reid import REID
    from .tracking import TRACKERS
    for reg in (DETECTORS, POSE, REID, TRACKERS):
        print(reg.describe())


def cmd_show(args):
    from .config import component_cfg, run_name
    cfg = _load(args)
    print(f"run_name: {run_name(cfg)}")
    for s in ("detector", "pose", "reid", "tracker"):
        print(yaml.safe_dump({s: component_cfg(cfg[s])}, sort_keys=False).rstrip())
    print(yaml.safe_dump({k: cfg[k] for k in ("pipeline", "crosscam") if k in cfg}, sort_keys=False))


def cmd_run(args):
    from .runner import run_session
    cfg = _load(args)
    run_session(cfg, Path(args.out) if args.out else None)


def cmd_stitch(args):
    from .runner import list_videos, run_crosscam
    cfg = _load(args)
    run_crosscam(Path(args.root), cfg, list_videos(cfg))


def cmd_evaluate(args):
    from .evaluation.metrics import run_eval
    run_eval(args.root, args.global_file)


def _passthrough(module: str, argv) -> None:
    """`plots` / `study` keep their own argparse; forward the raw arguments."""
    import importlib
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    mod = importlib.import_module(f"posepipe.evaluation.{module}")
    sys.argv = [f"posepipe {module}"] + list(argv)
    mod.main()


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] in ("plots", "study"):
        return _passthrough(argv[0], argv[1:])

    ap = argparse.ArgumentParser(prog="posepipe", description="Modular pose estimation + tracking pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list registered components").set_defaults(fn=cmd_list)

    p = sub.add_parser("show", help="print the resolved configuration")
    _add_cfg_args(p)
    p.set_defaults(fn=cmd_show)

    p = sub.add_parser("run", help="run the full pipeline on all videos")
    _add_cfg_args(p)
    p.add_argument("--out", help="output dir (default: <out_root>/<run_name>)")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("stitch", help="re-run cross-camera stitching on existing outputs")
    _add_cfg_args(p)
    p.add_argument("--root", required=True)
    p.set_defaults(fn=cmd_stitch)

    p = sub.add_parser("evaluate", help="unsupervised metrics for one run directory")
    p.add_argument("--root", required=True)
    p.add_argument("--global-file", default="merged_global_tracks_union.jsonl")
    p.set_defaults(fn=cmd_evaluate)

    sub.add_parser("plots", help="plots from evaluate's CSVs (see: posepipe plots -h)")
    sub.add_parser("study", help="parameter study (see: posepipe study -h)")

    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args.fn(args)


if __name__ == "__main__":
    main()
