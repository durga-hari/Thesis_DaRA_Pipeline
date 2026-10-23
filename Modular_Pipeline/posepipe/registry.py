"""
Component registries.

Every pluggable stage (detector, pose estimator, ReID extractor, tracker) has a
Registry. Implementations are registered *lazily* as "module:attr" strings, so
choosing `pose.type: alphapose` never imports MMPose and vice versa. This is
what lets each backend run in an environment that only has its own deps.

    POSE.register_lazy("rtmpose", "posepipe.pose.rtmpose:RTMPoseEstimator")
    est = POSE.build({"type": "rtmpose", "cfg": ..., "ckpt": ...})
"""

from __future__ import annotations

import importlib
from typing import Any, Callable, Dict, Iterable, Optional


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._targets: Dict[str, Any] = {}      # name -> callable or "module:attr"
        self._aliases: Dict[str, str] = {}      # alias -> name
        self._defaults: Dict[str, dict] = {}    # name -> default kwargs

    # ------------------------------------------------------------------
    def register(self, name: str, aliases: Iterable[str] = (), defaults: Optional[dict] = None):
        """Decorator form: @REG.register("name")"""
        def deco(obj: Callable):
            self._add(name, obj, aliases, defaults)
            return obj
        return deco

    def register_lazy(self, name: str, target: str, aliases: Iterable[str] = (),
                      defaults: Optional[dict] = None) -> None:
        self._add(name, target, aliases, defaults)

    def _add(self, name, target, aliases, defaults):
        name = name.lower()
        self._targets[name] = target
        self._defaults[name] = dict(defaults or {})
        for a in aliases:
            self._aliases[a.lower()] = name

    # ------------------------------------------------------------------
    def canonical(self, name: str) -> str:
        n = str(name).lower()
        n = self._aliases.get(n, n)
        if n not in self._targets:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {', '.join(self.names())}"
            )
        return n

    def resolve(self, name: str) -> Callable:
        n = self.canonical(name)
        target = self._targets[n]
        if isinstance(target, str):
            mod_name, attr = target.split(":")
            target = getattr(importlib.import_module(mod_name), attr)
            self._targets[n] = target
        return target

    def build(self, cfg: dict, **extra) -> Any:
        """Build from a config dict: {"type": <name>, **kwargs}."""
        if not isinstance(cfg, dict) or "type" not in cfg:
            raise ValueError(f"{self.kind} config needs a 'type' key, got: {cfg!r}")
        n = self.canonical(cfg["type"])
        kwargs = {**self._defaults[n], **{k: v for k, v in cfg.items() if k != "type"}, **extra}
        return self.resolve(n)(**kwargs)

    def names(self):
        return sorted(self._targets)

    def describe(self) -> str:
        rows = []
        for n in self.names():
            al = sorted(a for a, t in self._aliases.items() if t == n)
            rows.append(f"  {n}" + (f"  (aliases: {', '.join(al)})" if al else ""))
        return f"{self.kind}:\n" + "\n".join(rows)
