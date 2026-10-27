"""The pipeline in its own process.

Two of the libraries underneath (pycolmap's C++ core, the trainer's GPU runtime) end the
process when they hit an internal check, not the call; the API server cannot be that
process. So the job manager launches this module with a JSON spec, reads progress as JSON
lines from its stdout, and reads the manifest it wrote. Killing it is how a job is
cancelled hard, and a crash is a failed job rather than a dead service.

    python -m biosphere.splat.worker spec.json

Spec: ``{"base": {key, name, videos[]} | null, "rooms": [{key, name, videos[], floor_index}],
"quality": "fast|balanced|studio", "camera_height_m": 1.45, "keep_work": false,
"work_dir": ..., "out_dir": ..., "env": {name: value}}`` (``env`` fills environment
variables the parent's settings know and its environment may not, e.g. the models dir). Progress lines: ``{"stage": ..., "fraction": ..., "message": ...}``;
the last line is ``{"result": "<manifest path>"}`` or ``{"error": {...}}``.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    sys.stdout.flush()


class _Progress:
    def stage(self, name: str, fraction: float, message: str = "") -> None:
        _emit({"stage": name, "fraction": float(fraction), "message": message})

    def log(self, message: str, **fields) -> None:
        _emit({"log": message, "fields": {k: str(v) for k, v in fields.items()}})


def main(argv: list[str]) -> int:
    # The parent reads UTF-8; a Windows console child would otherwise write the code page
    # of the day and die on the first room named with a Turkish letter.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001 - not a real console
            pass
    if len(argv) != 2:
        _emit({"error": {"type": "usage", "message": "python -m biosphere.splat.worker spec.json"}})
        return 2
    spec = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    import os

    for key, value in (spec.get("env") or {}).items():
        os.environ.setdefault(str(key), str(value))  # the process's own environment wins
    try:
        from .pipeline import build_tour
        from .types import AreaInput, SplatConfig, SplatQuality

        def area(d: dict) -> AreaInput:
            return AreaInput(key=d["key"], name=d["name"], videos=tuple(Path(v) for v in d["videos"]), floor_index=int(d.get("floor_index", 0)))

        base = area(spec["base"]) if spec.get("base") else None
        rooms = [area(r) for r in spec.get("rooms", [])]
        config = SplatConfig(
            quality=SplatQuality.for_mode(spec.get("quality", "balanced")),
            camera_height_m=float(spec.get("camera_height_m", 1.45)),
            keep_work=bool(spec.get("keep_work", False)),
        )
        result = build_tour(base, rooms, config=config, work_dir=Path(spec["work_dir"]), out_dir=Path(spec["out_dir"]), progress=_Progress())
        _emit({"result": str(result.manifest_path)})
        return 0
    except Exception as exc:  # noqa: BLE001 - everything becomes a structured error line
        payload = {"type": type(exc).__name__, "message": str(exc)[:2000], "traceback": traceback.format_exc()[-4000:]}
        en = getattr(exc, "user_message_en", None)
        tr = getattr(exc, "user_message_tr", None)
        if en:
            payload["en"] = en
        if tr:
            payload["tr"] = tr
        _emit({"error": payload})
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
