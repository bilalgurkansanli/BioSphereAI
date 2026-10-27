"""A stand-in walkthrough worker that writes megabytes to stderr before it reports success."""

import json
import sys
from pathlib import Path

spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(json.dumps({"stage": "extract", "fraction": 0.5, "message": "frames"}), flush=True)
# Far more than any pipe buffer holds: a parent that does not drain stderr never sees the end.
for i in range(40_000):
    sys.stderr.write(f"I0906 12:00:00.000000 0 mapper.cc:1] Registering image #{i} (with a padding line of text)\n")
sys.stderr.flush()
print(json.dumps({"stage": "train", "fraction": 1.0, "message": "done"}), flush=True)
out = Path(spec["out_dir"])
out.mkdir(parents=True, exist_ok=True)
(out / "tour.json").write_text(json.dumps({"manifest": "ok"}), encoding="utf-8")
print(json.dumps({"result": str(out / "tour.json")}), flush=True)
