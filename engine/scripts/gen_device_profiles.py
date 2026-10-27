"""Generate the capture app's copy of the phone lens table from the engine's own.

ONE TABLE, TWO CONSUMERS, AND THEY MUST NOT DRIFT. The engine uses the table to bound the
focal search; the capture app uses it to know, before a single frame is recorded, whether
this phone has optical stabilisation (which decides how fast the user may sweep), which
quality preset it can carry, and whether to say anything about a LiDAR scanner it cannot
read. If the two copies disagreed, the app would coach for one lens while the engine solved
for another -- so the app's copy is generated, never written.

    engine/.venv/Scripts/python.exe engine/scripts/gen_device_profiles.py

Writes web/src/lib/capture/deviceProfiles.generated.ts. `engine/tests/test_devices.py`
checks the file is current, so a forgotten regeneration fails the suite rather than shipping.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ENGINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINE))

SOURCE = ENGINE / "biosphere" / "assets" / "device_profiles.json"
TARGET = ENGINE.parent / "web" / "src" / "lib" / "capture" / "deviceProfiles.generated.ts"

HEADER = '''/**
 * GENERATED FILE -- DO NOT EDIT.
 *
 * Written by engine/scripts/gen_device_profiles.py from
 * engine/biosphere/assets/device_profiles.json, which is the single source of truth for
 * what lens each phone has. The engine bounds its focal search with these numbers; the
 * capture app uses them to coach before recording. If you edit this file by hand the two
 * will disagree, the app will coach for one lens while the engine solves for another, and
 * nothing will warn you.
 *
 * `equivMm` is the 35mm-equivalent focal length on the DIAGONAL convention (CIPA, 43.267
 * mm), for the 4:3 frame the app records. See biosphere/pano/devices.py for the two unit
 * traps that make this wrong if you take a number from anywhere else.
 */

export interface PhoneLens {
  id: string;
  brand: string;
  model: string;
  tier: "flagship" | "upper-mid" | "mid" | "budget";
  equivMm: number;
  ois: boolean;
  lidar: boolean;
  /** Exact `navigator.userAgentData` model strings, lower-cased. Empty for every iPhone. */
  aliases: string[];
}

/** One iPhone screen, and the models that share it. Several, usually. */
export interface IosFingerprint {
  cssWidth: number;
  cssHeight: number;
  dpr: number;
  equivMm: number;
  /** "yes" only when every model behind this screen has the scanner. */
  lidar: "yes" | "no" | "maybe";
  models: string[];
}
'''


def main() -> int:
    raw = json.loads(SOURCE.read_text(encoding="utf-8"))
    lenses = []
    for d in raw["devices"]:
        lenses.append(
            {
                "id": d["id"],
                "brand": d.get("brand", ""),
                "model": d.get("model", ""),
                "tier": d.get("tier", "mid"),
                "equivMm": d["video_equiv_mm"],
                "ois": bool(d.get("ois", False)),
                "lidar": bool(d.get("lidar", False)),
                "aliases": list(d.get("aliases", [])),
            }
        )
    prints = [
        {
            "cssWidth": f["css_width"],
            "cssHeight": f["css_height"],
            "dpr": f["dpr"],
            "equivMm": f.get("equiv_mm", 26.0),
            "lidar": f.get("lidar", "maybe"),
            "models": list(f.get("models", [])),
        }
        for f in raw.get("ios_fingerprints", [])
    ]
    body = (
        HEADER
        + "\nexport const PHONE_LENSES: readonly PhoneLens[] = "
        + json.dumps(lenses, indent=2, ensure_ascii=False)
        + " as const;\n\nexport const IOS_FINGERPRINTS: readonly IosFingerprint[] = "
        + json.dumps(prints, indent=2, ensure_ascii=False)
        + " as const;\n"
    )
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(body, encoding="utf-8", newline="\n")
    print(f"wrote {len(lenses)} lenses and {len(prints)} fingerprints to {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
