"""A stand-in walkthrough worker that reports one stage and then never speaks again."""

import json
import sys
import time

print(json.dumps({"stage": "sfm", "fraction": 0.2, "message": "solving"}), flush=True)
sys.stderr.write("hung inside a native call\n")
sys.stderr.flush()
time.sleep(600)
