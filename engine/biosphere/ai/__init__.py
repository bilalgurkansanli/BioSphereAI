"""AI features that operate on a finished panorama.

Both modules here are classical computer vision today and report ``model_backed: false``
honestly. Each exposes a single swap-in indirection (``INPAINT_BACKEND``) so a learned model
can replace the deterministic path without touching the API.
"""

from __future__ import annotations

__all__: list[str] = []
