"""BioSphere AI — video-to-360 panorama engine.

Nothing heavy is imported here on purpose. The API process starts in well under a
second because OpenCV, SciPy and Torch are only pulled in when a stitch actually runs.
"""

from __future__ import annotations

__version__ = "0.1.0"
__all__ = ["__version__"]
