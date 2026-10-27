"""Walkthrough video -> 3D Gaussian Splatting property.

The 360 engine in :mod:`biosphere.pano` answers "what does the room look like from *here*";
this package answers "what does the room look like from *anywhere*". A phone video walked
through the space is turned into camera poses (structure from motion), the poses into a
trained radiance field (3D Gaussian splats), the splats into a compact streamable asset
(SOG) and a manifest the web viewer walks through.

Layout
------
types.py      contracts: quality presets, area inputs/results, the property frame
ply.py        the 3DGS PLY attribute table, read and written with numpy
raster.py     a torch rasteriser for thumbnails, environment panoramas and the floor plan
frames.py     video -> the frames worth posing (sharpness, spacing, count)
features.py   learned features and matches (ALIKED + LightGlue), written into a COLMAP database
sfm.py        pycolmap mapping (global first, incremental fallback), undistortion for training
register.py   one video's model into another's frame (Sim3 through shared images)
level.py      gravity, floor and metric scale
train.py      Brush, the trainer, run as a subprocess with its progress read back
clean.py      floaters, giants and out-of-room splats
export.py     SOG compression and the per-area pictures
bundle.py     the property manifest the viewer consumes
pipeline.py   the orchestration
worker.py     the orchestration in its own process, for the job queue
"""
