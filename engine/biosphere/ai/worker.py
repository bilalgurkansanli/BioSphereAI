"""The GPU service. Run this on the machine with the good card.

    python -m uvicorn biosphere.ai.worker:app --host 0.0.0.0 --port 8010

Then point the engine at it::

    BIOSPHERE_AI_WORKER_URL=http://<that machine>:8010

Nothing else moves. The media root, the database, the job queue and the web app all stay
where they are; only two model calls travel, and each is one image over a local network.

What it deliberately does not know
-----------------------------------
Panoramas, rooms, furniture, or what BioSphere considers part of a building. It answers two
questions about ordinary photographs:

``POST /segment``   this photo -> a PNG of ADE20K class ids, one byte per pixel
``POST /inpaint``   this photo + this mask -> the photo redrawn without the masked part

Keeping the vocabulary at that level is what makes it replaceable. The definition of a
room's shell lives in ``furniture_seg._ROOM_SHELL``, in the product, where it can be argued
about; swapping the model behind this service does not reopen that argument — and it has
already been swapped once: ``/inpaint`` answers from Stable Diffusion or from LaMa depending
on ``cyclic`` and on what this machine has, and no caller was changed for it. The choice is
made by ``backend.local_inpainter``, the same function the engine uses in-process, so the two
cannot disagree about which model answers which question.

Security posture
----------------
There is none, and that is a deployment constraint rather than an oversight: this binds to a
LAN address and takes unauthenticated image uploads. Run it on a trusted network. It is the
same posture as the engine's own dev server and the same reason `/telemetry` refuses to
answer outside development.
"""

from __future__ import annotations

import time
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from biosphere.core.logging import configure_logging, get_logger

__all__ = ["app", "create_worker_app"]

log = get_logger("biosphere.ai.worker")

#: Guard against a caller that sends a whole 8K panorama by mistake. The windows this
#: service is fed are about a megapixel; ten is generous and still refuses the accident.
_MAX_PIXELS = 10_000_000


def _read_image(data: bytes, flags: int) -> np.ndarray | None:
    if not data or len(data) > 64 * 1024 * 1024:
        return None
    image = cv2.imdecode(np.frombuffer(data, np.uint8), flags)
    if image is None or image.size == 0:
        return None
    if image.shape[0] * image.shape[1] > _MAX_PIXELS:
        return None
    return image


def create_worker_app() -> FastAPI:
    configure_logging()
    app = FastAPI(
        title="BioSphere AI worker",
        version="0.1.0",
        summary="Segmentation and inpainting, on whichever machine has the GPU.",
    )

    @app.get("/health")
    def health() -> JSONResponse:
        """What this machine can actually do, checked rather than assumed."""
        from biosphere.ai import backend, furniture_seg, sd_inpaint

        device = "cpu"
        name = None
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
                name = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001 - torch is an optional extra even here
            pass

        # `inpaint` stays the field the engine's `backend.ready` tests, and stays truthy for
        # the same reason it always was: something on this machine can fill a window. What
        # changed is that it may now be a diffusion model, so it names which.
        return JSONResponse(
            {
                "status": "ok",
                "device": device,
                "gpu": name,
                "segment": furniture_seg.backend_name(),
                "inpaint": backend.local_inpainter(cyclic=False),
                "inpaint_canvas": backend.local_inpainter(cyclic=True),
                "sd_checkpoint": sd_inpaint.model_ref(),
            }
        )

    @app.post("/segment")
    async def segment(image: Annotated[UploadFile, File()]) -> Response:
        """One photograph in, a map of ADE20K class ids out."""
        started = time.perf_counter()
        photo = _read_image(await image.read(), cv2.IMREAD_COLOR)
        if photo is None:
            return JSONResponse({"error": "unreadable or oversized image"}, status_code=422)

        from biosphere.ai import furniture_seg

        labels = furniture_seg.segment_photo_local(photo)
        if labels is None:
            return JSONResponse({"error": "no segmentation model on this machine"}, status_code=503)

        ok, buffer = cv2.imencode(".png", np.clip(labels, 0, 255).astype(np.uint8))
        if not ok:  # pragma: no cover
            return JSONResponse({"error": "could not encode the label map"}, status_code=500)
        log.info(
            "worker.segment",
            size=f"{photo.shape[1]}x{photo.shape[0]}",
            ms=int((time.perf_counter() - started) * 1000),
        )
        return Response(content=buffer.tobytes(), media_type="image/png")

    @app.post("/inpaint")
    async def inpaint(
        image: Annotated[UploadFile, File()],
        mask: Annotated[UploadFile, File()],
        cyclic: Annotated[str, Form()] = "1",
    ) -> Response:
        """A photograph and a mask in, the photograph redrawn out.

        ``cyclic`` says the image wraps horizontally — true for an equirect canvas, false
        for a perspective window. Getting it wrong glues one side of a room to the other.
        """
        started = time.perf_counter()
        photo = _read_image(await image.read(), cv2.IMREAD_COLOR)
        hole = _read_image(await mask.read(), cv2.IMREAD_GRAYSCALE)
        if photo is None or hole is None:
            return JSONResponse({"error": "unreadable or oversized image"}, status_code=422)
        if hole.shape[:2] != photo.shape[:2]:
            hole = cv2.resize(hole, (photo.shape[1], photo.shape[0]), interpolation=cv2.INTER_NEAREST)

        from biosphere.ai import backend

        wraps = cyclic not in ("0", "false", "False")
        # Same selector the engine would have used in-process, so moving the GPU work here
        # changes where the model runs and nothing about which model runs.
        chosen = backend.local_inpainter(cyclic=wraps)
        result = backend.local_inpaint(photo, hole, cyclic=wraps)
        if result is None:
            return JSONResponse({"error": "no inpainting model on this machine"}, status_code=503)

        ok, buffer = cv2.imencode(".png", result)
        if not ok:  # pragma: no cover
            return JSONResponse({"error": "could not encode the result"}, status_code=500)
        log.info(
            "worker.inpaint",
            model=chosen,
            cyclic=wraps,
            size=f"{photo.shape[1]}x{photo.shape[0]}",
            hole=round(float((hole > 127).mean()), 4),
            ms=int((time.perf_counter() - started) * 1000),
        )
        return Response(content=buffer.tobytes(), media_type="image/png")

    return app


app = create_worker_app()
