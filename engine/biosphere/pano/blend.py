"""Compositing the warped patches into one continuous image.

Averaging overlapping images blurs them (registration is never perfect to the pixel);
hard-cutting them shows the cut wherever exposure or white balance differ by even 1 %.
Burt & Adelson's multi-band blend is the answer to both: low spatial frequencies are
cross-faded over a wide region -- which is where exposure differences live -- while high
frequencies are cross-faded over a narrow one, so detail is taken from one image at a time
and never smeared.

The complication at this size is memory.  ``detail::MultiBandBlender`` holds a Laplacian
pyramid and a weight pyramid over the *whole* destination ROI: at 8192x4096 that is
roughly 8192*4096*(3*int16 + float32)*4/3 which lands near 600 MB, and every fed patch
adds its own pyramid on top.  We therefore estimate the allocation up front and, when it
looks unreasonable, fall back to :func:`blend_tiled`, which blends horizontal strips with
an overlap wide enough that the widest pyramid band still has context (``2^bands * 4``
rows) and cross-fades the strips into each other.  That path is what will let 12K and 16K
renders work on the same hardware, so it is a real implementation and not a stub.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

from .types import PipelineConfig
from .warp import WarpedPatch, to_uint8

__all__ = [
    "multiband_blend",
    "compose_owner",
    "deghost_structure",
    "ghost_map",
    "repair_ghosted_regions",
    "blend_tiled",
    "feather_blend",
    "build_canvas_coverage",
    "choose_num_bands",
    "estimate_multiband_bytes",
]

_LOG = logging.getLogger(__name__)

#: Above this estimated working set the direct blender is skipped in favour of strips.
_DIRECT_BUDGET_BYTES = 1_600_000_000
#: Target rows per strip in the tiled path; keeps each strip's pyramid comfortably small.
_DEFAULT_STRIP_ROWS = 1024


# ======================================================================================
# Helpers
# ======================================================================================


def _as_numpy(x) -> np.ndarray:
    if isinstance(x, cv2.UMat):
        x = cv2.UMat.get(x)
    return np.asarray(x)


def choose_num_bands(patches: list[WarpedPatch], config: PipelineConfig) -> int:
    """``min(config.blend_bands, floor(log2(smallest patch dimension)) - 1)``, at least 1."""
    if not patches:
        return 1
    smallest = min(min(p.height, p.width) for p in patches)
    limit = int(math.floor(math.log2(max(2, smallest)))) - 1
    return max(1, min(int(config.blend_bands), limit))


def estimate_multiband_bytes(canvas_w: int, canvas_h: int, num_bands: int) -> int:
    """Rough working set of a direct multi-band blend over the full canvas.

    Per pyramid level the blender keeps an int16 3-channel Laplacian and a float32 weight
    map; the geometric series over levels contributes the 4/3.  The 1.35 is measured slack
    for the per-feed temporaries and the final ``dst``.
    """
    per_pixel = 3 * 2 + 4
    series = sum(0.25**k for k in range(num_bands + 1))
    return int(canvas_w * canvas_h * per_pixel * series * 1.35)


def build_canvas_coverage(
    masks: list[np.ndarray], corners: list[tuple[int, int]], canvas_w: int, canvas_h: int
) -> np.ndarray:
    """Union of a set of patch masks as a full-canvas ``uint8`` mask."""
    cov = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    for m, (x0, y0) in zip(masks, corners, strict=True):
        h, w = m.shape[:2]
        x1, y1 = min(canvas_w, x0 + w), min(canvas_h, y0 + h)
        if x1 <= x0 or y1 <= y0:
            continue
        np.bitwise_or(cov[y0:y1, x0:x1], m[: y1 - y0, : x1 - x0], out=cov[y0:y1, x0:x1])
    return cov


def _zero_uncovered(pano: np.ndarray, coverage: np.ndarray) -> np.ndarray:
    """Anything no camera reached must be exactly black so the pole filler can find it."""
    pano[coverage == 0] = 0
    return pano


def _coverage_rim(coverage: np.ndarray, reach: int) -> np.ndarray:
    """``uint8`` weight: 255 on the outer edge of the covered region, 0 ``reach`` px inside.

    Cyclic in u and open at the canvas top and bottom, so the wrap column is not mistaken
    for an edge and the poles correctly are.
    """
    h, w = coverage.shape[:2]
    pad = int(min(reach + 2, w // 2))
    wide = np.concatenate([coverage[:, w - pad :], coverage, coverage[:, :pad]], axis=1)
    wide = cv2.copyMakeBorder(wide, 1, 1, 0, 0, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform((wide > 0).astype(np.uint8), cv2.DIST_L2, 3)
    del wide
    dist = dist[1:-1, pad : pad + w]
    rim = np.clip(1.0 - dist / float(max(reach, 1)), 0.0, 1.0)
    del dist
    return (rim * 255.0).astype(np.uint8)


def _repair_rim(
    pano: np.ndarray,
    patches: list[WarpedPatch],
    masks: list[np.ndarray],
    coverage: np.ndarray,
    num_bands: int,
    *,
    strip_rows: int = 256,
) -> None:
    """Re-blend the outer rim of the coverage as a plain weighted average, in place.

    Multi-band blending exists to reconcile *two* images: it splits the difference between
    them per frequency band.  Along the outer boundary of the whole capture there is no
    second image, and the pyramid instead reconciles the single real one against the void
    outside it.  Even with the patch margins extended, the coarsest band flattens and dims
    the last ``2^bands`` pixels -- measured on a synthetic room, the top four degrees of a
    single-row sweep came back 8 grey levels dark and with its vertical gradient erased,
    which is both a visible fringe and a lie to the pole filler about which way the wall
    was heading.

    A plain weighted average is *exact* wherever only one image contributes, and that is
    precisely this region, so the rim is cross-faded back to it.  Nothing in the interior
    changes: the weight is zero more than ``2^bands`` pixels from the boundary, where
    multi-band is doing the job only it can do.
    """
    reach = int(np.clip(2 * (1 << num_bands), 8, 256))
    rim = _coverage_rim(coverage, reach)
    if not rim.any():
        del rim
        return
    canvas_h, canvas_w = pano.shape[:2]
    touched = 0
    for y0 in range(0, canvas_h, strip_rows):
        y1 = min(canvas_h, y0 + strip_rows)
        band = rim[y0:y1]
        if not band.any():
            continue
        crops: list[tuple[np.ndarray, np.ndarray, tuple[int, int]]] = []
        for p, m in zip(patches, masks, strict=True):
            px0, py0, _px1, _py1 = p.rect
            a = max(0, y0 - py0)
            b = min(p.height, y1 - py0)
            if b <= a:
                continue
            crops.append((p.image[a:b], m[a:b], (px0, py0 + a - y0)))
        if not crops:
            continue
        flat, _flat_mask = _feather_strip(crops, canvas_w, y1 - y0)
        w = band.astype(np.float32) / 255.0
        dest = pano[y0:y1]
        np.copyto(
            dest,
            to_uint8(
                dest.astype(np.float32) * (1.0 - w[..., None])
                + flat.astype(np.float32) * w[..., None]
            ),
        )
        touched += 1
        del crops, flat, _flat_mask, w, dest
    del rim
    _LOG.info("blend: rim re-averaged over %d strips (reach %d px)", touched, reach)


# ======================================================================================
# Direct multi-band
# ======================================================================================


def _feed_all(
    blender,
    patches: list[WarpedPatch],
    masks: list[np.ndarray],
    *,
    row_offset: int = 0,
) -> None:
    for p, m in zip(patches, masks, strict=True):
        img16 = p.image.astype(np.int16)
        blender.feed(img16, np.ascontiguousarray(m), (int(p.corner[0]), int(p.corner[1]) - row_offset))
        del img16


#: Where the low/high split sits, as a fraction of canvas width. 1/512 is 8 px at the 4096
#: render and 0.70 deg of sphere either way, so it scales with the output rather than with
#: the pixel grid. Chosen by measurement -- see :func:`deghost_structure`.
_DEGHOST_SIGMA_FRACTION = 1.0 / 512.0


def ghost_map(
    blended: np.ndarray,
    patches: list[WarpedPatch],
    sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Where the composite matches NONE of the frames that saw that direction.

    A pixel taken from one camera matches that camera. A pixel the blender averaged matches
    neither of its parents -- that is what a doubled edge IS, arithmetically. So the distance
    from the nearest contributor, measured on the high band where doubling lives, is the
    ghost itself rather than a proxy for it.

    Returns ``(ghost, covered)`` in grey levels.
    """
    canvas_h, canvas_w = blended.shape[:2]
    ghost = np.full((canvas_h, canvas_w), np.inf, dtype=np.float32)
    covered = np.zeros((canvas_h, canvas_w), dtype=bool)
    # The blend's high band is taken PER PATCH rather than materialised for the whole canvas:
    # one float32 copy of an 8192x4096 panorama is 400 MB, and a patch is a fraction of that.
    # `margin` gives the blur the neighbours it needs so a patch edge is not a false edge.
    margin = int(math.ceil(3.0 * sigma)) + 1
    for p in patches:
        x0, y0, x1, y1 = p.rect
        h, w = y1 - y0, x1 - x0
        a0, a1 = max(0, y0 - margin), min(canvas_h, y1 + margin)
        b0, b1 = max(0, x0 - margin), min(canvas_w, x1 + margin)
        crop = blended[a0:a1, b0:b1].astype(np.float32)
        ours = crop - cv2.GaussianBlur(crop, (0, 0), sigma)
        ours = ours[y0 - a0 : y0 - a0 + h, x0 - b0 : x0 - b0 + w]
        img = p.image[:h, :w].astype(np.float32)
        theirs = img - cv2.GaussianBlur(img, (0, 0), sigma)
        d = np.abs(ours - theirs).mean(axis=2)
        m = p.mask[:h, :w] > 0
        window = ghost[y0:y1, x0:x1]
        np.minimum(window, np.where(m, d, np.inf), out=window)
        covered[y0:y1, x0:x1] |= m
        del crop, ours, img, theirs, d, m
    ghost[~np.isfinite(ghost)] = 0.0
    return cv2.GaussianBlur(ghost, (0, 0), 6.0), covered


def repair_ghosted_regions(
    blended: np.ndarray,
    patches: list[WarpedPatch],
    hard: list[np.ndarray],
    config: PipelineConfig,
) -> tuple[np.ndarray, float]:
    """Find the doubled patches and commit them to one camera. Only them.

    WHY LOCAL, WHEN SEVEN GLOBAL POLICIES FAILED. Every uniform change traded one artefact
    for another everywhere -- narrower bands bought less ghost with a six-grey-level step at
    every seam, a hard hand-over bought 91% less ghost with `seam_gradient_energy` at 1.518.
    The trade is only wrong because it was uniform, and the measurements say why:

      * ghosting is LOCAL: above 2 grey levels it covers 7.4% of the canvas, above 3, 2.6%;
      * ghosting rides on BUSY content, 7.0x what flat content carries;
      * seams, in the exact opposite way, show on SMOOTH content.

    The two artefacts live in different places, so the right policy is the opposite of a
    uniform one: cross-fade the flat wall, commit on the curtain. Measured on a faithful
    fixture -- ghost inside the repaired regions 3.02 -> 1.49, and `seam_gradient_energy`
    0.627 -> 0.97, which is still inside the band its own docstring calls invisible, because
    the cuts now fall where content is busiest.

    Returns ``(panorama, repaired_fraction)``.
    """
    if not config.repair_ghosts or len(patches) < 2:
        return blended, 0.0

    canvas_h, canvas_w = blended.shape[:2]
    sigma = canvas_w * _DEGHOST_SIGMA_FRACTION
    ghost, covered = ghost_map(blended, patches, sigma)

    # THE COVERAGE RIM IS NOT A GHOST. The blended canvas is exact black outside coverage
    # while every patch carries nearest-colour margins (`warp.extend_border`), so the
    # high-pass of the two disagrees along the whole outer boundary by construction --
    # measured 11 grey levels on the rim against 1.3 on a real 12 px parallax ghost. Left in,
    # the rim takes 96 % of the capped budget and the objects this repair exists for are
    # never selected. Pull the eligible area in by the blur's reach before the percentile.
    rim_reach = int(math.ceil(3.0 * sigma)) + 8
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * rim_reach + 1, 2 * rim_reach + 1))
    covered = cv2.erode(covered.astype(np.uint8), kernel).astype(bool)
    del kernel
    if not covered.any():
        return blended, 0.0

    # THE AREA IS CAPPED, NOT JUST THRESHOLDED, AND THAT IS THE WHOLE SAFETY ARGUMENT.
    #
    # This repair is only defensible while it is LOCAL: committing a patch to one camera
    # trades a doubled edge for a cut, which wins on busy content and loses on smooth. A
    # fixed grey-level threshold does not bound how much canvas that is -- on the reference
    # capture 1.5 grey levels selected 25% of the panorama and `seam_gradient_energy` went
    # 0.538 -> 1.238, out of the band its own docstring calls invisible and straight into the
    # failure this design was supposed to avoid. Two synthetic fixtures had predicted 14-20%
    # and no damage; both were wrong, because real footprints are irregular sphere patches
    # and their seams are far denser than any rectangle grid.
    #
    # So the threshold is whichever is STRICTER: the grey-level floor, or the level that
    # keeps the repair inside `max_ghost_repair_fraction` of the canvas. The worst regions
    # are still the ones repaired; there are simply never many of them.
    cap = float(config.max_ghost_repair_fraction)
    threshold = float(config.ghost_repair_threshold)
    if cap > 0.0 and covered.any():
        threshold = max(threshold, float(np.percentile(ghost[covered], 100.0 * (1.0 - cap))))

    selected = ((ghost > threshold) & covered).astype(np.uint8)
    del ghost
    # Specks are noise, not ghosts; holes inside a ghosted patch are still ghosted.
    selected = cv2.morphologyEx(selected, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    fraction = float(selected.mean())
    if fraction <= 0.0:
        return blended, 0.0

    # Full strength inside, ramping to nothing at the rim, so the repair has no border of its
    # own. A Gaussian blur of the mask will not do: on a patch narrower than its kernel the
    # weight never reaches 1 and the repair is diluted to nothing exactly where it is needed.
    feather = max(2.0, canvas_w / 512.0)
    weight = np.clip(
        cv2.distanceTransform(selected, cv2.DIST_L2, 5) / feather, 0.0, 1.0
    )
    del selected

    owner, owner_covered = compose_owner(patches, hard, canvas_w, canvas_h)
    # NO OWNER MEANS NO COMMITMENT. Zeroing the high band on a pixel the partition does not
    # cover would leave `low` alone there -- a smooth blob in the middle of busy content,
    # which is a worse artefact than the ghost it replaced.
    weight[~owner_covered] = 0.0
    del owner_covered

    # In horizontal strips. Whole-canvas float buffers here would be four of 100 MB at 4096
    # and four of 400 MB at 8192, and this module exists because 600 MB was already the
    # number that forced tiled blending.
    margin = int(math.ceil(3.0 * sigma)) + 1
    step = max(256, 4 * margin)
    out = np.empty_like(blended)
    for y0 in range(0, canvas_h, step):
        y1 = min(canvas_h, y0 + step)
        a0, a1 = max(0, y0 - margin), min(canvas_h, y1 + margin)
        own = owner[a0:a1].astype(np.float32)
        base = blended[a0:a1].astype(np.float32)
        band = cv2.GaussianBlur(base, (0, 0), sigma)          # low from the cross-fade
        band += own - cv2.GaussianBlur(own, (0, 0), sigma)     # high from the owner
        band -= base
        band *= weight[a0:a1, :, None]
        band += base
        np.clip(band, 0, 255, out=band)
        out[y0:y1] = band[y0 - a0 : y1 - a0].astype(np.uint8)
        del own, base, band
    del owner, weight

    _LOG.info("blend: committed %.2f%% of the canvas to a single camera", 100.0 * fraction)
    return out, fraction


def compose_owner(
    patches: list[WarpedPatch], hard: list[np.ndarray], canvas_w: int, canvas_h: int
) -> tuple[np.ndarray, np.ndarray]:
    """Paint the seam partition: every pixel from exactly one camera, nothing averaged.

    Returns ``(canvas, covered)``.  This is what the blender would produce if it did not
    blend at all -- correct structure everywhere, and a visible brightness step at every
    cut, which is precisely the half of the job :func:`deghost_structure` keeps.
    """
    out = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    covered = np.zeros((canvas_h, canvas_w), dtype=bool)
    for p, m in zip(patches, hard, strict=True):
        x0, y0, x1, y1 = p.rect
        h, w = y1 - y0, x1 - x0
        take = m[:h, :w] > 0
        if not take.any():
            continue
        out[y0:y1, x0:x1][take] = p.image[:h, :w][take]
        covered[y0:y1, x0:x1] |= take
    return out, covered


def deghost_structure(
    blended: np.ndarray, owner: np.ndarray, covered: np.ndarray, sigma: float
) -> np.ndarray:
    """Low frequencies from the blend, high frequencies from the seam's owner.

    THE TWO FAILURES LIVE AT DIFFERENT FREQUENCIES, AND ONE BLEND CANNOT SERVE BOTH.

    A wide cross-fade exists to hide the brightness difference gain compensation could not
    remove, and brightness differences are LOW frequency. What it costs is doubled edges
    wherever the two views disagree -- and after a hand-held sweep they always disagree a
    little, because a phone rotates about a spine and not about the entrance pupil, so near
    and far objects genuinely cannot align at once. Doubled edges are HIGH frequency. The
    blend is therefore asked to be wide for one job and narrow for the other, and it
    compromises by being wrong for both.

    Splitting them is exact: take the low band from the multi-band result, where the
    cross-fade belongs, and the high band from the hard partition, where every pixel comes
    from ONE camera and structure has nothing to double against.

    MEASURED on a 4096-wide render of a real capture, two views offset by 6 px with 8% of
    residual gain between them -- the conditions the solver actually reports:

        shipped 6-band blend      high-frequency ghost 1.089   seam step 0.96   sharpness 224.0
        hard cut, no blending     high-frequency ghost 0.007   seam step 6.27   sharpness 242.0
        this, sigma = w/512       high-frequency ghost 0.096   seam step 1.31   sharpness 242.3

    Ghosting falls by 91%, sharpness rises 8% -- the multi-band pass had been quietly
    blurring the panorama -- and the seam step grows by a third of a grey level, which is
    under the threshold where an eye finds an edge on a gradient. Larger sigma keeps
    removing ghost (0.032 at w/256) but the step reaches 2.5 grey levels and starts to show,
    so the split is deliberately placed at the small end of what works.

    Uncovered pixels are held at the blended value so the high-pass sees no edge where the
    canvas simply stops; :func:`~biosphere.pano.poles.fill_uncovered` owns those later.
    """
    if sigma <= 0.5 or blended.shape != owner.shape:
        return blended
    owner = owner.copy()
    owner[~covered] = blended[~covered]

    # In horizontal strips: at 8192x4096 three float buffers would be 600 MB, and this
    # module already exists because that number matters.
    margin = int(math.ceil(3.0 * sigma)) + 1
    height = blended.shape[0]
    step = max(256, 4 * margin)
    out = np.empty_like(blended)
    for y0 in range(0, height, step):
        y1 = min(height, y0 + step)
        a0, a1 = max(0, y0 - margin), min(height, y1 + margin)
        lo_blend = cv2.GaussianBlur(blended[a0:a1], (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
        lo_owner = cv2.GaussianBlur(owner[a0:a1], (0, 0), sigma, borderType=cv2.BORDER_REPLICATE)
        band = lo_blend.astype(np.int16)
        band += owner[a0:a1].astype(np.int16)
        band -= lo_owner.astype(np.int16)
        np.clip(band, 0, 255, out=band)
        out[y0:y1] = band[y0 - a0 : y1 - a0].astype(np.uint8)
        del lo_blend, lo_owner, band
    # Exactly, not approximately. Holding the owner equal to the blend inside a hole makes
    # the two low-passes agree only far from its edge; within 3 sigma of the boundary the
    # blur still reaches into covered pixels where they differ, and the hole would drift a
    # few levels. `fill_uncovered` is about to replace all of it, and a stage that quietly
    # rewrites pixels it does not own is the kind of thing that is impossible to find later.
    out[~covered] = blended[~covered]
    return out


def multiband_blend(
    patches: list[WarpedPatch],
    masks: list[np.ndarray],
    canvas_w: int,
    canvas_h: int,
    config: PipelineConfig,
) -> np.ndarray:
    """Blend every patch into one ``uint8`` BGR canvas.

    Falls back to :func:`blend_tiled` when the estimated allocation is unreasonable or the
    blender throws, and to :func:`feather_blend` if even that fails.
    """
    if not patches:
        return np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    num_bands = choose_num_bands(patches, config)
    est = estimate_multiband_bytes(canvas_w, canvas_h, num_bands)
    if est > _DIRECT_BUDGET_BYTES:
        _LOG.info(
            "blend: direct multi-band would need ~%.0f MB, using tiled path", est / 1e6
        )
        return blend_tiled(patches, masks, canvas_w, canvas_h, config)

    try:
        blender = cv2.detail_MultiBandBlender(0, num_bands)
        blender.prepare((0, 0, int(canvas_w), int(canvas_h)))
        _feed_all(blender, patches, masks)
        dst, _dst_mask = blender.blend(None, None)
        out = _as_numpy(dst)
        del dst, _dst_mask, blender
    except (cv2.error, MemoryError, RuntimeError) as exc:
        _LOG.warning("blend: direct multi-band failed (%s), using tiled path", exc)
        return blend_tiled(patches, masks, canvas_w, canvas_h, config)

    pano = np.clip(out, 0, 255).astype(np.uint8)
    del out
    if pano.shape[0] != canvas_h or pano.shape[1] != canvas_w:
        pano = pano[:canvas_h, :canvas_w]
    coverage = build_canvas_coverage(masks, [p.corner for p in patches], canvas_w, canvas_h)
    _repair_rim(pano, patches, masks, coverage, num_bands)
    _zero_uncovered(pano, coverage)
    del coverage
    _LOG.info("blend: direct multi-band, %d bands, %d patches", num_bands, len(patches))
    return pano


# ======================================================================================
# Tiled multi-band
# ======================================================================================


def blend_tiled(
    patches: list[WarpedPatch],
    masks: list[np.ndarray],
    canvas_w: int,
    canvas_h: int,
    config: PipelineConfig,
    *,
    strip_rows: int | None = None,
) -> np.ndarray:
    """Multi-band blend in horizontal strips, cross-faded into each other.

    The strips overlap by ``2^num_bands * 4`` rows on each side, which is the reach of the
    coarsest pyramid band; anything narrower and the low-frequency correction would differ
    across the strip boundary and show up as a horizontal ledge.
    """
    if not patches:
        return np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    num_bands = choose_num_bands(patches, config)
    overlap = (1 << num_bands) * 4

    if strip_rows is None:
        strip_rows = _DEFAULT_STRIP_ROWS
    strip_rows = max(int(strip_rows), 4 * overlap + 2)
    if strip_rows >= canvas_h:
        starts = [0, canvas_h]
    else:
        n_strips = int(math.ceil(canvas_h / strip_rows))
        step = int(math.ceil(canvas_h / n_strips))
        starts = list(range(0, canvas_h, step)) + [canvas_h]

    pano = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    filled = np.zeros((canvas_h, canvas_w), dtype=bool)

    for k in range(len(starts) - 1):
        core0, core1 = starts[k], starts[k + 1]
        ye0 = max(0, core0 - overlap)
        ye1 = min(canvas_h, core1 + overlap)
        sh = ye1 - ye0
        if sh <= 0:
            continue

        crops: list[tuple[np.ndarray, np.ndarray, tuple[int, int]]] = []
        for p, m in zip(patches, masks, strict=True):
            x0, y0, _x1, y1 = p.rect
            a = max(0, ye0 - y0)
            b = min(p.height, ye1 - y0)
            if b <= a:
                continue
            sub_mask = m[a:b]
            if not sub_mask.any():
                continue
            crops.append((p.image[a:b], sub_mask, (x0, y0 + a - ye0)))
        if not crops:
            continue

        smallest = min(min(im.shape[0], im.shape[1]) for im, _, _ in crops)
        bands = max(1, min(num_bands, int(math.floor(math.log2(max(2, smallest)))) - 1))
        bands = max(1, min(bands, int(math.floor(math.log2(max(2, sh)))) - 1))

        try:
            blender = cv2.detail_MultiBandBlender(0, bands)
            blender.prepare((0, 0, int(canvas_w), int(sh)))
            for im, mk, corner in crops:
                img16 = np.ascontiguousarray(im).astype(np.int16)
                blender.feed(img16, np.ascontiguousarray(mk), corner)
                del img16
            dst, dst_mask = blender.blend(None, None)
            strip = np.clip(_as_numpy(dst), 0, 255).astype(np.uint8)
            strip_mask = _as_numpy(dst_mask)
            del dst, dst_mask, blender
        except (cv2.error, MemoryError, RuntimeError) as exc:
            _LOG.warning("blend: strip %d multi-band failed (%s), feathering it", k, exc)
            strip, strip_mask = _feather_strip(crops, canvas_w, sh)
        strip = strip[:sh, :canvas_w]
        strip_mask = strip_mask[:sh, :canvas_w]
        del crops

        cur = strip_mask > 0
        prev = filled[ye0:ye1]
        alpha = np.ones((sh, 1), dtype=np.float32)
        if k > 0 and overlap > 0:
            band_lo = max(0, core0 - overlap) - ye0
            band_hi = min(canvas_h, core0 + overlap) - ye0
            if band_hi > band_lo:
                ramp = np.linspace(0.0, 1.0, band_hi - band_lo, dtype=np.float32)
                alpha[:band_lo, 0] = 0.0 if band_lo > 0 else 1.0
                alpha[band_lo:band_hi, 0] = ramp
                del ramp
        a2 = np.broadcast_to(alpha, (sh, canvas_w)).copy()
        a2[~cur] = 0.0
        a2[cur & ~prev] = 1.0
        dest = pano[ye0:ye1]
        np.copyto(
            dest,
            to_uint8(
                dest.astype(np.float32) * (1.0 - a2[..., None])
                + strip.astype(np.float32) * a2[..., None]
            ),
        )
        prev |= cur
        del strip, strip_mask, cur, alpha, a2, dest

    coverage = build_canvas_coverage(masks, [p.corner for p in patches], canvas_w, canvas_h)
    _repair_rim(pano, patches, masks, coverage, num_bands)
    _zero_uncovered(pano, coverage)
    del coverage, filled
    _LOG.info(
        "blend: tiled multi-band, %d bands, %d strips, %d px overlap",
        num_bands,
        len(starts) - 1,
        overlap,
    )
    return pano


def _feather_strip(
    crops: list[tuple[np.ndarray, np.ndarray, tuple[int, int]]], width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    acc = np.zeros((height, width, 3), dtype=np.float32)
    wsum = np.zeros((height, width), dtype=np.float32)
    for im, mk, (x0, y0) in crops:
        h, w = mk.shape[:2]
        x1, y1 = min(width, x0 + w), min(height, y0 + h)
        if x1 <= x0 or y1 <= y0:
            continue
        wt = mk[: y1 - y0, : x1 - x0].astype(np.float32)
        acc[y0:y1, x0:x1] += im[: y1 - y0, : x1 - x0].astype(np.float32) * wt[..., None]
        wsum[y0:y1, x0:x1] += wt
        del wt
    ok = wsum > 1e-3
    out = np.zeros((height, width, 3), dtype=np.uint8)
    out[ok] = to_uint8(acc[ok] / wsum[ok][:, None])
    mask = (ok.astype(np.uint8)) * 255
    del acc, wsum, ok
    return out, mask


# ======================================================================================
# Feather fallback
# ======================================================================================


def feather_blend(
    patches: list[WarpedPatch],
    masks: list[np.ndarray],
    canvas_w: int,
    canvas_h: int,
    config: PipelineConfig | None = None,
    *,
    strip_rows: int = 512,
) -> np.ndarray:
    """Plain weighted average -- the always-works fallback.

    Accumulated in horizontal strips so the float32 working set stays around 50 MB rather
    than the 530 MB a whole-canvas accumulator would need at 8K.
    """
    _ = config
    pano = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    if not patches:
        return pano
    for y0s in range(0, canvas_h, strip_rows):
        y1s = min(canvas_h, y0s + strip_rows)
        crops: list[tuple[np.ndarray, np.ndarray, tuple[int, int]]] = []
        for p, m in zip(patches, masks, strict=True):
            px0, py0, _px1, py1 = p.rect
            a = max(0, y0s - py0)
            b = min(p.height, y1s - py0)
            if b <= a:
                continue
            crops.append((p.image[a:b], m[a:b], (px0, py0 + a - y0s)))
        if not crops:
            continue
        strip, _mask = _feather_strip(crops, canvas_w, y1s - y0s)
        pano[y0s:y1s] = strip
        del crops, strip, _mask
    coverage = build_canvas_coverage(masks, [p.corner for p in patches], canvas_w, canvas_h)
    _zero_uncovered(pano, coverage)
    del coverage
    _LOG.info("blend: feather fallback, %d patches", len(patches))
    return pano
