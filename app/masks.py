"""Mask rasterisation, physical-overlap measurement and PNG rendering."""

from __future__ import annotations

import base64

import cv2
import numpy as np

from .geometry import rotation_matrix


def _affine(theta: float, t: np.ndarray) -> np.ndarray:
    """2x3 affine mapping local coordinates to target coordinates."""
    A = np.zeros((2, 3), dtype=np.float64)
    A[:2, :2] = rotation_matrix(theta)
    A[:2, 2] = np.asarray(t, dtype=np.float64).ravel()
    return A


def overlap_pixels(mask_a: np.ndarray, mask_b: np.ndarray,
                   theta_ba: float, t_ba: np.ndarray) -> int:
    """Count occupied pixels of fragment B landing on occupied pixels of A.

    ``p_a = R(theta_ba) p_b + t_ba`` maps B-local pixels into A's image
    frame.  B is warped once (nearest neighbour) onto A's canvas; pixels
    falling outside A's canvas cannot overlap A and are ignored.
    """
    h, w = mask_a.shape
    warped = cv2.warpAffine(
        mask_b.astype(np.uint8),
        _affine(theta_ba, t_ba),
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return int(np.count_nonzero((mask_a > 0) & (warped > 0)))


def _warped_corners(mask: np.ndarray, theta: float, t: np.ndarray):
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros((0, 2))
    R = rotation_matrix(theta)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    return pts @ R.T + np.asarray(t).ravel()


def _premultiplied(img_bgra: np.ndarray) -> np.ndarray:
    """Straight (non-premultiplied) BGRA uint8 -> premultiplied float64.

    Returns ``(B*a, G*a, R*a, a)`` with colour channels scaled to 0..255
    and coverage ``a`` in 0..1.  Premultiplying zeroes the RGB hidden in
    fully transparent texels, so bilinear resampling at rotated edges can
    never bleed that junk colour into partially covered pixels.
    """
    a = img_bgra[:, :, 3:4].astype(np.float64) / 255.0
    prem = np.empty((*img_bgra.shape[:2], 4), dtype=np.float64)
    prem[:, :, :3] = img_bgra[:, :, :3].astype(np.float64) * a
    prem[:, :, 3:4] = a
    return prem


def render_cluster(images: dict, poses: dict, pad: int = 8) -> bytes:
    """Render all fragments in a cluster under one common pose to PNG bytes.

    Decoded images are straight-alpha BGRA.  Resampling and source-over
    compositing happen in *premultiplied* colour, where bilinear
    interpolation and Porter-Duff "over" are both linear and exact: the
    straight RGB of a single semi-transparent shard is preserved, stacked
    shards blend without darkening, and rotated edges are not polluted by
    the RGB hidden in fully transparent texels.  Straight RGB is written
    back only once, when the final PNG is encoded.
    """
    ids = sorted(images)
    corners = []
    for fid in ids:
        c = _warped_corners(images[fid][:, :, 3] > 0, poses[fid][0], poses[fid][1])
        if c.shape[0]:
            corners.append(c)
    if not corners:
        canvas = np.zeros((1, 1, 4), dtype=np.uint8)
        ok, buf = cv2.imencode(".png", canvas)
        return buf.tobytes()

    allc = np.concatenate(corners, axis=0)
    x0 = int(np.floor(allc[:, 0].min())) - pad
    y0 = int(np.floor(allc[:, 1].min())) - pad
    x1 = int(np.ceil(allc[:, 0].max())) + pad
    y1 = int(np.ceil(allc[:, 1].max())) + pad
    W, H = x1 - x0, y1 - y0

    # Accumulator kept in premultiplied float: channels B,G,R are
    # premultiplied colour (0..255), channel A is coverage (0..1).
    acc = np.zeros((H, W, 4), dtype=np.float64)

    for fid in ids:
        theta, t = poses[fid]
        A = _affine(theta, t + np.array([-x0, -y0]))
        warped = cv2.warpAffine(
            _premultiplied(images[fid]), A, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0.0, 0.0, 0.0, 0.0),
        )
        src_rgb, src_a = warped[:, :, :3], warped[:, :, 3:4]
        # Premultiplied source-over: co = cs + cb*(1-as),
        # ao = as + ab*(1-as).
        inv_a = 1.0 - src_a
        acc[:, :, :3] = src_rgb + acc[:, :, :3] * inv_a
        acc[:, :, 3:4] = src_a + acc[:, :, 3:4] * inv_a

    # Convert back to straight-alpha BGRA exactly once, on encoding.
    coverage = acc[:, :, 3:4]
    with np.errstate(divide="ignore", invalid="ignore"):
        straight_rgb = acc[:, :, :3] / coverage
    straight_rgb = np.where(coverage > 1e-12, straight_rgb, 0.0)
    canvas = np.empty((H, W, 4), dtype=np.uint8)
    canvas[:, :, :3] = np.clip(np.rint(straight_rgb), 0, 255).astype(np.uint8)
    canvas[:, :, 3] = np.clip(np.rint(coverage[:, :, 0] * 255.0),
                              0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def encode_png_base64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")
