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


def render_cluster(images: dict, poses: dict, pad: int = 8) -> bytes:
    """Render all fragments in a cluster under one common pose to PNG bytes.

    Images are RGBA with **straight** (non-premultiplied) colours, and PNG
    stores straight colours too.  Bilinear interpolation and source-over
    compositing are both linear only in *premultiplied* colours
    ``(rgb * a, a)``, so each fragment is premultiplied before warping --
    fully transparent pixels then carry exactly zero colour and their
    stored (hidden) RGB cannot bleed into rotated edges -- and every layer
    is composited in premultiplied float space.  The accumulated canvas is
    unpremultiplied back to straight RGBA exactly once, after the last
    fragment; unpremultiplying earlier (or writing the premultiplied
    values straight into the PNG) darkens every semi-transparent pixel by
    a factor of its alpha, and stacking layers darkens them repeatedly.

    Fragments are painted ordered by fragment id so output is
    deterministic.
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

    # Premultiplied-alpha accumulators; source-over is a plain linear
    # combination in this representation.
    acc_rgb = np.zeros((H, W, 3), dtype=np.float64)
    acc_a = np.zeros((H, W, 1), dtype=np.float64)

    for fid in ids:
        theta, t = poses[fid]
        A = _affine(theta, t + np.array([-x0, -y0]))
        src = images[fid].astype(np.float32)
        src_a = src[:, :, 3:4] / 255.0
        premul = np.concatenate([src[:, :, :3] * src_a, src_a], axis=2)
        warped = cv2.warpAffine(
            premul, A, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )
        w_rgb = warped[:, :, :3].astype(np.float64)
        w_a = warped[:, :, 3:4].astype(np.float64)
        acc_rgb = w_rgb + acc_rgb * (1.0 - w_a)
        acc_a = w_a + acc_a * (1.0 - w_a)

    # Unpremultiply once, now that every layer is composited.  Fully
    # transparent pixels get zero colour; everywhere else the straight
    # colour is recovered by dividing the premultiplied accumulator by
    # the accumulated alpha.
    canvas = np.zeros((H, W, 4), dtype=np.uint8)
    solid = acc_a[:, :, 0] > 0.0
    straight = np.zeros((H, W, 3), dtype=np.float64)
    straight[solid] = acc_rgb[solid] / acc_a[solid]
    canvas[:, :, :3] = np.clip(np.rint(straight), 0, 255).astype(np.uint8)
    canvas[:, :, 3] = np.clip(np.rint(acc_a[:, :, 0] * 255.0), 0, 255).astype(np.uint8)

    ok, buf = cv2.imencode(".png", canvas)
    if not ok:
        raise RuntimeError("PNG encoding failed")
    return buf.tobytes()


def encode_png_base64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode("ascii")
