"""Alpha-correctness of cluster rendering (straight RGBA round-trip).

The PNG inputs carry *straight* (non-premultiplied) BGRA.  A single
semi-transparent shard must keep its straight RGB and alpha after the
affine warp; stacked shards must source-over blend without darkening; and
resampling a rotated edge must not bleed the RGB hidden in fully
transparent texels into partially covered pixels.
"""

import base64

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.masks import render_cluster
from tests.synth import fragment_payload

client = TestClient(app)


def _decode(png_bytes):
    return cv2.imdecode(np.frombuffer(png_bytes, np.uint8),
                        cv2.IMREAD_UNCHANGED)


def _render(images, poses):
    return _decode(render_cluster(images, poses))


def test_single_semitransparent_shard_keeps_straight_color():
    # Straight BGRA: R=200, A=128. Premultiplied output would darken R to
    # roughly 100; the fixed pipeline must write straight RGB back.
    img = np.zeros((6, 6, 4), np.uint8)
    img[:, :] = (0, 0, 200, 128)
    out = _render({"X": img}, {"X": (0.0, np.zeros(2))})
    px = out[out[:, :, 3] > 0]
    assert np.all(px[:, 3] == 128)
    assert np.all(px[:, 2] == 200)          # red preserved
    assert np.all(px[:, 0] == 0) and np.all(px[:, 1] == 0)


def test_stacked_semitransparent_shards_blend_without_darkening():
    # Two identical red shards, A=128 each, same footprint.
    # Source-over: alpha = 0.5 + 0.5*0.5 = 0.75 (~192); straight R stays 200.
    img = np.zeros((6, 6, 4), np.uint8)
    img[:, :] = (0, 0, 200, 128)
    out = _render(
        {"A": img, "B": img.copy()},
        {"A": (0.0, np.zeros(2)), "B": (0.0, np.zeros(2))})
    px = out[out[:, :, 3] > 0]
    assert np.all(np.abs(px[:, 3].astype(int) - 192) <= 1)
    assert np.all(px[:, 2] == 200)          # not darkened to ~150
    assert np.all(px[:, 0] == 0) and np.all(px[:, 1] == 0)


def test_rotated_edge_not_polluted_by_hidden_rgb():
    # Opaque red block embedded in fully transparent texels whose hidden
    # RGB is poison white. After rotation, edge texels are interpolated:
    # their straight colour must stay pure red with only alpha ramping.
    img = np.zeros((24, 24, 4), np.uint8)
    img[4:20, 4:20] = (0, 0, 200, 255)
    img[:, :, :3] = np.where(img[:, :, 3:4] == 0,
                             np.array([255, 255, 255], dtype=np.uint8),
                             img[:, :, :3])
    out = _render({"X": img}, {"X": (0.6, np.zeros(2))})
    edge = (out[:, :, 3] > 0) & (out[:, :, 3] < 255)
    assert edge.any()
    assert np.all(out[edge, 0] == 0) and np.all(out[edge, 1] == 0)
    assert np.all(out[edge, 2] == 200)
    # fully transparent surroundings carry no colour
    clear = out[out[:, :, 3] == 0]
    assert np.all(clear[:, :3] == 0)


def test_opaque_fragment_renders_opaque_unchanged():
    img = np.zeros((8, 8, 4), np.uint8)
    img[:, :] = (11, 22, 33, 255)
    out = _render({"X": img}, {"X": (0.0, np.zeros(2))})
    px = out[out[:, :, 3] > 0]
    assert np.all(px[:, 3] == 255)
    assert np.all(px[:, :3] == [11, 22, 33])


def test_api_semitransparent_shards_round_trip():
    """End-to-end POST: two matching semi-transparent shards come back with
    their straight colours preserved everywhere (single- and double-layer).
    """
    h = 60
    # Two vertical strips with a shared edge, R=200/G=100, A=128.
    mk = lambda x0, x1: _strip(h, x0, x1, bgr=(40, 100, 200), alpha=128)
    A = mk(0, 30)
    B = mk(30, 60)
    pairs = [{"a": [29.0, float(y)], "b": [0.0, float(y)]}
             for y in range(4, h - 4, 8)]
    body = {
        "fragments": [fragment_payload("A", A), fragment_payload("B", B)],
        "seams": [
            {"id": "s1", "fragment_a": "A", "fragment_b": "B",
             "score": 10.0, "pose_error_limit": 2.0,
             "overlap_limit": 10 ** 9, "pairs": pairs}
        ],
    }
    r = client.post("/assemblies", json=body)
    assert r.status_code == 200, r.text
    cl = r.json()["clusters"][0]
    out = cv2.imdecode(
        np.frombuffer(base64.b64decode(cl["image_base64"]), np.uint8),
        cv2.IMREAD_UNCHANGED)
    single = out[out[:, :, 3] == 128]
    assert len(single) > 0
    assert np.all(single[:, 2] == 200) and np.all(single[:, 1] == 100)
    # seam-overlap columns carry two layers (alpha ~192) but stay same color
    double = out[np.abs(out[:, :, 3].astype(int) - 192) <= 1]
    if len(double):
        assert np.all(double[:, 2] == 200) and np.all(double[:, 1] == 100)


def _strip(h, x0, x1, bgr, alpha):
    img = np.zeros((h, 60, 4), np.uint8)
    img[:, x0:x1, :3] = bgr
    img[:, x0:x1, 3] = alpha
    yy, xx = np.nonzero(img[:, :, 3])
    return img[yy.min():yy.max() + 1, xx.min():xx.max() + 1].copy()
