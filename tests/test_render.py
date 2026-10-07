"""Rendering colour-representation regression tests.

The renderer must keep straight (non-premultiplied) RGBA colours end to
end: interpolation and source-over compositing happen in premultiplied
space (so fully transparent pixels cannot bleed their hidden RGB into
rotated edges), and the canvas is unpremultiplied exactly once before
PNG encoding.  Writing premultiplied values into the PNG darkens every
semi-transparent pixel by a factor of its alpha; stacking layers then
darkens them again.
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


def _solid(bgr, alpha, size=20, box=(4, 16)):
    img = np.zeros((size, size, 4), np.uint8)
    img[box[0]:box[1], box[0]:box[1], :3] = bgr
    img[box[0]:box[1], box[0]:box[1], 3] = alpha
    return img


def test_single_semitransparent_fragment_keeps_straight_colour():
    """R=200, A=128 must render as R=200, A=128 (not the darkened R=100)."""
    img = _solid((0, 0, 200), 128)
    out = _decode(render_cluster({"A": img}, {"A": (0.0, np.zeros(2))}))
    solid = out[:, :, 3] == 128
    assert solid.any()
    assert set(out[:, :, 2][solid].ravel()) == {200}
    assert set(out[:, :, 0][solid].ravel()) == {0}
    assert set(out[:, :, 1][solid].ravel()) == {0}


def test_stacked_semitransparent_fragments_keep_colour_compose_alpha():
    """Two identical layers: colour unchanged, alpha follows source-over."""
    img = _solid((0, 0, 200), 128)
    poses = {"A": (0.0, np.zeros(2)), "B": (0.0, np.zeros(2))}
    out = _decode(render_cluster({"A": img, "B": img.copy()}, poses))
    covered = out[:, :, 3] > 0
    assert covered.any()
    assert set(out[:, :, 2][covered].ravel()) == {200}
    a = 128 / 255
    expect_a = round((a + (1 - a) * a) * 255)
    assert set(out[:, :, 3][covered].ravel()) == {expect_a}


def test_source_over_matches_analytic_composite():
    """Green over red, both alpha 128: classic source-over values."""
    red = _solid((0, 0, 200), 128)
    green = _solid((0, 200, 0), 128)
    poses = {"A": (0.0, np.zeros(2)), "B": (0.0, np.zeros(2))}
    out = _decode(render_cluster({"A": red, "B": green}, poses))
    a = 128 / 255
    out_a = a + (1 - a) * a
    exp_g = 200 * a / out_a
    exp_r = 200 * a * (1 - a) / out_a
    px = out[12, 12]  # interior of the 4..16 box (pad=8 offset cancels)
    assert abs(int(px[3]) - round(out_a * 255)) <= 1
    assert abs(int(px[1]) - round(exp_g)) <= 1
    assert abs(int(px[2]) - round(exp_r)) <= 1
    assert int(px[0]) == 0


def test_rotated_edges_not_polluted_by_hidden_rgb():
    """Fully transparent pixels carry junk RGB; rotation must not show it."""
    img = np.zeros((30, 30, 4), np.uint8)
    img[:, :, :3] = 250                      # hidden junk, alpha still 0
    img[8:22, 8:22, :3] = (0, 0, 200)
    img[8:22, 8:22, 3] = 255
    out = _decode(render_cluster({"A": img}, {"A": (0.4, np.zeros(2))}))
    edge = (out[:, :, 3] > 0) & (out[:, :, 3] < 255)
    assert edge.any()
    assert int(out[:, :, 0][edge].max()) <= 1   # no blue/green junk bleed
    assert int(out[:, :, 1][edge].max()) <= 1
    assert int(out[:, :, 2][edge].min()) >= 199  # red survives interpolation


def test_opaque_fragment_renders_pixel_identical_colour():
    img = _solid((90, 150, 210), 255, size=24, box=(2, 20))
    out = _decode(render_cluster({"A": img}, {"A": (0.0, np.zeros(2))}))
    solid = out[:, :, 3] == 255
    assert solid.any()
    for ch, val in enumerate((90, 150, 210)):
        assert set(out[:, :, ch][solid].ravel()) == {val}


def test_fully_transparent_region_stays_zero():
    img = _solid((90, 150, 210), 255, size=24, box=(2, 20))
    out = _decode(render_cluster({"A": img}, {"A": (0.0, np.zeros(2))}))
    transparent = out[:, :, 3] == 0
    assert transparent.any()
    assert int(out[transparent].max()) == 0


def test_post_assemblies_returns_correct_semitransparent_png():
    """End-to-end: a semi-transparent shard keeps its colour through POST."""
    img = _solid((0, 0, 200), 128)
    body = {"fragments": [fragment_payload("A", img)], "seams": []}
    r = client.post("/assemblies", json=body)
    assert r.status_code == 200, r.text
    js = r.json()
    assert len(js["clusters"]) == 1
    out = _decode(base64.b64decode(js["clusters"][0]["image_base64"]))
    solid = out[:, :, 3] == 128
    assert solid.any()
    assert set(out[:, :, 2][solid].ravel()) == {200}
