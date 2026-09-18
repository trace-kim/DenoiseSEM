from __future__ import annotations

import base64
import gzip
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np
import pytest

pytest.importorskip("scipy")

from sem_noise.difference_viewer import write_difference_viewer


def read_viewer(path: Path) -> tuple[dict, np.ndarray]:
    html = path.read_text(encoding="utf-8")
    payload = json.loads(re.search(r'<script id="difference-data" type="application/json">(.*?)</script>', html).group(1))
    values = np.frombuffer(gzip.decompress(base64.b64decode(payload["data"])), dtype=f"<f{payload['bytes_per_value']}").reshape(payload["shape"])
    return payload, values


@pytest.mark.parametrize("scale", [0, 1e-100, 1e-6, 255, 65535, 1e8, 1e100])
def test_viewer_preserves_signed_native_differences_and_failures(tmp_path: Path, scale: float) -> None:
    base = np.linspace(-scale, scale, 23 * 31).reshape(23, 31)
    differences = [base.copy(), base * 0.37, np.full(base.shape, np.nan)]
    valid = np.ones(base.shape, dtype=bool)
    valid[1, 1] = False
    original = [delta.copy() for delta in differences]
    path = tmp_path / "viewer.html"
    write_difference_viewer(path, differences, valid, ("Raw", "Corrected", "Failed"), 'A < B & "C"', blur_sigma_px=0)
    payload, saved = read_viewer(path)
    assert saved.shape == (3, 23, 31)
    for exported, expected, untouched in zip(saved, differences, original):
        np.testing.assert_array_equal(exported[valid], expected.astype(saved.dtype)[valid])
        assert np.isnan(exported[~valid]).all()
        np.testing.assert_array_equal(expected, untouched)
    assert payload["maximum"] == pytest.approx(scale)
    html = path.read_text(encoding="utf-8")
    assert 'type="number" step="any"' in html and 'type="range"' in html
    assert 'fetch(' not in html and '<script src=' not in html
    assert 'A &lt; B &amp; &quot;C&quot;' in html


@pytest.mark.skipif(not os.environ.get("SEM_NOISE_TEST_BROWSER"), reason="set SEM_NOISE_TEST_BROWSER to a Chrome/Edge executable for offline UI checks")
def test_offline_browser_controls_and_native_pixel_export(tmp_path: Path) -> None:
    base = np.tile([-220, -5, -0.125, 0, 0.125, 5, 220, 65535], (8, 1)).astype(float)
    differences = [base, base * 0.5, np.zeros_like(base), np.full_like(base, np.nan), base * 0.01]
    valid = np.ones(base.shape, dtype=bool)
    valid[0, 0] = False
    path = tmp_path / "viewer.html"
    write_difference_viewer(path, differences, valid, ("Raw", "Translation", "Affine", "Failed", "Percentile"), "Browser regression", blur_sigma_px=0)
    check = Path(__file__).with_name("viewer_browser_check.js").read_text(encoding="utf-8")
    html = path.read_text(encoding="utf-8").replace('</body>', '<script>' + check + '</script></body>')
    path.write_text(html, encoding="utf-8")
    result = subprocess.run([os.environ["SEM_NOISE_TEST_BROWSER"], "--headless=new", "--disable-gpu", "--no-first-run",
                             f"--user-data-dir={tmp_path / 'profile'}", "--virtual-time-budget=10000", "--dump-dom", path.as_uri()],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    check = re.findall(r'<p id="browser-check">(.*?)</p>', result.stdout)
    assert len(check) == 1 and check[0].startswith("PASS:"), (check, result.stderr[-1500:])


def test_blurred_viewer_suppresses_noise_and_preserves_edge_displacement(tmp_path: Path) -> None:
    from scipy import ndimage

    rng = np.random.default_rng(3)
    fixed = np.zeros((96, 96))
    fixed[:, 48:] = 80
    moving = np.roll(fixed, 4, axis=1)
    a = fixed + rng.normal(0, 25, fixed.shape)
    b = moving + rng.normal(0, 25, fixed.shape)
    differences = [b - a, np.roll(b, -4, axis=1) - a, np.full_like(a, np.nan)]
    valid = np.ones(a.shape, dtype=bool)
    valid[20, 20] = False
    differences[0][20, 20] = 1e20  # must not leak through the Gaussian into valid pixels
    before = [d.copy() for d in differences]
    path = tmp_path / "blurred.html"
    write_difference_viewer(path, differences, valid, ("Raw", "Translation", "Failed"), "Blurred edges")
    payload, values = read_viewer(path)
    assert payload["blur_sigma_px"] == 2
    mask = ndimage.minimum_filter(valid, size=17, mode="constant", cval=0)
    np.testing.assert_array_equal(np.isfinite(values[0]), mask)
    np.testing.assert_array_equal(np.isfinite(values[1]), mask)
    expected = ndimage.gaussian_filter(b, 2) - ndimage.gaussian_filter(a, 2)
    np.testing.assert_allclose(values[0][mask], expected[mask], atol=4e-6)
    assert np.isnan(values[2]).all()
    flat = np.s_[35:65, 15:30]
    edge = np.s_[35:65, 47:54]
    assert values[0][flat].std() < (b - a)[flat].std() / 5
    assert np.sqrt(np.mean(values[1][edge] ** 2)) < np.sqrt(np.mean(values[0][edge] ** 2)) / 5
    for delta, original in zip(differences, before):
        np.testing.assert_array_equal(delta, original)
    assert "Gaussian-blurred" in path.read_text(encoding="utf-8")


@pytest.mark.parametrize("sigma", [-1, np.nan, np.inf])
def test_viewer_rejects_invalid_blur(tmp_path: Path, sigma: float) -> None:
    with pytest.raises(ValueError, match="sigma"):
        write_difference_viewer(tmp_path / "bad.html", [np.zeros((32, 32))], np.ones((32, 32), bool),
                                ("Raw",), "Invalid", blur_sigma_px=sigma)
