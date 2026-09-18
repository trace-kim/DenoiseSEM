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
    write_difference_viewer(path, differences, valid, ("Raw", "Corrected", "Failed"), 'A < B & "C"')
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
    write_difference_viewer(path, differences, valid, ("Raw", "Translation", "Affine", "Failed", "Percentile"), "Browser regression")
    check = Path(__file__).with_name("viewer_browser_check.js").read_text(encoding="utf-8")
    html = path.read_text(encoding="utf-8").replace('</body>', '<script>' + check + '</script></body>')
    path.write_text(html, encoding="utf-8")
    result = subprocess.run([os.environ["SEM_NOISE_TEST_BROWSER"], "--headless=new", "--disable-gpu", "--no-first-run",
                             f"--user-data-dir={tmp_path / 'profile'}", "--virtual-time-budget=10000", "--dump-dom", path.as_uri()],
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    check = re.findall(r'<p id="browser-check">(.*?)</p>', result.stdout)
    assert len(check) == 1 and check[0].startswith("PASS:"), (check, result.stderr[-1500:])
