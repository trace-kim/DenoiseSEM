"""CLI and output artifacts, exercised through the classical backend.

Everything here runs with no GPU, no network, and no model weights, which is the
point of shipping a no-download backend.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
import yaml
from synthetic_images import make_config, write_png, write_rgb_with_border
from typer.testing import CliRunner

from sem_segment.cli import app

runner = CliRunner()


def disk_image(size=140, width=220):
    from scipy.special import erf

    image = np.full((size, width), 0.15)
    yy, xx = np.mgrid[0:size, 0:width].astype(float)
    for cx, radius in ((60.0, 20.0), (150.0, 26.0)):
        distance = np.hypot(yy - 70.0, xx - cx)
        image = np.maximum(image, 0.15 + 0.7 * 0.5 * (1 + erf((radius - distance) / (np.sqrt(2) * 1.5))))
    return image


def write_config(tmp_path, **overrides):
    config = make_config(**overrides)
    path = tmp_path / "config.yml"
    path.write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True), encoding="utf-8")
    return path


def test_help_lists_every_command():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    for command in ("segment", "backends", "download-weights"):
        assert command in result.output


def test_backends_command_reports_readiness():
    result = runner.invoke(app, ["backends"])
    assert result.exit_code == 0, result.output
    assert "classical" in result.output and "ready" in result.output
    assert "signed in:" in result.output  # reports auth however it was set


def test_segment_writes_every_artifact(tmp_path):
    source = write_png(tmp_path / "in" / "disks.png", disk_image())
    config = write_config(tmp_path, report={"enabled": True, "zoom_insets": 1, "profile_samples": 2})
    result = runner.invoke(
        app, ["segment", "--config", str(config), "--input", str(source), "--out", str(tmp_path / "out")]
    )
    assert result.exit_code == 0, result.output

    out = tmp_path / "out" / "disks"
    for name in ("metrology.csv", "contours.json", "masks.npz", "summary.json", "provenance.json", "index.html"):
        assert (out / name).is_file(), f"missing {name}"

    rows = list(csv.DictReader((out / "metrology.csv").open(encoding="utf-8")))
    assert len(rows) == 2
    assert "cd_px_coarse" in rows[0] and "cd_px_refined" in rows[0]


def test_report_is_self_contained(tmp_path):
    """It is read after copying one file off a remote machine."""
    source = write_png(tmp_path / "in" / "disks.png", disk_image())
    config = write_config(tmp_path, report={"enabled": True, "zoom_insets": 1, "profile_samples": 2})
    runner.invoke(app, ["segment", "--config", str(config), "--input", str(source), "--out", str(tmp_path / "out")])

    html = (tmp_path / "out" / "disks" / "index.html").read_text(encoding="utf-8")
    assert "data:image/png;base64," in html
    assert 'src="http' not in html and "href=\"http" not in html
    assert "<style>" in html  # the stylesheet is inline too


def test_report_states_the_pixel_unit_caveat_when_no_scale_is_given(tmp_path):
    source = write_png(tmp_path / "in" / "disks.png", disk_image())
    config = write_config(tmp_path, report={"enabled": True, "zoom_insets": 0, "profile_samples": 0})
    runner.invoke(app, ["segment", "--config", str(config), "--input", str(source), "--out", str(tmp_path / "out")])
    html = (tmp_path / "out" / "disks" / "index.html").read_text(encoding="utf-8")
    assert "No pixel size was supplied" in html


def test_pixel_size_adds_nanometre_columns(tmp_path):
    source = write_png(tmp_path / "in" / "disks.png", disk_image())
    config = write_config(tmp_path)
    result = runner.invoke(
        app,
        ["segment", "--config", str(config), "--input", str(source), "--out", str(tmp_path / "out"),
         "--pixel-size-nm", "5.82812"],
    )
    assert result.exit_code == 0, result.output
    rows = list(csv.DictReader((tmp_path / "out" / "disks" / "metrology.csv").open(encoding="utf-8")))
    assert "cd_nm_refined" in rows[0]
    assert float(rows[0]["cd_nm_refined"]) == pytest.approx(
        float(rows[0]["cd_px_refined"]) * 5.82812, rel=1e-6
    )


def test_folder_input_writes_one_directory_per_image_and_no_rollup(tmp_path):
    """The package makes no assumption that images in a folder are related."""
    for name in ("a", "b", "c"):
        write_png(tmp_path / "in" / f"{name}.png", disk_image())
    config = write_config(tmp_path)
    result = runner.invoke(
        app, ["segment", "--config", str(config), "--input", str(tmp_path / "in"), "--out", str(tmp_path / "out")]
    )
    assert result.exit_code == 0, result.output

    children = sorted(p.name for p in (tmp_path / "out").iterdir())
    assert children == ["a", "b", "c"]
    assert all((tmp_path / "out" / c).is_dir() for c in children)
    # No cross-image summary.csv, index.html, or anything else at the top level.
    assert not [p for p in (tmp_path / "out").iterdir() if p.is_file()]


def test_rerun_refuses_to_overwrite_unless_asked(tmp_path):
    source = write_png(tmp_path / "in" / "disks.png", disk_image())
    config = write_config(tmp_path)
    args = ["segment", "--config", str(config), "--input", str(source), "--out", str(tmp_path / "out")]
    assert runner.invoke(app, args).exit_code == 0

    second = runner.invoke(app, args)
    assert second.exit_code == 1
    assert "already exists" in second.output and "--overwrite" in second.output

    assert runner.invoke(app, args + ["--overwrite"]).exit_code == 0


def test_crop_is_applied_before_the_grayscale_check(tmp_path):
    """Reproduces the real instrument export: a colored frame on grayscale data."""
    source = write_rgb_with_border(tmp_path / "in" / "framed.png", disk_image(), border=2)
    config = write_config(tmp_path)
    args = ["segment", "--config", str(config), "--input", str(source), "--out", str(tmp_path / "out")]

    without = runner.invoke(app, args)
    assert without.exit_code == 1
    assert "RGB channels differ" in without.output

    with_crop = runner.invoke(app, args + ["--crop", "2,138,2,218", "--overwrite"])
    assert with_crop.exit_code == 0, with_crop.output


def test_malformed_crop_is_rejected(tmp_path):
    source = write_png(tmp_path / "in" / "disks.png", disk_image())
    config = write_config(tmp_path)
    result = runner.invoke(
        app,
        ["segment", "--config", str(config), "--input", str(source), "--out", str(tmp_path / "out"),
         "--crop", "1,2,3"],
    )
    assert result.exit_code != 0
    assert "four integers" in result.output


def test_backend_override_reaches_the_provenance(tmp_path):
    source = write_png(tmp_path / "in" / "disks.png", disk_image())
    config = write_config(tmp_path, segmentation={"backend": "sam3_auto"})
    result = runner.invoke(
        app,
        ["segment", "--config", str(config), "--input", str(source), "--out", str(tmp_path / "out"),
         "--backend", "classical"],
    )
    assert result.exit_code == 0, result.output
    provenance = json.loads((tmp_path / "out" / "disks" / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["backend"]["backend"] == "classical"
    assert provenance["config"]["segmentation"]["backend"] == "classical"


def test_invalid_override_is_rejected_not_silently_applied(tmp_path):
    source = write_png(tmp_path / "in" / "disks.png", disk_image())
    config = write_config(tmp_path)
    result = runner.invoke(
        app,
        ["segment", "--config", str(config), "--input", str(source), "--out", str(tmp_path / "out"),
         "--estimator", "made_up"],
    )
    assert result.exit_code != 0


def test_missing_config_is_reported(tmp_path):
    result = runner.invoke(
        app, ["segment", "--config", str(tmp_path / "nope.yml"), "--input", str(tmp_path), "--out", str(tmp_path / "o")]
    )
    assert result.exit_code != 0
    assert "config not found" in result.output


def test_provenance_records_the_input_hashes(tmp_path):
    source = write_png(tmp_path / "in" / "disks.png", disk_image())
    config = write_config(tmp_path)
    runner.invoke(app, ["segment", "--config", str(config), "--input", str(source), "--out", str(tmp_path / "out")])
    provenance = json.loads((tmp_path / "out" / "disks" / "provenance.json").read_text(encoding="utf-8"))
    assert len(provenance["file_sha256"]) == 64
    assert len(provenance["pixel_sha256"]) == 64
    assert provenance["source"].endswith("disks.png")


def test_contours_json_keeps_both_methods_on_one_parametrisation(tmp_path):
    source = write_png(tmp_path / "in" / "disks.png", disk_image())
    config = write_config(tmp_path)
    runner.invoke(app, ["segment", "--config", str(config), "--input", str(source), "--out", str(tmp_path / "out")])
    payload = json.loads((tmp_path / "out" / "disks" / "contours.json").read_text(encoding="utf-8"))

    assert payload["coordinate_order"] == "yx"
    region = payload["regions"][0]
    # displacement is indexed against coarse, so the two line up vertex by vertex.
    assert len(region["displacement"]) == len(region["coarse"])
    assert len(region["valid"]) == len(region["coarse"])
    assert len(region["refined"]) == sum(region["valid"])


def test_masks_npz_round_trips(tmp_path):
    source = write_png(tmp_path / "in" / "disks.png", disk_image())
    config = write_config(tmp_path)
    runner.invoke(app, ["segment", "--config", str(config), "--input", str(source), "--out", str(tmp_path / "out")])
    data = np.load(tmp_path / "out" / "disks" / "masks.npz")
    assert data["labels"].dtype == np.uint16
    assert set(np.unique(data["labels"])) == {0, 1, 2}
    assert data["scores"].shape == (2,)
