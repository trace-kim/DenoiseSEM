from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from tools.real_sem_experiment import SETTINGS, png_pixels, resolve_settings, run


def test_defaults_follow_checkpoint_and_output(tmp_path):
    settings = {**SETTINGS, "checkpoint": "runs/edge_denoise/example/ckpt_latest.pt"}
    resolved = resolve_settings(settings, tmp_path)
    assert resolved["source_dir"] == tmp_path / "data/SEM-test/example"
    assert resolved["png_dir"] == tmp_path / "output/example/png"
    resolved = resolve_settings({**settings, "output_dir": "output/another"}, tmp_path)
    assert resolved["raw_report_dir"] == tmp_path / "output/another/noise-raw"
    with pytest.raises(ValueError, match="Set checkpoint"):
        resolve_settings(SETTINGS, tmp_path)


def test_conversion_restores_fixed_levels_without_contrast_stretch():
    result = png_pixels(np.array([[0.25, 0.5, 0.75]], dtype=np.float32), 20, 220)
    np.testing.assert_array_equal(result, [[70, 120, 170]])
    assert result.dtype == np.uint8


@pytest.mark.parametrize("white", [255, 200])
def test_workflow_matches_raw_and_png_and_loads_once(tmp_path, monkeypatch, white):
    from edge_denoise.infer import Denoiser
    import sem_noise.pipeline

    source = tmp_path / "source"
    source.mkdir()
    for index in range(8):
        Image.fromarray(np.full((16, 16), 50 + index, dtype=np.uint8)).save(source / f"f{index}.png")
    checkpoint = tmp_path / "run" / "ckpt.pt"
    checkpoint.parent.mkdir()
    checkpoint.touch()
    output = tmp_path / "output"
    calls = []
    fake = SimpleNamespace(
        config=SimpleNamespace(data=SimpleNamespace(black_level=0, white_level=white)),
        image_size=8,
        load_measurement=lambda path: np.asarray(Image.open(path), dtype=np.float32) / white,
        denoise_full=lambda frame, **kwargs: frame,
    )

    def load(*args, **kwargs):
        calls.append("load")
        return fake

    monkeypatch.setattr(Denoiser, "from_checkpoint", load)

    def analyze(folder, report, **kwargs):
        calls.append(folder.name)
        paths = sorted(folder.glob("*.png"))
        assert len(paths) == 8
        for index, path in enumerate(paths):
            np.testing.assert_array_equal(np.asarray(Image.open(path)), np.full((16, 16), 50 + index))
        report.mkdir()
        return {"status": "complete"}

    monkeypatch.setattr(sem_noise.pipeline, "analyze_dataset", analyze)
    settings = {**SETTINGS, "checkpoint": str(checkpoint), "source_dir": str(source), "output_dir": str(output)}
    run(settings)
    assert calls == ["load", "raw", "png"]
    assert len(list((output / "inference").glob("*.tif"))) == 8
    with pytest.raises(ValueError, match="already exists"):
        run(settings)


def test_duplicate_stems_rejected_before_loading(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(8):
        Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(source / f"f{index}.png")
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8)).save(source / "f0.tif")
    checkpoint = tmp_path / "ckpt.pt"
    checkpoint.touch()
    with pytest.raises(ValueError, match="unique stems"):
        run({**SETTINGS, "checkpoint": str(checkpoint), "source_dir": str(source), "output_dir": str(tmp_path / "output")})
