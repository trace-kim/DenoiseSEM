from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from tools import check_real_sem_equivalence as equivalence


@pytest.mark.parametrize("delta,passed", [(.001, True), (.012, False)])
def test_equivalence_measures_decoded_outputs_and_detects_quantized_differences(tmp_path, monkeypatch, delta, passed):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"mocked model")
    source = tmp_path / "source.png"
    Image.fromarray(np.full((20, 24, 3), 127, dtype=np.uint8)).save(source)
    calls = []
    class Model:
        config = SimpleNamespace(data=SimpleNamespace(black_level=0, white_level=255))
        image_size = 16
        def denoise(self, tensor, *, clip_output):
            return torch.full_like(tensor, 0 if len(calls) == 1 else delta)
    def load(*args, **kwargs):
        calls.append(kwargs["device"])
        return Model()
    monkeypatch.setattr(equivalence.Denoiser, "from_checkpoint", load)
    decoded = []
    original = equivalence.read_native
    def read(path):
        decoded.append(path)
        return original(path)
    monkeypatch.setattr(equivalence, "read_native", read)
    result = equivalence.check_checkpoint(checkpoint, [source], tmp_path / "output", device="cpu")
    assert result["passed"] is passed
    assert any(path.parent.name == "cpu" for path in decoded)
    assert any(path.parent.name == "candidate" for path in decoded)
    assert result["frames"][0]["changed_fraction"] == (0 if passed else 1)


def test_equivalence_rejects_unsafe_names_and_invalid_thresholds(tmp_path):
    assert equivalence.main(["--checkpoint", "../escape=file.pt", "--output-dir", str(tmp_path / "out")]) == 1
    assert equivalence.main(["--max-dn", "-1", "--output-dir", str(tmp_path / "out")]) == 1
    assert not (tmp_path / "out").exists()
