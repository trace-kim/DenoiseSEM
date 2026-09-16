"""Backend registry, the classical segmenter, and the SAM 3 adapters.

The SAM 3 adapters are exercised against a fake ``transformers`` module, so the
output-shape normalisation for all three prompting modes is covered without ever
downloading 0.9 B parameters or needing an account.
"""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
from synthetic_images import blurred_disk

from sem_segment.backends import (
    BackendUnavailable,
    InstanceMask,
    available_backends,
    build_segmenter,
)
from sem_segment.config import Config, SegmentationConfig
from sem_segment.image_io import to_model_rgb


def three_disks(size=160, width=260):
    from scipy.special import erf

    image = np.full((size, width), 0.15)
    yy, xx = np.mgrid[0:size, 0:width].astype(float)
    for cx, radius in ((50.0, 18.0), (130.0, 24.0), (215.0, 12.0)):
        distance = np.hypot(yy - 80.0, xx - cx)
        image = np.maximum(image, 0.15 + 0.7 * 0.5 * (1 + erf((radius - distance) / (np.sqrt(2) * 1.5))))
    return image


def test_registry_lists_every_backend():
    names = available_backends()
    assert {"classical", "sam3_auto", "sam3_text", "sam3_prompt"} <= set(names)


def test_unknown_backend_is_rejected_with_the_available_list():
    config = SegmentationConfig(backend="classical")
    object.__setattr__(config, "backend", "nope")  # bypass the Literal for this check
    with pytest.raises(BackendUnavailable, match="unknown backend"):
        build_segmenter(config)


def test_classical_backend_finds_every_disk():
    config = Config.model_validate(
        {"segmentation": {"backend": "classical", "contrast_stretch": None}}
    )
    instances = build_segmenter(config).segment(to_model_rgb(three_disks(), contrast_stretch=None))
    assert len(instances) == 3
    radii = sorted(np.sqrt(m.area / np.pi) for m in instances)
    assert radii == pytest.approx([12.0, 18.0, 24.0], abs=0.3)


def test_classical_backend_handles_a_blank_image():
    config = Config.model_validate({"segmentation": {"backend": "classical"}})
    assert build_segmenter(config).segment(np.zeros((32, 32, 3), dtype=np.uint8)) == []


def test_polarity_auto_picks_features_even_when_they_cover_most_of_the_frame():
    """The failure this heuristic replaced: bright features over half the image.

    A "features are the minority" rule inverts here and segments the gaps
    between the features instead, which looks plausible and is entirely wrong.
    """
    from scipy.special import erf

    # Four disks on a 100 px grid with radius 46: an 8 px gap keeps them
    # separate components, while still covering 66% of the frame.
    image = np.full((200, 200), 0.1)
    yy, xx = np.mgrid[0:200, 0:200].astype(float)
    for cy, cx in ((50.0, 50.0), (50.0, 150.0), (150.0, 50.0), (150.0, 150.0)):
        distance = np.hypot(yy - cy, xx - cx)
        image = np.maximum(image, 0.1 + 0.8 * 0.5 * (1 + erf((46.0 - distance) / (np.sqrt(2) * 1.5))))
    assert (image > 0.5).mean() > 0.5  # features really do dominate the frame

    config = Config.model_validate(
        {"segmentation": {"backend": "classical", "contrast_stretch": None, "split_touching": False}}
    )
    instances = build_segmenter(config).segment(to_model_rgb(image, contrast_stretch=None))
    assert len(instances) == 4
    for instance in instances:
        assert instance.meta["polarity"] == "bright"
        assert np.sqrt(instance.area / np.pi) == pytest.approx(46.0, abs=1.5)


def test_polarity_auto_finds_dark_holes_in_a_bright_matrix():
    """The mirror case, which a fixed 'bright is foreground' rule would miss."""
    from scipy.special import erf

    image = np.full((200, 200), 0.85)
    yy, xx = np.mgrid[0:200, 0:200].astype(float)
    for cy, cx in ((70.0, 70.0), (70.0, 140.0), (140.0, 105.0)):
        distance = np.hypot(yy - cy, xx - cx)
        image = np.minimum(image, 0.85 - 0.7 * 0.5 * (1 + erf((22.0 - distance) / (np.sqrt(2) * 1.5))))
    config = Config.model_validate(
        {"segmentation": {"backend": "classical", "contrast_stretch": None}}
    )
    instances = build_segmenter(config).segment(to_model_rgb(image, contrast_stretch=None))
    assert len(instances) == 3
    assert all(m.meta["polarity"] == "dark" for m in instances)


def test_polarity_can_be_forced():
    config = Config.model_validate(
        {"segmentation": {"backend": "classical", "contrast_stretch": None, "polarity": "dark"}}
    )
    instances = build_segmenter(config).segment(to_model_rgb(three_disks(), contrast_stretch=None))
    assert all(m.meta["polarity"] == "dark" for m in instances)


# --------------------------------------------------------------------------
# SAM 3 adapters against a fake transformers module.
# --------------------------------------------------------------------------


def _canned_masks(shape=(64, 96), count=2):
    masks = []
    for index in range(count):
        mask = np.zeros(shape, dtype=bool)
        mask[10 + index * 20 : 25 + index * 20, 10 + index * 25 : 30 + index * 25] = True
        masks.append(mask)
    return masks


@pytest.fixture()
def fake_transformers(monkeypatch):
    """A stand-in exposing exactly the SAM 3 surface the adapters call."""
    import torch

    shape = (64, 96)
    module = types.ModuleType("transformers")
    module.__version__ = "5.14.1"

    class _Inputs(dict):
        def to(self, *_args, **_kwargs):
            return self

        def get(self, key, default=None):
            if key == "original_sizes":
                return torch.tensor([[shape[0], shape[1]]])
            return super().get(key, default)

    class _Processor:
        @classmethod
        def from_pretrained(cls, *_a, **_k):
            return cls()

        def __call__(self, **kwargs):
            inputs = _Inputs(kwargs)
            inputs["original_sizes"] = torch.tensor([[shape[0], shape[1]]])
            return inputs

        def post_process_instance_segmentation(self, outputs, **_k):
            return [{"masks": _canned_masks(shape), "scores": np.array([0.9, 0.8]),
                     "boxes": np.zeros((2, 4))}]

        def post_process_masks(self, masks, *_a, **_k):
            return [torch.as_tensor(np.stack([np.stack(_canned_masks(shape))] * 1)[0])[:, None]
                    .repeat(1, 3, 1, 1)]

    class _Model:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, *_a, **_k):
            return cls()

        def to(self, *_a, **_k):
            return self

        def eval(self):
            return self

        def __call__(self, **_kwargs):
            out = types.SimpleNamespace()
            out.pred_masks = torch.zeros(2, 3, 16, 16)
            out.iou_scores = torch.tensor([[0.2, 0.95, 0.4], [0.7, 0.1, 0.3]])
            return out

    module.Sam3Model = _Model
    module.Sam3Processor = _Processor
    module.Sam3TrackerModel = _Model
    module.Sam3TrackerProcessor = _Processor
    module.pipeline = lambda *a, **k: (
        lambda image, **kw: {"masks": _canned_masks(shape), "scores": np.array([0.9, 0.8])}
    )
    monkeypatch.setitem(sys.modules, "transformers", module)
    # Weight resolution must not touch the network either.
    monkeypatch.setattr("sem_segment.weights.cached_snapshot", lambda *_a, **_k: "/fake/cache")
    return module


def test_sam3_auto_normalises_pipeline_output(fake_transformers):
    config = Config.model_validate({"segmentation": {"backend": "sam3_auto", "device": "cpu"}})
    instances = build_segmenter(config).segment(np.zeros((64, 96, 3), dtype=np.uint8))
    assert len(instances) == 2
    assert all(isinstance(m, InstanceMask) for m in instances)
    assert [round(m.score, 2) for m in instances] == [0.9, 0.8]
    assert all(m.backend == "sam3_auto" and m.prompt == "grid" for m in instances)


def test_sam3_text_normalises_instance_segmentation_output(fake_transformers):
    config = Config.model_validate(
        {"segmentation": {"backend": "sam3_text", "text": "particle", "device": "cpu"}}
    )
    segmenter = build_segmenter(config)
    instances = segmenter.segment(np.zeros((64, 96, 3), dtype=np.uint8))
    assert len(instances) == 2
    assert all(m.prompt == "particle" for m in instances)
    assert segmenter.describe()["native_resolution_px"] == 1008


def test_sam3_prompt_keeps_the_best_of_three_candidate_masks(fake_transformers):
    config = Config.model_validate(
        {"segmentation": {"backend": "sam3_prompt", "points": [[20.0, 15.0]], "device": "cpu"}}
    )
    instances = build_segmenter(config).segment(np.zeros((64, 96, 3), dtype=np.uint8))
    # iou_scores row 0 is [0.2, 0.95, 0.4]: the arg-max, not the first candidate.
    assert instances and instances[0].score == pytest.approx(0.95)


def test_score_threshold_filters_low_confidence_masks(fake_transformers):
    config = Config.model_validate(
        {"segmentation": {"backend": "sam3_auto", "device": "cpu", "score_threshold": 0.85}}
    )
    instances = build_segmenter(config).segment(np.zeros((64, 96, 3), dtype=np.uint8))
    assert len(instances) == 1


def test_missing_transformers_names_the_install_command(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "transformers":
            raise ImportError("No module named 'transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setitem(sys.modules, "transformers", None)
    monkeypatch.delitem(sys.modules, "transformers")
    monkeypatch.setattr(builtins, "__import__", blocked)

    from sem_segment.weights import transformers_available

    ok, detail = transformers_available()
    assert not ok
    assert '.[segment]' in detail


def test_gated_repository_error_explains_the_offline_route(monkeypatch):
    from sem_segment.weights import GatedRepositoryError, resolve_model_source

    monkeypatch.setattr("sem_segment.weights.cached_snapshot", lambda *_a, **_k: None)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(GatedRepositoryError) as error:
        resolve_model_source("facebook/sam3")
    message = str(error.value)
    assert "download-weights" in message and "scp" in message
    assert "classical" in message  # the no-weights way out is always offered


def test_gated_repository_error_explains_the_token_route(monkeypatch):
    from sem_segment.weights import GatedRepositoryError, resolve_model_source

    monkeypatch.setattr("sem_segment.weights.cached_snapshot", lambda *_a, **_k: None)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    with pytest.raises(GatedRepositoryError) as error:
        resolve_model_source("facebook/sam3")
    message = str(error.value)
    assert "huggingface.co/facebook/sam3" in message
    assert "hf auth login" in message  # the documented flow comes first


def test_local_model_path_bypasses_the_hub(tmp_path, monkeypatch):
    from sem_segment.weights import resolve_model_source

    monkeypatch.setattr("sem_segment.weights.cached_snapshot", lambda *_a, **_k: None)
    local = tmp_path / "sam3"
    local.mkdir()
    assert resolve_model_source("facebook/sam3", local) == str(local)


def test_download_skips_the_redundant_original_checkpoint(monkeypatch):
    """facebook/sam3 ships the model twice; transformers reads only one copy.

    model.safetensors is 3.44 GB and sam3.pt is another 3.45 GB in Meta's own
    format, which from_pretrained never loads. Fetching both doubles the
    transfer for no benefit, which matters on a slow link to an HPC host.
    """
    import sem_segment.weights as weights

    captured = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return "/fake/snapshot"

    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = fake_snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)

    weights.fetch_weights("facebook/sam3")
    assert captured["ignore_patterns"] == ["*.pt"]

    captured.clear()
    weights.fetch_weights("facebook/sam3", all_files=True)
    assert "ignore_patterns" not in captured


def test_download_lets_huggingface_hub_resolve_credentials(monkeypatch):
    """Do not re-implement token lookup; the library already does it.

    `hf auth login` - the flow the model card documents - stores a token in a
    file, not the environment. Passing our own env-var-only token would override
    that with None and duplicate logic we get for free.
    """
    import sem_segment.weights as weights

    captured = {}
    module = types.ModuleType("huggingface_hub")
    module.snapshot_download = lambda **kw: (captured.update(kw), "/fake")[1]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setenv("HF_TOKEN", "hf_example")

    weights.fetch_weights("facebook/sam3")
    assert "token" not in captured


def test_signed_in_via_cli_login_is_detected(monkeypatch):
    """A token file with no env var must not read as 'not signed in'."""
    import sem_segment.weights as weights

    module = types.ModuleType("huggingface_hub")
    module.get_token = lambda: "hf_from_login_file"
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    assert weights._token() == "hf_from_login_file"
    assert weights._token_source() == "hf auth login"


def test_env_var_token_is_still_detected_and_labelled(monkeypatch):
    import sem_segment.weights as weights

    module = types.ModuleType("huggingface_hub")
    module.get_token = lambda: "hf_from_env"
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    monkeypatch.setenv("HF_TOKEN", "hf_from_env")

    assert weights._token() == "hf_from_env"
    assert weights._token_source() == "environment"


def test_gated_error_leads_with_the_documented_cli_flow(monkeypatch):
    from sem_segment.weights import GatedRepositoryError, resolve_model_source

    monkeypatch.setattr("sem_segment.weights.cached_snapshot", lambda *_a, **_k: None)
    monkeypatch.setattr("sem_segment.weights._token", lambda: None)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    with pytest.raises(GatedRepositoryError) as error:
        resolve_model_source("facebook/sam3")
    assert "hf auth login" in str(error.value)


def test_missing_model_path_is_reported_clearly(tmp_path):
    from sem_segment.weights import GatedRepositoryError, resolve_model_source

    with pytest.raises(GatedRepositoryError, match="model_path does not exist"):
        resolve_model_source("facebook/sam3", tmp_path / "absent")
