"""SAM 3 adapters.

SAM 3 exposes three quite different doors into one checkpoint, and they are not
interchangeable on SEM imagery:

``sam3_auto`` (the default)
    ``pipeline("mask-generation")`` - a grid of point prompts followed by
    deduplication.  It needs no semantic understanding of the image at all,
    which is exactly why it is the default here: SAM 3's concept head was
    trained on natural-image vocabulary, and micrographs are far outside that
    distribution.

``sam3_text``
    ``Sam3Model`` with a noun phrase, returning every instance of the concept in
    one pass.  Much better ergonomics when it works, and worth trying, but
    whether "hole" or "particle" means anything on a given micrograph is an
    empirical question, not an assumption.

``sam3_prompt``
    ``Sam3TrackerModel``, the SAM 2-style geometric API: one object per point or
    box, three candidate masks ranked by predicted IoU.  Purely geometric, so it
    degrades gracefully, but it needs the operator to say where to look.

Each returns a different output shape; this module's job is to normalise all
three into ``list[InstanceMask]`` so nothing downstream has to care.

One checkpoint serves all three. Note that ``facebook/sam3`` declares
``model_type: "sam3_video"`` and ``architectures: ["Sam3VideoModel"]`` - not
``sam3``/``Sam3Model`` as the documentation page might suggest - so
``AutoModelForMaskGeneration`` resolves it to ``Sam3TrackerModel``. Verified by
loading the real weights: ``sam3_auto`` gets a ``Sam3TrackerModel``.

Every import here is deferred into the call that needs it, so ``--help``, config
validation and the entire classical path never load torch or transformers.
"""

from __future__ import annotations

import numpy as np

from .backends import BackendUnavailable, InstanceMask, register_backend
from .config import SegmentationConfig
from .weights import GatedRepositoryError, resolve_model_source, transformers_available


def _resolve_device(requested: str) -> str:
    import torch

    if requested and requested != "auto":
        return requested
    return "cuda" if torch.cuda.is_available() else "cpu"


def _require_transformers() -> None:
    available, detail = transformers_available()
    if not available:
        raise BackendUnavailable(detail)


def _source(config: SegmentationConfig) -> str:
    try:
        return resolve_model_source(config.model_id, config.model_path)
    except GatedRepositoryError as error:
        raise BackendUnavailable(str(error)) from error


def _to_pil(model_rgb: np.ndarray):
    from PIL import Image

    return Image.fromarray(np.asarray(model_rgb, dtype=np.uint8), mode="RGB")


def _instances_from_masks(
    masks, scores, *, backend: str, prompt: str | None, threshold: float, limit: int
) -> list[InstanceMask]:
    """Normalise any of SAM 3's mask outputs into the package's own type."""
    instances: list[InstanceMask] = []
    for mask, score in zip(masks, scores):
        value = float(score)
        if value < threshold:
            continue
        array = np.asarray(mask)
        if array.ndim > 2:
            array = array.squeeze()
        if array.ndim != 2:
            continue
        instance = InstanceMask.from_mask(
            array.astype(bool), score=value, backend=backend, prompt=prompt
        )
        if instance is not None:
            instances.append(instance)
    instances.sort(key=lambda m: -m.score)
    return instances[:limit]


class Sam3AutoSegmenter:
    """Automatic mask generation from a grid of point prompts."""

    name = "sam3_auto"

    def __init__(self, config: SegmentationConfig) -> None:
        self.config = config
        self._generator = None

    def _load(self):
        if self._generator is None:
            _require_transformers()
            from transformers import pipeline

            import torch

            device = _resolve_device(self.config.device)
            self._generator = pipeline(
                "mask-generation",
                model=_source(self.config),
                device=0 if device.startswith("cuda") else -1,
                # transformers 5 renamed torch_dtype -> dtype on pipeline().
                dtype=torch.float32,
            )
        return self._generator

    def segment(self, model_rgb: np.ndarray) -> list[InstanceMask]:
        generator = self._load()
        outputs = generator(
            _to_pil(model_rgb),
            points_per_batch=self.config.points_per_batch,
            points_per_crop=self.config.points_per_crop,
        )
        masks = outputs.get("masks", [])
        scores = outputs.get("scores", [1.0] * len(masks))
        scores = [float(s) for s in np.asarray(scores).reshape(-1)] if len(masks) else []
        return _instances_from_masks(
            masks, scores, backend=self.name, prompt="grid",
            threshold=self.config.score_threshold, limit=self.config.max_instances,
        )

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "model_id": self.config.model_id,
            "mode": "automatic mask generation (point grid)",
            "points_per_crop": self.config.points_per_crop,
            "native_resolution_px": 1008,
        }


class Sam3TextSegmenter:
    """Promptable concept segmentation: every instance matching a phrase."""

    name = "sam3_text"

    def __init__(self, config: SegmentationConfig) -> None:
        self.config = config
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            _require_transformers()
            from transformers import Sam3Model, Sam3Processor

            source = _source(self.config)
            device = _resolve_device(self.config.device)
            self._model = Sam3Model.from_pretrained(source).to(device).eval()
            self._processor = Sam3Processor.from_pretrained(source)
        return self._model, self._processor

    def segment(self, model_rgb: np.ndarray) -> list[InstanceMask]:
        import torch

        model, processor = self._load()
        kwargs: dict = {"images": _to_pil(model_rgb), "text": self.config.text}
        if self.config.boxes:
            kwargs["input_boxes"] = [[list(b) for b in self.config.boxes]]
            kwargs["input_boxes_labels"] = [[1] * len(self.config.boxes)]
        inputs = processor(**kwargs, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs)
        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=self.config.score_threshold,
            mask_threshold=self.config.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]
        masks = [np.asarray(m) for m in results["masks"]]
        scores = [float(s) for s in np.asarray(results["scores"]).reshape(-1)]
        return _instances_from_masks(
            masks, scores, backend=self.name, prompt=self.config.text,
            threshold=self.config.score_threshold, limit=self.config.max_instances,
        )

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "model_id": self.config.model_id,
            "mode": "promptable concept segmentation",
            "text": self.config.text,
            "native_resolution_px": 1008,
        }


class Sam3PromptSegmenter:
    """Geometric prompting: one object per point or box."""

    name = "sam3_prompt"

    def __init__(self, config: SegmentationConfig) -> None:
        self.config = config
        self._model = None
        self._processor = None

    def _load(self):
        if self._model is None:
            _require_transformers()
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor

            source = _source(self.config)
            device = _resolve_device(self.config.device)
            self._model = Sam3TrackerModel.from_pretrained(source).to(device).eval()
            self._processor = Sam3TrackerProcessor.from_pretrained(source)
        return self._model, self._processor

    def segment(self, model_rgb: np.ndarray) -> list[InstanceMask]:
        import torch

        model, processor = self._load()
        image = _to_pil(model_rgb)
        kwargs: dict = {"images": image}
        prompt: str
        if self.config.points:
            # Nesting is (image, object, point, xy): one object per point, so
            # each prompt yields its own mask rather than one merged blob.
            kwargs["input_points"] = [[[list(p)] for p in self.config.points]]
            kwargs["input_labels"] = [[[1] for _ in self.config.points]]
            prompt = f"points:{len(self.config.points)}"
        else:
            kwargs["input_boxes"] = [[list(b) for b in self.config.boxes or []]]
            prompt = f"boxes:{len(self.config.boxes or [])}"

        inputs = processor(**kwargs, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, multimask_output=True)
        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
        iou = outputs.iou_scores.cpu().numpy().reshape(masks.shape[0], -1)

        instances: list[InstanceMask] = []
        for index in range(masks.shape[0]):
            # Three candidates per prompt, ranked by predicted IoU; keep the best.
            best = int(np.argmax(iou[index]))
            array = np.asarray(masks[index][best]).astype(bool)
            instance = InstanceMask.from_mask(
                array, score=float(iou[index][best]), backend=self.name, prompt=prompt
            )
            if instance is not None and instance.score >= self.config.score_threshold:
                instances.append(instance)
        instances.sort(key=lambda m: -m.score)
        return instances[: self.config.max_instances]

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "model_id": self.config.model_id,
            "mode": "promptable visual segmentation",
            "n_points": len(self.config.points or []),
            "n_boxes": len(self.config.boxes or []),
            "native_resolution_px": 1008,
        }


register_backend("sam3_auto", Sam3AutoSegmenter)
register_backend("sam3_text", Sam3TextSegmenter)
register_backend("sam3_prompt", Sam3PromptSegmenter)
