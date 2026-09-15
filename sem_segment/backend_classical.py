"""A segmentation backend that downloads nothing.

This exists for three reasons, none of them "fallback":

1. It is the continuous-integration path.  The whole numerical core - contours,
   sub-pixel refinement, metrology - can be validated end to end without a GPU,
   a Hugging Face account, or a 0.9 B-parameter download.
2. It is the only backend that works on an air-gapped machine.
3. It is the control arm.  Running it alongside SAM 3 on the same image is what
   shows whether the learned model actually bought anything over a threshold,
   which is otherwise easy to assume and hard to know.

The algorithm is deliberately plain: a global or local threshold, connected
components, and an optional distance-transform watershed to split features that
touch.  It has no notion of what a feature *is*, which is exactly the capability
SAM 3 adds.
"""

from __future__ import annotations

import numpy as np

from .backends import InstanceMask, register_backend
from .config import SegmentationConfig


class ClassicalSegmenter:
    """Threshold, label, and optionally split touching regions."""

    name = "classical"

    def __init__(self, config: SegmentationConfig) -> None:
        self.config = config

    def segment(self, model_rgb: np.ndarray) -> list[InstanceMask]:
        from scipy import ndimage
        from skimage.filters import threshold_otsu
        from skimage.segmentation import watershed

        gray = np.asarray(model_rgb, dtype=np.float64)
        if gray.ndim == 3:
            gray = gray[..., 0]
        gray = gray / 255.0 if gray.max() > 1.0 else gray

        if float(np.ptp(gray)) < 1e-9:
            return []
        threshold = float(threshold_otsu(gray))

        bright = gray > threshold
        if self.config.polarity == "bright":
            foreground, polarity = bright, "bright"
        elif self.config.polarity == "dark":
            foreground, polarity = ~bright, "dark"
        else:
            foreground, polarity = self._choose_polarity(bright, ndimage)

        labels, count = ndimage.label(foreground)
        if count and self.config.split_touching:
            labels = self._split_touching(foreground, labels, watershed, ndimage)
            count = int(labels.max())

        instances: list[InstanceMask] = []
        # find_objects gives the bounding slice of every label in one pass, which
        # avoids building a full-frame boolean array per instance.
        for index, window in enumerate(ndimage.find_objects(labels), start=1):
            if window is None:
                continue
            crop = labels[window] == index
            instance = InstanceMask(
                bbox=(window[0].start, window[0].stop, window[1].start, window[1].stop),
                crop=np.ascontiguousarray(crop),
                # A threshold has no confidence of its own. Reporting a constant
                # is honest; inventing a score from area or contrast would let a
                # meaningless number drive the score-ordered deduplication.
                score=1.0,
                backend=self.name,
                prompt=f"otsu:{polarity}",
                meta={"threshold": threshold, "polarity": polarity},
            )
            instances.append(instance)
            if len(instances) >= self.config.max_instances:
                break
        return instances

    @staticmethod
    def _choose_polarity(bright: np.ndarray, ndimage) -> tuple[np.ndarray, str]:
        """Pick the side of the threshold that behaves like features, not substrate.

        The obvious heuristic - "features are whichever side covers less of the
        frame" - inverts as soon as features cover more than half the image, and
        then segments the gaps between them instead. Real micrographs do that
        routinely: a field of packed spheres is mostly sphere.

        The more durable distinction is topological. Features are many compact
        regions, often wholly inside the frame; substrate is one large region
        running off every edge. So each side is scored by the fraction of its
        area sitting in components that touch the border, and the side with less
        of that is the foreground. This gets bright spheres on a dark substrate
        and dark holes in a bright matrix both right without assuming either.
        """
        def border_fraction(mask: np.ndarray) -> float:
            total = int(np.count_nonzero(mask))
            if total == 0:
                return 1.0
            labels, count = ndimage.label(mask)
            if count == 0:
                return 1.0
            edge_ids = set(np.unique(np.concatenate([
                labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]
            ])).tolist()) - {0}
            if not edge_ids:
                return 0.0
            touching = int(np.count_nonzero(np.isin(labels, list(edge_ids))))
            return touching / total

        dark = ~bright
        if border_fraction(bright) <= border_fraction(dark):
            return bright, "bright"
        return dark, "dark"

    @staticmethod
    def _split_touching(foreground, labels, watershed, ndimage):
        """Separate merged blobs at their necks via a distance-transform watershed."""
        from skimage.feature import peak_local_max

        distance = ndimage.distance_transform_edt(foreground)
        if not np.any(distance > 0):
            return labels
        # A footprint tied to the typical object scale keeps one seed per blob;
        # too small a footprint shatters a single feature into fragments.
        min_distance = max(2, int(round(float(distance.max()) / 2.0)))
        coordinates = peak_local_max(
            distance, min_distance=min_distance, labels=foreground, exclude_border=False
        )
        if coordinates.shape[0] <= 1:
            return labels
        markers = np.zeros(distance.shape, dtype=np.int32)
        markers[tuple(coordinates.T)] = np.arange(1, coordinates.shape[0] + 1)
        markers, _ = ndimage.label(markers > 0)
        return watershed(-distance, markers, mask=foreground)

    def describe(self) -> dict:
        return {
            "backend": self.name,
            "algorithm": "otsu-threshold + connected-components",
            "split_touching": self.config.split_touching,
            "polarity": self.config.polarity,
            "model_id": None,
            "native_resolution_px": None,
        }


register_backend("classical", ClassicalSegmenter)
