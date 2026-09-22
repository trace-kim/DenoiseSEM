"""CPU-only numerical stand-in for tests of optional CuPy execution paths."""

from types import ModuleType, SimpleNamespace
import sys
import time

import pytest


@pytest.fixture
def fake_cupy(monkeypatch):
    import numpy as np
    from scipy import ndimage

    state = SimpleNamespace(devices=[], uploads=[], downloads=[], filters=[], labels=[], synchronizations=0)

    class CuPy(ModuleType):
        def __getattr__(self, name):
            return getattr(np, name)

    class Device:
        def __init__(self, index):
            self.index = index
        def __enter__(self):
            state.devices.append(self.index)
        def __exit__(self, *args):
            pass

    class Event:
        def record(self):
            self.when = time.perf_counter()
        def synchronize(self):
            state.synchronizations += 1

    def upload(value, *args, **kwargs):
        array = np.asarray(value, *args, **kwargs)
        state.uploads.append((array.shape, array.dtype))
        return array.copy()

    def download(value):
        state.downloads.append((value.shape, value.dtype))
        return np.asarray(value).copy()

    def gaussian_filter(values, sigma, **kwargs):
        state.filters.append((values.shape, sigma))
        return ndimage.gaussian_filter(values, sigma, **kwargs)

    def label(values, **kwargs):
        state.labels.append(values.shape)
        return ndimage.label(values, **kwargs)

    cp = CuPy("cupy")
    cp.cuda = SimpleNamespace(Device=Device, Event=Event,
                              get_elapsed_time=lambda a, b: (b.when - a.when) * 1000)
    cp.asarray, cp.asnumpy = upload, download
    cuda_ndimage = ModuleType("cupyx.scipy.ndimage")
    cuda_ndimage.gaussian_filter, cuda_ndimage.label = gaussian_filter, label
    scipy_module = ModuleType("cupyx.scipy")
    scipy_module.ndimage = cuda_ndimage
    cupyx_module = ModuleType("cupyx")
    cupyx_module.scipy = scipy_module
    for name, module in (("cupy", cp), ("cupyx", cupyx_module), ("cupyx.scipy", scipy_module),
                         ("cupyx.scipy.ndimage", cuda_ndimage)):
        monkeypatch.setitem(sys.modules, name, module)
    state.cp = cp
    return state
