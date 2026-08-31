"""Original DDIM implementation (Song, Meng & Ermon) packaged for this repository.

The upstream research code is preserved here unchanged apart from import
namespacing: ``models``/``runners``/``functions``/``datasets`` became
``ddim.models`` and friends so that every workflow in this repository can be
launched from the repository root without ``sys.path`` manipulation.

Nothing in this package depends on ``runctl``, ``burst_diffusion`` or
``noising_pipeline``; the dependency arrow points the other way.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
