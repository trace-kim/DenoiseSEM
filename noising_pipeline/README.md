# Portable noisy microscopy dataset pipeline

`noising_pipeline` creates deterministic pairs of clean and synthetically noisy
images. It is deliberately independent of DDIM's datasets, models, training
configuration, and command-line code.

## Requirements

- Python 3.10 or newer
- NumPy
- Pillow
- Requests (only used by the built-in BBBC038 downloader)

Import the single public function with:

```python
from noising_pipeline import create_noisy_dataset
```

## Local images

The local source is scanned recursively for single-frame PNG, JPEG, BMP, and
TIFF files. Inputs must be 8-bit grayscale, 8-bit RGB, opaque 8-bit RGBA, or
16-bit grayscale. Opaque alpha channels are discarded without changing the RGB
pixels. Dimensions and bit depth are retained; images are never resized.

```python
manifest = create_noisy_dataset(
    source_dir="microscopy/clean",
    output_dir="microscopy/paired",
    n=3,
    steps=2,
    noise_type="gaussian",
    seed=123,
    source_license="CC BY 4.0",
)
```

`n` is the number of noisy replicas made for every source image. Every replica
starts from the unchanged clean image. `steps` controls accumulated corruption
strength. The default fused mode applies that strength with one vectorized draw
per selected distribution, so runtime does not grow linearly with `steps`.

## Download BBBC038v1

The built-in [BBBC038v1](https://bbbc.broadinstitute.org/BBBC038) stage-one
training source contains 670 diverse microscopy images from the 2018 Data
Science Bowl. It spans fluorescence and histology stains, multiple organisms,
magnifications, and illumination conditions. The source PNGs have useful 8-bit
contrast in ordinary image viewers and are published as CC0 by the Broad
Bioimage Benchmark Collection.

```python
manifest = create_noisy_dataset(
    source_dir="data/BBBC038v1",
    output_dir="data/BBBC038v1-noisy",
    n=100,
    steps=1,
    noise_type="poisson",
    download="bbbc038",
)
```

The archive is cached in `source_dir`, verified against SHA-256
`dcb6edc2690f137406638b2309581a71522c4dff19157d118453b448dcddcb68`, and
only the 670 `<ImageId>/images/*.png` source images are extracted; segmentation
masks and archive metadata are ignored. A complete existing cache is reused.
With `n=100`, the result contains 670 clean targets and 67,000 noisy images.

## Noise combinations and parameters

Names are case-insensitive. A sequence applies the fused distributions in
exactly the given order:

```python
manifest = create_noisy_dataset(
    "clean",
    "paired",
    n=5,
    steps=3,
    noise_type=["Poisson", "gaussian", "salt_pepper"],
    noise_params={
        "poisson": {"peak": 5000},
        "gaussian": {"mean": 0.0, "std": 0.02},
        "salt_pepper": {"amount": 0.005, "salt_ratio": 0.4},
    },
)
```

Defaults, applied per step, are:

| Distribution | Parameters |
| --- | --- |
| `gaussian` | `mean=0.0`, `std=0.01` |
| `poisson` | `peak=10000` |
| `salt_pepper` | `amount=0.001`, `salt_ratio=0.5` |
| `ddim` | `beta_schedule="linear"`, `beta_start=0.0001`, `beta_end=0.02`, `num_diffusion_timesteps=1000` |

### DDIM forward process (`ddim`)

The three distributions above are observation models: the signal is never
attenuated, so averaging replicas converges to the clean image. `ddim` is
different in kind. It reproduces the DDPM/DDIM forward process `q(x_t | x_0)`
used to train diffusion models (Ho et al. 2020; Song, Meng & Ermon 2021):

```text
x0  = 2 * observation - 1                                  # DDIM's "rescaled" transform
x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps,   eps ~ N(0, I)
alpha_bar_t = prod_{i = 1..t} (1 - beta_i)
```

- `steps` is the timestep `t`: the number of betas applied, with
  `1 <= t <= num_diffusion_timesteps`. `steps=k` corresponds to index `k - 1`
  of DDIM's 0-based timestep tensor.
- `beta_schedule` (`linear`, `quad`, `const`, `jsd`, `sigmoid`), `beta_start`,
  `beta_end` and `num_diffusion_timesteps` follow DDIM's `get_beta_schedule`
  term for term. The defaults are Ho et al.'s; this repository's SEM DDIM
  configs use `beta_start=0.001`, `beta_end=0.2`, `num_diffusion_timesteps=100`.
- Fused mode draws the marginal once. Iterative mode walks the Markov kernel
  `x_i = sqrt(1 - beta_i) * x_{i-1} + sqrt(beta_i) * eps_i` for `i = 1..t`,
  which composes to the same marginal. Neither mode clips.
- The output is a float32 `.npy` file, not a PNG: `x_t` is unbounded (at
  `t = T` it is close to `N(0, I)`), and clipping or 8-bit rounding would change
  the distribution. The clean target is still a lossless PNG.
- `ddim` must be the last entry of `noise_type` and may appear once. In a
  sequence such as `["poisson", "ddim"]` the observation types run first with
  their usual fusion over `steps`, and `ddim` is applied once to the finished,
  clipped observation with `t = steps`.
- The manifest row records `effective_noise_params.ddim` with `t` and
  `alpha_bar`, so the exact noise realisation is recoverable:
  `eps = (x_t - sqrt(alpha_bar) * x0) / sqrt(1 - alpha_bar)` with
  `x0 = 2 * clean / scale - 1`.

The match with a PyTorch DDIM run is algebraic and distributional, not bitwise:
NumPy's generator draws the normals. `ddim` is not an observation model, so
`burst_diffusion generate` rejects it.

### Fast step fusion

The default `step_mode="fused"` removes the Python loop over `steps`. Per-step
parameters are combined before processing:

| Distribution | Effective parameter for `s` steps |
| --- | --- |
| Gaussian | `mean * s`, `std * sqrt(s)` |
| Poisson | `peak / s` |
| Salt-and-pepper | `amount = 1 - (1 - amount) ** s` |
| DDIM | `alpha_bar_t = prod_{i = 1..t} (1 - beta_i)` with `t = s`; iterative mode walks the `t` kernels instead |

These preserve accumulated Gaussian mean/variance, accumulated Poisson
mean/variance, and cumulative salt-and-pepper replacement probability. Because
clipping happens once per selected distribution, fused results are not
bit-for-bit identical to repeatedly clipped results near intensity boundaries.
For ordered combinations, each fused distribution is applied once in the given
order rather than interleaving complete cycles.

The original behavior remains available when exact legacy reproduction matters:

```python
create_noisy_dataset(..., steps=50, step_mode="iterative")
```

Iterative mode performs every old step and therefore scales linearly with the
step count.

### Progress output

Progress is printed to standard error immediately, after the first output, at
five-second intervals, on failure, and on completion. Each update includes the
completed image count, source and replica indices, rate, elapsed time, and ETA:

```text
[noising_pipeline] 71/67000 noisy images (0.1%); source 1/670, replica 71/100; 3.42 images/s; elapsed 00:21; ETA 5:26:03
```

Use `progress_interval=2.0` for more frequent updates or `progress=False` to
silence them.

## Output and manifest

If the output directory already exists, the function asks before replacing it:

```text
output_dir already exists: .../paired
Overwrite it and delete all existing contents? [y/N]:
```

Only `y` or `yes` confirms; every other response preserves the existing output
and raises `FileExistsError`. For non-interactive use, pass `overwrite=True` to
replace without prompting or `overwrite=False` to reject without prompting.
After confirmation, the old directory is removed and the new dataset is written
directly to the exact requested path. If processing is interrupted, completed
clean/noisy images and line-buffered manifest rows remain there; the pipeline
does not delete partial output. The output still cannot overlap the source
directory.

Only clean targets and final noisy states are saved:

```text
paired/
  clean/00000.png
  noisy/00000_00000.png    # or noisy/00000_00000.npy when noise_type ends with "ddim"
  manifest.jsonl
```

There is one JSON object per noisy image. Each row has `clean_path`,
`noisy_path`, `source_path`, `source_index`, `replica_index`, `shape`,
`bit_depth`, ordered `noise_types`, per-step `noise_params`,
`effective_noise_params`, `step_mode`, `steps`, the derived `sample_seed`,
`dataset`, `license`, and `noisy_encoding`. The last says how the stored noisy
file maps back to the normalized signal: `dtype`, `scale` (divide stored values
by it), `signal_range` (`[0, 1]` for PNGs, `[-1, 1]` for `ddim` latents) and
`clipped` (`true` for PNGs, `false` for latents). Paths in the manifest are
relative and use forward slashes. Local data is identified as `dataset="local"`; its license is the
supplied `source_license` or JSON `null`. Downloaded data is identified as
`dataset="BBBC038v1-stage1-train"` with `license="CC0"`.

## Moving the package

Copy the entire `noising_pipeline/` directory into another repository and make
sure NumPy, Pillow, and Requests are installed. No other directory from this
repository is imported or required. The root `pyproject.toml` only registers the
package for editable installation in this repository.
