# Run a complete real SEM experiment

Use [tools/real_sem_experiment.py](../../tools/real_sem_experiment.py) to denoise
all frames, prepare PNGs in the original intensity units, and analyze both
raw and denoised images. It supports ordinary N2N, registration-aligned N2N,
`ft_avgfull_consist`, and other `edge_denoise` checkpoints.

## 1. Edit the settings

Open the script and edit `SETTINGS` near the top. Only `checkpoint` is required:

```python
SETTINGS = {
    "checkpoint": "runs/edge_denoise/my_run/ckpt_latest.pt",
    "source_dir": "",
    "output_dir": "",
    "png_dir": "",
    "raw_report_dir": "",
    "png_report_dir": "",
    "analysis_config": "",
    "device": "auto",
    "tile_batch": 4,
    "ema": True,
}
```

Blank paths use these defaults. `<run_name>` is the checkpoint's immediate
parent folder (`my_run` in the example).

| Setting | Default |
|---|---|
| `source_dir` | `data/SEM-test/<run_name>` |
| `output_dir` | `output/<run_name>` |
| `png_dir` | `<output_dir>/png` |
| `raw_report_dir` | `<output_dir>/noise-raw` |
| `png_report_dir` | `<output_dir>/noise-png` |

The source default is a naming convention: the checkpoint cannot tell us where
your test images live. Put the images there or set `source_dir` to their existing
folder. Relative paths resolve from the repository root; absolute paths work too.

Use **one flat folder of repeated uint8 acquisitions of the same site**, at least
8 frames by default. PNG, TIFF, BMP, and JPEG inputs are supported. Frames must
have matching dimensions and unique filename stems; filenames must encode
acquisition order. No `prepare-real` step is needed.

## 2. Run the script

From the repository root in your server environment/container:

```bash
python -m pip install -e ".[analysis]"
python tools/real_sem_experiment.py
```

You can also override settings on the command line without editing the file:

```bash
CUDA_VISIBLE_DEVICES=0 python tools/real_sem_experiment.py \
  --checkpoint runs/edge_denoise/my_run/ckpt_latest.pt \
  --source-dir data/SEM-test/site_01 \
  --output-dir output/my_run_site_01 \
  --device cuda --tile-batch 4
```

The script loads the model once, runs full-frame inference with default tiling,
and stops on errors. Lower `tile_batch` if GPU memory is insufficient.
Use `analysis_config` (or `--analysis-config`) for a `sem_noise` YAML config,
including frame interval, pixel size, ROI, or minimum frame count.

**Use a new output directory for each experiment**, including different
checkpoints from the same run or reruns after failure. Existing output folders
are rejected to prevent results from mixing. Blank PNG/report paths follow your
chosen `output_dir`; explicit overrides stay at the paths you set.

## 3. Open the reports

The default experiment folder contains:

```text
output/<run_name>/
  experiment.json       Resolved settings, input filenames, levels, and status
  inference/            Input/output preview PNGs and normalized float32 TIFFs
  raw/                  Lossless PNG copies of decoded raw pixels
  png/                  Denoised uint8 PNGs in original intensity units
  noise-raw/index.html  Raw-image analysis
  noise-png/index.html  Denoised-PNG analysis
```

Compare brightness over time, temporal sigma, adjacent-frame differences, and
drift between the two reports. Registration is estimated separately, so check
accepted frames and regions before comparing numbers. Do not point analysis at
`inference/`: that folder mixes inputs, previews, and TIFFs.

## Intensity and precision

The script uses the checkpoint's **fixed** black/white levels. It never stretches
each image by its observed minimum and maximum. With levels 0 and 255 it copies
the existing denoised PNGs; otherwise it converts the TIFFs using:

```text
original_units = normalized * (white_level - black_level) + black_level
PNG = round(clip(original_units, 0, 255))
```

This preserves the original intensity scale without consulting the original
image's brightness. It cannot recover clipped values or undo changes introduced
by the model. Compare frame brightness to see whether the denoiser altered your
acquisition's decreasing intensity trend.

All brightness, noise, registration, contour and metrology measurements must
decode the final saved uint8 PNGs, including their rounding/clipping. Normalized
TIFFs are diagnostic artifacts and must not supply measurements. Lower noise
alone does not prove that edges or absolute intensity are accurate.

For the current phase use [the multi-model comparison](real_sem_comparison.md)
and [the five-model training suite](real_sem_next_phase.md), which preserve the
saved-uint8 contract and the agreed detector throughout the workflow.

See the [sem_noise guide](../../sem_noise/README.md) for analysis options and
report interpretation.
