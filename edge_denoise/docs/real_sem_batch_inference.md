# Batch inference and noise analysis on a real noisy-only folder

You have trained checkpoints and a folder of real noisy SEM frames with no
matched clean images and no `manifest.json`. This page is the runbook for
denoising every frame in that folder and characterizing the noise before and
after. Commands are bash, for the RHEL training server; run them from the
repository root inside the container.

## Verdict first

**No data preparation is needed.** `prepare-real` exists to build *training*
data — registered repeats, locked site splits, fixed detector levels. Inference
needs none of it. The checkpoint stores `data.black_level` and
`data.white_level`, so `denoise --input` applies the exact normalization the
model was trained with, straight from one raw file. See
[real_sem_training.md](real_sem_training.md) §9.

Run `prepare-real` on this folder only if you later want reference-based CD and
sigma numbers from `evaluate-real`, which needs repeats of the same site in
per-site subfolders.

Three steps, all below:

```text
data/SEM-test/*.jpg          raw noisy frames
      |
      |  (2) python -m edge_denoise denoise, once per file
      v
output/test-infer/           *_input.png  *_denoised.png  *_denoised.tif
      |
      |  (3) copy only the float32 TIFFs into one site folder
      v
output/test-noise-in/site_01/
      |
      |  (4) python -m sem_noise analyze   (also run it on the raw folder)
      v
output/noise-denoised/index.html
```

## Layout

Substitute your own paths; these placeholders are used throughout.

```text
data/SEM-test/                                 <- your folder of real noisy frames
  frame_000.jpg
  frame_001.jpg
  ...
runs/edge_denoise/<run_name>/ckpt_latest.pt    <- the checkpoint you pick
```

Supported inputs are grayscale uint8 or uint16 PNG, TIFF, BMP, and 8-bit JPEG,
one frame per file. RGB is accepted only when all three decoded channels match
exactly.

## 1. Check one frame first

Confirm the checkpoint loads and the output looks sane before spending an hour
on the whole folder.

```bash
CKPT="runs/edge_denoise/<run_name>/ckpt_latest.pt"

CUDA_VISIBLE_DEVICES=0 python -m edge_denoise denoise \
  --checkpoint "$CKPT" \
  --input data/SEM-test/frame_000.jpg \
  --out output/test-infer-smoke \
  --tile-batch 4 --device cuda
```

It prints the frame size, tile, stride, and margin it used. Inspect the
quantitative output:

```bash
python -c "from PIL import Image; import numpy as np; a=np.asarray(Image.open('output/test-infer-smoke/frame_000_denoised.tif')); print(a.shape, a.dtype, a.min(), a.max())"
```

## 2. Denoise every image in the folder

`denoise` takes one measurement per invocation, so the batch is a shell loop.
Each file's exit code is captured and the loop continues past a failure, with
the failures listed at the end — no silent skips.

```bash
CKPT="runs/edge_denoise/<run_name>/ckpt_latest.pt"
SRC="data/SEM-test"
OUT="output/test-infer"
mkdir -p "$OUT"
: > "$OUT/failed.txt"

find "$SRC" -type f \( -iname '*.png' -o -iname '*.tif' -o -iname '*.tiff' \
     -o -iname '*.bmp' -o -iname '*.jpg' -o -iname '*.jpeg' \) -print0 \
| sort -z \
| while IFS= read -r -d '' f; do
    CUDA_VISIBLE_DEVICES=0 python -m edge_denoise denoise \
      --checkpoint "$CKPT" --input "$f" --out "$OUT" \
      --tile-batch 4 --device cuda
    rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "$rc  $f" >> "$OUT/failed.txt"
    fi
  done

wc -l < "$OUT/failed.txt"   # 0 means every frame succeeded
cat "$OUT/failed.txt"
```

To detach a long run, put the block in a script and keep its log:

```bash
nohup bash denoise_folder.sh > output/test-infer/infer.log 2>&1 &
```

Each input produces three files in `$OUT`:

```text
<stem>_input.png       8-bit preview of the input
<stem>_denoised.png    8-bit preview of the output
<stem>_denoised.tif    float32 in [0, 1] -- the only quantitative output
```

Convert the TIFF back to detector units with
`value * (white_level - black_level) + black_level` if your metrology software
expects counts. Keep the float result for analysis; the PNGs are previews only.

### Notes and limits

- **Tiling defaults are already correct.** Stride is half the tile (256 for a
  512-px model) and `--margin` defaults to 3 for real-data checkpoints — the
  loss border training never supervised. The frame is never padded or resized.
  Only lower `--stride`; a larger one just reduces how many tiles vote per
  pixel.
- **`--tile-batch` is memory only.** Reduce it on OOM; the result is identical.
- **Filename collisions.** Outputs are keyed on the file stem alone, so two
  files named `frame_000.jpg` in different subfolders overwrite each other.
  Flatten or rename first if your folder is nested.
- **Startup cost dominates.** Every iteration reloads the checkpoint and
  re-initializes CUDA, several seconds each. On a 128-frame set that is minutes
  of pure overhead. There is no `--input-dir` on `denoise` today; adding one is
  a small change to `edge_denoise/cli.py` if the overhead becomes a problem.
- **`--center-crop`** processes a single training-resolution center tile instead
  of the full frame. Use it to test tiling, not for deliverables.

## 3. Collect the quantitative outputs

Required, not cosmetic. `sem_noise` reads *every* supported image in a
directory, so pointing it at `$OUT` would pool inputs, previews, and denoised
TIFFs into one meaningless "site".

```bash
SITE="output/test-noise-in/site_01"
mkdir -p "$SITE"
cp "$OUT"/*_denoised.tif "$SITE"/
ls "$SITE" | head
```

The constant `_denoised` suffix preserves natural filename order, so
`frame_000_denoised.tif` still sorts before `frame_010_denoised.tif`.

## 4. Noise analysis, before and after

`sem_noise` characterizes repeated measurements of the **same site**. It needs
at least 8 frames per site by default (`min_frames`, see
`sem_noise/configs/default.yml`) and uses no GPU.

```bash
python -m pip install -e ".[analysis]"

SRC="data/SEM-test"

# denoised
python -m sem_noise analyze \
  --input output/test-noise-in --output output/noise-denoised

# raw, for the before/after comparison
python -m sem_noise analyze \
  --input "$SRC" --output output/noise-raw
```

Open `output/noise-denoised/index.html`. Reports are offline and self-contained.

No acquisition metadata is required. Nothing above needs a pixel size, a frame
interval, or anything produced by `prepare-real` — those belong to other
commands and are not part of this procedure.

- If filenames do not encode acquisition order, build an inventory and edit the
  site labels and frame indices to match the acquisition record:

  ```bash
  python -m sem_noise inventory --input "$SRC" --output tmp/order.csv
  python -m sem_noise analyze --input "$SRC" --manifest tmp/order.csv \
    --output output/noise-raw
  ```

- Exit code 1 means some sites failed and the index identifies them; exit code 2
  is a command, input, or configuration error.

### Optional: physical units

Both of the following are optional and default to unset. Every result above is
produced without them; supplying them only relabels axes and adds columns.

```bash
python -m sem_noise analyze \
  --input output/test-noise-in --output output/noise-denoised-units \
  --pixel-size-nm 1.5 --frame-interval-s 2.4
```

- `--pixel-size-nm` adds `drift_dy_nm` and `drift_dx_nm` columns to the drift
  table. Every other distance stays in native pixels either way.
- `--frame-interval-s` is the time between acquisitions, **not** pixel dwell
  time, and it is used only when the manifest carries no `timestamp_s` column.
  Without it the time axis is the frame index: Allan curves are in frames rather
  than seconds and FFT frequency is cycles per frame. Noise, registration, and
  stability results are unchanged.

Full option reference: [sem_noise/README.md](../../sem_noise/README.md).

### Reading the denoised report

Two families of panels are meaningful; two are not.

**Trust these.** Per-pixel temporal sigma and the adjacent-difference estimator
on the denoised stack measure the residual noise the estimator transmitted from
its input. That is the repeatability quantity this project optimizes, and it is
directly comparable to the same panel in `output/noise-raw`.

**Do not read these as detector physics.** The mean-variance affine fit and the
1/sqrt(N) Allan reference assume spatially independent repeats. A denoiser
correlates its output noise across pixels, so on the denoised stack those panels
describe the estimator, not the detector. Read them only as a difference against
the raw run.

Registration also changes meaning: the denoised frames carry the same stage
drift as their inputs, and `sem_noise` re-estimates translations from the
denoised images, which are easier to register than raw ones. Compare drift
curves between the two runs rather than treating either as truth.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no such file or directory ... ckpt_latest.pt` | The training stage that produces this checkpoint never finished | Check that run's own log; do not assume a chained stage ran |
| `input must be at least NxN` | `--center-crop` on a frame smaller than the training crop | Drop `--center-crop`; full-frame tiling handles any size >= the tile |
| CUDA out of memory | Too many tiles per forward pass | Lower `--tile-batch`; the result does not change |
| `requires at least 8 frames, found N` | Fewer repeats than `min_frames` | Supply more repeats, or lower `min_frames` in a config passed with `--config` |
| Noise report shows one site with 3x the expected frames | `sem_noise` was pointed at the raw `denoise` output directory | Run step 3 first; analyze only the `_denoised.tif` copies |
| Denoised output looks tiled or seamed | Non-default `--stride` or `--margin` | Return to the defaults; keep stride at half the tile or less |
