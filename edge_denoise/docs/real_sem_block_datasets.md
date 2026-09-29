# Block-averaged datasets for training and comparison

From the repository root on the server, run:

```bash
python tools/prepare_real_sem_blocks.py \
  --raw-dir /data/260904_raw_data \
  --prepared-dir /data/260904_prep_data/train_align_none \
  --output-dir /data/260929_block_data \
  --device cuda:0 --cpu-threads 2
```

This creates all four variants. The input paths above follow the dataset paths
specified for this task. `--prepared-dir` is the existing dataset whose site
splits should be preserved; use a different path if that dataset has moved.

| Variant directory | Averaging | Output images per 128-frame site |
|---|---|---:|
| `average2` | Consecutive pairs, no registration | 64 |
| `average4` | Consecutive groups of four, no registration | 32 |
| `average2_registered` | Affine registration within each pair | 64 |
| `average4_registered` | Affine registration within each group of four | 32 |

Each variant has this layout:

```text
/data/260929_block_data/average2/
  raw/
    all/<site>/block_001_001-002.png ...
    train/<site>/block_001_001-002.png ...
    test/260904_0947-13/block_001_001-002.png ...
  train_align_none/
    arrays/
    previews/
    qc.csv
    real_dataset.json
  site_splits.json
```

The other three directories have the same structure. The script mirrors every
site in `all`, `train`, and `test`. When copies in `all` have identical ordered
pixel hashes, their generated images are reused without repeating registration.
The prepared training cache uses only the raw `train` tree. Its train/validation/
test site assignments are copied from the existing `real_dataset.json`.
The external raw `test` tree never enters training preparation.

## Train using a variant

Set `data.dataset_dir` to the variant's **`train_align_none`** directory, or pass
the existing command-line override. For example:

```bash
VARIANT=average2
python -m edge_denoise train \
  --config edge_denoise/configs/sem_real_n2n.yml \
  --dataset-dir "/data/260929_block_data/$VARIANT/train_align_none" \
  --run-dir "runs/edge_denoise/260929_real_n2n_${VARIANT}_affine_percentile" \
  --real-registration affine --real-brightness percentile \
  --device cuda:0 --cpu-threads 2
```

Change `VARIANT` to any row in the table. These caches also work with the
existing fine-tuning, gradient, hybrid, and consistency recipes and suite's
`--dataset-dir` override. Use a separate experiment directory for each variant.
An old checkpoint cannot be resumed onto a different dataset; use the existing
weights-only initialization option for an intentional fine-tuning experiment.

`train_align_none` describes cache preparation: it stores the delivered block
images without another registration pass. The registered variants have already
aligned acquisitions **within** each block. The recipe's inline registration
still matches **different block images** when forming training pairs. Its
percentile matching is a training choice; dataset generation applies no
gain/offset or brightness matching.

## Compare using block-averaged test images

Change the comparison YAML's site directory to:

```yaml
sites:
  260904_0947-13: /data/260929_block_data/average2/raw/test/260904_0947-13
```

Or use the existing `--site-dir` override. Here is a complete command for the
example training run above:

```bash
VARIANT=average2
python tools/real_sem_compare.py \
  --config edge_denoise/configs/sem_real_compare.yml \
  --only-checkpoints \
  --checkpoint "n2n=runs/edge_denoise/260929_real_n2n_${VARIANT}_affine_percentile/ckpt_latest.pt" \
  --site-dir "/data/260929_block_data/$VARIANT/raw/test/260904_0947-13" \
  --output-dir "output/260929_real_n2n_${VARIANT}_comparison" \
  --device cuda:0 --metrology-device cuda:0 --no-tensorboard
```

Repeat `--checkpoint NAME=PATH` to include other trained models. Keep the
generated `block_average.json` and `.support.npz` files with their image folder:
the comparison reads them automatically, verifies the files, accepts 64 or 32
inputs, and checks original acquisition hashes against training/validation.
The full-reference mean divides by the actual number of supplied images, and
the viewer labels that count. The old internal `average128` JSON key remains
for compatibility; it means the full-input average in these reports.

In these comparisons, one input is already a block image. The report's
`--average-frames 8` baseline therefore averages eight **block images** (16 or
32 original acquisitions). Its K axis counts supplied input images. An optional
`frame_interval_s` still refers to the original acquisition interval; generated
block timestamps use original block centres. Explicit `timestamps_s` instead
requires one time per supplied block image.

## Averaging and validation details

- Files use natural filename order (`frame_2` before `frame_10`). Groups are
  consecutive and nonoverlapping; average4 is calculated directly from four
  originals, avoiding the extra rounding of two already-quantized average2s.
- Dimensions stay unchanged. RGB input must have identical channels. Output is
  lossless uint8 RGB with identical channels, using float64 accumulation and a
  single round-to-nearest conversion. Training preparation and report analysis
  read the saved PNGs.
- Registration reuses the comparison's translation-seeded affine ECC estimator
  and bicubic sampler. Each block starts with its first usable frame as the
  reference. Blank/noise-only/low-contrast and failed-fit frames remain included
  in native coordinates, with recorded reasons. Failed fits do not change the
  last successful seed. The affine fit uses CPU OpenCV; bounded batched warps
  and accumulation use the selected device. No whole dataset is loaded into RAM.
- Outside common registration support, the delivered PNG keeps the native
  unregistered mean. Support masks exclude those pixels from report contour
  measurements for inputs, predictions and further averages. Training consumes
  the delivered images with the usual crop margin; it does not consume support
  masks as a loss mask.
- The existing training manifest must match raw training content and order.
  Original content is hashed before copying splits. Independent blocks that
  quantize to identical pixels, including blanks, are retained. Split checks
  use their original acquisition hashes. Ordinary preparation still rejects
  duplicate raw frames as before.
- By default every site must have exactly 128 images. Missing/extra files cause
  an error instead of silently dropping a partial block. `--expected-frames`
  can specify another count, at least eight and divisible by four. Use
  `--image-size` if the minimum training crop differs from 512.

## Resume and remote performance check

If preparation is interrupted, rerun the same command with `--resume`:

```bash
python tools/prepare_real_sem_blocks.py \
  --raw-dir /data/260904_raw_data \
  --prepared-dir /data/260904_prep_data/train_align_none \
  --output-dir /data/260929_block_data \
  --device cuda:0 --cpu-threads 2 --resume
```

Completed sites and training caches are verified and reused. An interrupted site
is rebuilt; it is never published as complete. Changed inputs, settings, or
saved files are rejected. `block_datasets.json` is written after all requested
variants and caches complete. `--variants average2 average4` selects a subset
when desired; a resumed invocation must select the same variants.

Timing includes source decoding/hashing and the full build in
`block_datasets.json`. Each site manifest also records image production/export
times and per-block CPU geometry and device warp/average times. CPU/GPU checks
can run directly on the raw test site, without an existing comparison report:

```bash
for K in 2 4; do
  python tools/check_real_sem_averages.py \
    --site-dir /data/260904_raw_data/test/260904_0947-13 \
    --output-dir "output/260929_block_average${K}_equivalence" \
    --average-frames "$K" --device cuda:0 --cpu-threads 2 || break
done
```

These checks compare decoded saved CPU/GPU PNGs and support masks on the first
block. They report total time including export/decode and fail if differences
exceed the existing tolerance (at most 1 DN, at most 0.1% changed pixels, identical
support). They are a small pilot, not a full-dataset equivalence benchmark.
H100 throughput has not been measured locally. GPU indices remain relative to
the scheduler's existing visibility; the command does not change it.
