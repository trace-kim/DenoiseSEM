# Real-SEM registration and percentile training comparisons

Use the existing **prepared `align none` dataset** for these comparisons. Its
`arrays/*_raw.npy` files preserve native acquisitions, and its manifest keeps
the original content hashes, normalization, and whole-site splits. Preparation
still supplies that storage/audit step. There is no need to prepare four new
datasets, rewrite the raw images, or register the network's inputs.

`train --real-registration ... --real-brightness ...` enables pair-time
matching. Omitting both options preserves the original prepared-data workflow.
The explicit matching path requires a prepared `align none` dataset so that
preparation and training cannot silently apply two different registrations.

## What is measured and applied

- **Translation** uses the existing training translation estimator, with the
  sigma, search radius, maximum shift, contrast threshold and failure policy
  recorded in the prepared manifest. This preserves the translation baseline's
  algorithm. Confirm these settings match the earlier translation preparation
  when comparing completed runs. It is not the report's translation ECC fit.
- **Affine** uses the report's full-frame translation-initialized affine ECC,
  against the first usable frame, with 1-pixel fit blur and clipped pixels
  excluded from fitting. Before ECC, apply the same native-frame block-contrast
  gate as translation, using the prepared manifest's `min_contrast` (default
  0.005). Blank/noise-only frames stay unregistered. Unmeasurable support or
  failed ECC estimates retain those frames without registration by default.
- **None** uses native coordinates without estimating geometry.
- **Percentile** fits `Q_A(p) = gain * Q_B(p) + offset` at 10,15,...,90%.
  Percentiles come from every pixel of each full raw frame, including clipping
  bounds, before registration or crop selection. The same 17-point OLS as
  the diagnostic is evaluated for each selected B→A pair. It does not compose
  mappings through the first frame and does not fit individual 512-pixel crops.
  If either frame's central percentiles are all equal, gain is unmeasurable:
  keep gain 1 and offset 0 for that pair, and record the skipped brightness fit.
- **Brightness none** leaves gain 1 and offset 0.

Frame transforms and percentile points are measured once when the run starts.
Each sample crops A directly from the raw array, composes the chosen A→B
transform, samples original B once, and applies its B→A brightness mapping.
The original dataset-wide intensity normalization remains in effect. Corrected
targets remain floating point and are not clipped back into `[0,1]` or the
storage dtype. A and the raw dataset are never modified. Registration blur is
only for estimation; no smoothing is applied to training inputs or targets.

Affine sampling excludes pixels without a full cubic interpolation footprint.
Losses retain the existing three-pixel border exclusion, additionally masking
invalid targets. Sobel loss also requires valid neighbouring target pixels.
Each crop contributes equally to the loss. No-overlap crops produce an explicit
error instead of supervising padded pixels. Translation crops/sampling retain
the old training behavior, including its common-overlap bounds.

The run writes `real_matching.json` with the dataset fingerprint, settings,
per-frame matrices, percentile points and registration/brightness skip records.
These measurements are also saved in checkpoints and restored on resume.
Changing matching settings requires a new run, not `--resume`. The dataset and
all its arrays are content-verified through the existing cache loader.

### Frames whose geometry cannot be measured

Affine matching and the five-model suite skip failed geometry by default; no
extra flag is needed. `--registration-failure error` explicitly requests strict
ECC failure handling on frames that pass the contrast gate. Translation still
inherits the prepared manifest's failure policy; `--registration-failure skip`
also handles rejected translation estimates (implausible shift/nonfinite
covariance). Low-contrast frames remain unmeasured under either policy.

Both paths use the standard deviation of nonoverlapping 16-pixel block means
in fixed dataset-normalized units. This existing heuristic suppresses pixel
noise; it is not a guarantee that every noisy acquisition has usable structure.
ECC nonconvergence or invalid numerical results are therefore also skippable.
An unusable first acquisition never becomes the affine reference; the first
usable frame does. An entirely blank site keeps all its native frames, with no
reference and no geometry available. A skipped fit cannot update the next
frame's translation seed. Successful ECC fitting and target sampling are unchanged.

**Skip means skip the correction, never drop the frame.** If either A or B has
unavailable registration, B is read at A's native crop coordinates with no
geometric correction. Do not compose a failed frame's identity placeholder
with the other frame's measured transform. Successful pairs still compose
normally. This rule also applies to every member of a leave-one-out mean and
the separately selected consistency pair. Percentile brightness matching stays
independent and still uses the actual sampled frames.

Each inline run writes `registration_report.html` and `registration_frames.csv`
beside `real_matching.json`, after measurements/restoration and before optimizer
updates. They remain available after training and list every measured train/val
frame's original relative filename, zero-based prepared frame index, split,
status, reason, available contrast/shift diagnostics and stored matrix. The HTML
lists failed/low-contrast frames first. None are removed from their original
split; test images are not measured. A fallback identity is labelled unmeasured,
not a successful zero-motion estimate. The report is fully local, with no
external resources; data and reports can stay on the training server.

Registration failure is not proof of low contrast: inspect the recorded reason.
Unrelated I/O, invalid arguments and OpenCV API/dependency errors still surface.
Brightness remains independent: noisy blank frames with a measurable percentile
spread still use the full-frame percentile fit. If either frame has flat
percentiles, both directions of that pair use native brightness, including
leave-one-out targets and consistency pairs. The report lists
`skipped_flat_percentiles` alongside registration status.

New measurement caches use format version 2. A matching version-1 shared cache
is refreshed once, with a log message, so new runs cannot bypass the gate.
Dataset/settings mismatches still fail validation. Existing checkpoint
measurements remain resumable without re-estimation.

## Four independent N2N runs

Run from the repository root with the updated code and analysis dependencies:

```bash
python -m pip install -e ".[analysis]"
```

In each terminal, set `DATASET` to the same prepared `align none` directory.
The paths below are portable examples; substitute the existing server path.
Use only your allocated GPU IDs. Each command uses one GPU, effective batch
`16 * 4 = 64`, native 512-pixel patches, 100,000 optimizer updates, FP32 and
learning rate 0.0002. Run-directory names are distinct and must be new.

Terminal 1 — translation + percentile:

```bash
DATASET=data/SEM-real-none
CUDA_VISIBLE_DEVICES=0 python -m edge_denoise train \
  --config edge_denoise/configs/sem_real_n2n.yml --dataset-dir "$DATASET" \
  --real-registration translation --real-brightness percentile \
  --run-dir runs/edge_denoise/sem_real_n2n_translation_percentile \
  --image-size 512 --batch-size 16 --accumulation-steps 4 \
  --max-steps 100000 --lr 0.0002 --precision fp32 --device cuda
```

Terminal 2 — affine + percentile:

```bash
DATASET=data/SEM-real-none
CUDA_VISIBLE_DEVICES=1 python -m edge_denoise train \
  --config edge_denoise/configs/sem_real_n2n.yml --dataset-dir "$DATASET" \
  --real-registration affine --real-brightness percentile \
  --run-dir runs/edge_denoise/sem_real_n2n_affine_percentile \
  --image-size 512 --batch-size 16 --accumulation-steps 4 \
  --max-steps 100000 --lr 0.0002 --precision fp32 --device cuda
```

Terminal 3 — no registration + percentile:

```bash
DATASET=data/SEM-real-none
CUDA_VISIBLE_DEVICES=2 python -m edge_denoise train \
  --config edge_denoise/configs/sem_real_n2n.yml --dataset-dir "$DATASET" \
  --real-registration none --real-brightness percentile \
  --run-dir runs/edge_denoise/sem_real_n2n_none_percentile \
  --image-size 512 --batch-size 16 --accumulation-steps 4 \
  --max-steps 100000 --lr 0.0002 --precision fp32 --device cuda
```

Terminal 4 — affine + no brightness correction:

```bash
DATASET=data/SEM-real-none
CUDA_VISIBLE_DEVICES=3 python -m edge_denoise train \
  --config edge_denoise/configs/sem_real_n2n.yml --dataset-dir "$DATASET" \
  --real-registration affine --real-brightness none \
  --run-dir runs/edge_denoise/sem_real_n2n_affine_none \
  --image-size 512 --batch-size 16 --accumulation-steps 4 \
  --max-steps 100000 --lr 0.0002 --precision fp32 --device cuda
```

For a smoke run, use a separate run directory and `--max-steps 10` first.
Startup performs full-frame measurements before optimizer steps begin, and
each independent run computes its own measurements. Resume with the same
command plus `--resume`; do not add `--overwrite` to continue a run.

## Second-stage consistency after all N2N arms finish

Each arm starts from **its own** N2N checkpoint, retaining its data and matching
settings. The real-data recipe `sem_real_ft_avgfull_consist.yml` uses the
leave-one-out target, with image/Sobel/consistency weights 1/4/1 and 10,000
additional optimizer steps. It is a fresh optimizer/EMA run from the teacher's
EMA weights (`--init-checkpoint`), not a resume of its 100,000-step budget.

For example, the affine + percentile continuation is:

```bash
CUDA_VISIBLE_DEVICES=1 python -m edge_denoise train \
  --config edge_denoise/configs/sem_real_ft_avgfull_consist.yml \
  --dataset-dir "$DATASET" --real-registration affine --real-brightness percentile \
  --init-checkpoint runs/edge_denoise/sem_real_n2n_affine_percentile/ckpt_latest.pt \
  --run-dir runs/edge_denoise/sem_real_ft_consist_affine_percentile \
  --image-size 512 --batch-size 16 --accumulation-steps 4 \
  --max-steps 10000 --lr 0.0002 --precision fp32 --device cuda
```

The mean target excludes A. Every other frame is sampled into A and, when
enabled, percentile-matched to A **before** averaging on common valid pixels.
This simple implementation processes all other target crops; it costs more CPU
work than N2N and should be timed before scheduling the six continuations.
Both consistency inputs stay native. The second prediction is differentiably
mapped into the first prediction's geometry and brightness before comparison,
with interpolation and network boundary support masked. Neither prediction
is used to estimate the correction. The validation consistency readout uses
the same mapping and validity rules.

For the two completed baselines, retain their original prepared dataset and
omit both inline matching flags to preserve their existing preprocessing.
Use each baseline's own checkpoint and a separate continuation directory.
Consistency processes two network inputs per sample; if batch 16 exceeds GPU
memory, batch 8 with accumulation 8 preserves effective batch 64. Run six
continuations in two waves when four GPUs are available.

Tests cover the four N2N arms, mean/consistency training and resume, full-frame
percentile equivalence, unchanged native inputs and translation behavior,
single affine sampling, fixed normalization, unclipped corrected values, valid
loss support, differentiability, and independent failure paths. H100 memory and
throughput require measurement on the training server.
