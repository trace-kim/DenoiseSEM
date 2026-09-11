# Train and evaluate denoisers on real SEM repeats

This guide covers real noisy SEM data: approximately 15–30 sites, 128 repeats
per site, 1024×1024 frames, and native 512×512 training patches. No clean
images or simulator manifest are required. Every model in this guide takes
**one raw frame at inference**.

The workflow is:

1. Organize and inspect the acquisitions.
2. Prepare native frame arrays, registration, and fixed site splits once.
3. Run a short training and inference smoke check.
4. Train a Noise2Noise teacher, then compare four continuations.
5. Evaluate held-out sites and test full-frame inference.

Run commands from the repository root. Commands below use a Linux shell for
the training server. Data and run paths are relative and can be changed in
the YAML recipes. The existing synthetic workflows keep their original format.

## 1. Set up the server environment

Use a server environment with a CUDA-enabled PyTorch installation compatible
with its NVIDIA driver. Install this repository into that environment:

```bash
python -m pip install -e ".[dev]"
nvidia-smi
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA build:', torch.version.cuda); print('Visible GPUs:', torch.cuda.device_count()); print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])"
python -m edge_denoise --help
```

The real-data recipes explicitly select CUDA, so a missing CUDA installation
fails instead of starting a long CPU run. A development CPU test is available
in step 10; full-resolution CPU training is not the intended server workflow.

The GPU masks below are examples for a server managed directly. Use only the
GPUs allocated to you. If Slurm or another scheduler already sets
`CUDA_VISIBLE_DEVICES`, preserve its allocation and select the corresponding
process count instead of replacing its mask with physical GPU IDs.

## 2. Organize the raw images

Place one acquisition site in each folder:

```text
data/SEM-raw/
  site_01/
    frame_000.jpg
    frame_001.jpg
    ...
    frame_127.jpg
  site_02/
    frame_000.jpg
    ...
```

Supported inputs are grayscale uint8 or uint16 PNG, TIFF, or BMP files, and
8-bit JPEG (`.jpg`/`.jpeg`) files, with one frame per file. RGB images are
accepted only when all three decoded channels match exactly; one channel is
extracted without averaging or resizing. Unequal RGB channels and
floating-point acquisitions are rejected by preparation. Export multi-page
TIFFs as individual frames before this step. Images within a site must have
matching dimensions; the dataset must use one storage bit depth.

The same identical-channel RGB handling applies to inference with real SEM
checkpoints. The cached pixels preserve the decoded JPEG values; preparation
does not re-encode them as JPEG.

Filenames are sorted naturally (`frame_2` before `frame_10`). Ensure that this
order is the actual acquisition order: registration uses the preceding shift
estimate to avoid locking onto the wrong period of a repetitive pattern.
Keep magnification, detector mode, scan settings, and pixel pitch consistent
within a training dataset. Remove acquisition labels/scale bars consistently
before import if they occupy pixels that should not be learned.

Keep the original files. Preparation never overwrites them and never resizes
the images. Exact duplicate frames within a site are rejected because copies
are not independent measurements. Sites sharing decoded image content are
grouped together for splitting. Visually repeated patterns that differ in
pixels still require your judgment when assigning generalization holdouts.

### Choose the intensity scale

Normalization is fixed for the dataset and is saved in each trained checkpoint:

```text
unit intensity = clip((stored value - black level) / (white level - black level), 0, 1)
```

The model then uses `2 * unit intensity - 1`. There is no per-frame min/max
normalization and no conversion of raw uint16 pixels to uint8.

For ordinary uint8 data, the default white level is 255. For uint16 storage,
the default is 65535. **A 12-bit detector stored in uint16 may need 4095
instead**, depending on whether the instrument stretches its exported values.
Confirm the export convention from the instrument; storage dtype alone does
not determine the detector's useful intensity range. Use a measured black
level if appropriate. The QC report records clipping and endpoint fractions.

## 3. Prepare the dataset once

Example for 1024×1024 8-bit JPEG frames, using one allocated GPU for registration:

```bash
CUDA_VISIBLE_DEVICES=0 python -m edge_denoise prepare-real \
  --source-dir data/SEM-raw \
  --out data/SEM-real \
  --image-size 512 \
  --white-level 255 \
  --align translation \
  --device cuda
```

Use `--white-level 65535` for true 16-bit values or `--white-level 4095`
for an appropriate 12-bit export, or omit the option
to use the storage range. Preparation also works with `--device cpu`.

The output directory must be new and outside the raw site folders. A failed
preparation does not publish a partial dataset. Reprepare into a new directory
when changing the data, normalization, registration settings, or splits.

Preparation stores full 1024×1024 frames, not a fixed set of patches.
`--image-size 512` checks that the common registration overlap can fit a
512×512 crop. Keep `data.image_size: 512` and `data.channels: 1` in all five
training recipes. Each training example draws a fresh random site, window,
and repeat selection, including each accumulation microbatch. Targets use
the corresponding registered region; raw inputs are not resized. Crops stay
inside the valid registration overlap, excluding its border margins.
Validation and evaluation retain fixed center/corner regions for comparison.

Output:

```text
data/SEM-real/
  real_dataset.json        file hashes, frame order, shifts, normalization, fixed splits
  qc.csv                   per-frame shift, uncertainty, intensity, clipping information
  arrays/
    00000_raw.npy          native uint8/uint16 frames [N,H,W]
    00000_aligned.npy      float32 registered copies, for targets/references
    00000_sum.npy          float32 registered sum, accumulated initially in float64
    ...
  previews/
    00000.png              first raw / last raw / registered mean preview
    00000.tif              full-resolution float32 reference mean
    ...
```

There is no fake clean image and no need to run `noising_pipeline`. The real
dataset has its own small metadata file; the legacy synthetic
`manifest.jsonl` format remains supported separately.

For 15–30 sites with 128 grayscale 1024×1024 uint8 frames, the prepared raw
and aligned arrays occupy approximately **9.4–18.8 GiB**, plus sums and previews.
The original inputs and checkpoints need additional disk space. Put the
prepared directory on fast local storage when available. Training opens the
arrays read-only through memory mapping, allowing processes to share OS file
cache pages instead of allocating four complete Python RAM caches.
Startup verifies the prepared file hashes, which reads the arrays and can
take time on a cold or network filesystem. The manifest is also fingerprinted
in checkpoints; a changed prepared dataset cannot silently resume an old run.

### Inspect acquisition quality before training

Inspect `qc.csv`, the preview strips, and the **full-resolution reference
TIFFs**. Check multiple parts of each site at native scale:

- Do edges become clearer in the registered mean, or show doubled contours?
- Do first and last frames show charging, contrast changes, contamination,
  new damage, or changing features?
- Are shifts plausible and continuous? Are uncertainty estimates large in
  directions that parallel line patterns cannot constrain?
- Are significant values clipped by the selected intensity scale?

Registration currently fits translations. It uses smoothed copies to estimate
motion; training inputs remain unsmoothed. It does not correct local charging
distortion, raster shear, rotation, or specimen changes. A plausible shift
estimate is not proof that the entire image is registered well.

If needed, change `--sigma`, `--radius`, or `--max-shift` after inspecting the
failure. The default search radius is 6 pixels around the preceding estimate;
the default allowed absolute shift is 32 pixels in either axis. Do not simply
raise the limit when the estimator has matched the wrong repeated feature.

To use a stable interval shared by all sites, for example frames 8–95:

```bash
CUDA_VISIBLE_DEVICES=0 python -m edge_denoise prepare-real \
  --source-dir data/SEM-raw --out data/SEM-real-stable \
  --image-size 512 --white-level 255 \
  --frame-start 8 --frame-stop 96 --device cuda
```

The interval is zero-based and the stop is exclusive, after filename sorting.
For site-specific exclusions, organize a separate curated input directory
containing the accepted frames in acquisition order. Use `--align none` only
for data already known to be aligned, or for a deliberate alignment ablation.

### Freeze whole-site holdouts

Preparation defaults to 10% validation and 10% test sites, using seed 2019.
With distinct sites this gives approximately 11/2/2 at 15 sites, or 24/3/3 at
30. Actual site assignments are printed and saved. All repeats and all patches
from a site stay in one split.

For deliberate assignments, supply `--split-file splits.json`. Its contents
map **every** folder name to a split; a three-site illustration is:

```json
{"site_01": "train", "site_02": "val", "site_03": "test"}
```

Add entries for every actual site. Assignments that put duplicate image
content across splits are rejected. Prepared splits are fixed: the legacy
fraction/seed fields in a training YAML do not redraw them. Use the same
prepared dataset for every experimental arm and keep the test sites reserved
until the recipe has been selected using validation.

## 4. Run a training smoke check

The supplied recipes point to `data/SEM-real`. Change `data.dataset_dir` if
you chose another prepared directory. Start with a separate smoke run:

```bash
CUDA_VISIBLE_DEVICES=0 python -m edge_denoise train \
  --config edge_denoise/configs/sem_real_n2n.yml \
  --run-dir runs/edge_denoise/sem_real_smoke \
  --max-steps 10 --batch-size 1 --precision fp32
```

Confirm finite losses, a validation image, `ckpt_latest.pt`, and
`provenance.json` in the run directory. The final step always validates and
saves even when the ordinary interval has not been reached. Smoke-check
inference using step 9 and the smoke checkpoint; its image quality is not a
trained-model result.

Before a long distributed run, repeat the smoke check on two or four GPUs
with a new run directory:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  -m edge_denoise train \
  --config edge_denoise/configs/sem_real_n2n.yml \
  --run-dir runs/edge_denoise/sem_real_ddp_smoke \
  --max-steps 10 --batch-size 1 --precision fp32
```

Confirm one process on each allocated GPU with `nvidia-smi`. The shared run
directory should contain one TensorBoard history and ordinary loadable
checkpoints. Check the recorded world size:

```bash
python -c "import torch; p=torch.load('runs/edge_denoise/sem_real_ddp_smoke/ckpt_latest.pt', map_location='cpu', weights_only=True); print('step:', p['step'], 'world size:', p.get('distributed', {}).get('world_size', 1))"
```

These server smoke checks are required to establish actual CUDA/NCCL behavior
and memory capacity; the repository tests exercise CPU data/training paths
and mocked distributed control, not the remote H100s.

## 5. Train the common N2N teacher on 1, 2, or 4 GPUs

`training.batch_size` is **per GPU per microstep**. The optimizer's effective
batch is:

```text
batch_size × accumulation_steps × number of GPU processes
```

The following alternatives all use an effective batch of 16. Choose one:

```bash
# Four GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  -m edge_denoise train --config edge_denoise/configs/sem_real_n2n.yml \
  --batch-size 4 --accumulation-steps 1
```

```bash
# Two allocated GPUs
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  -m edge_denoise train --config edge_denoise/configs/sem_real_n2n.yml \
  --batch-size 4 --accumulation-steps 2
```

```bash
# One allocated GPU
CUDA_VISIBLE_DEVICES=0 python -m edge_denoise train \
  --config edge_denoise/configs/sem_real_n2n.yml \
  --batch-size 4 --accumulation-steps 4
```

DDP uses one model replica per GPU; GPU memory is not pooled into one larger
device. A single-process CUDA run expects one GPU isolated by the allocation
or mask. Each rank samples different patches, gradients are synchronized,
and only rank 0 writes checkpoints, validation output, logs, and provenance.

The teacher recipe allows 30,000 optimizer steps. Treat this as an initial
budget, then judge validation behavior. `--max-steps 2000` is useful for a
shorter pilot. A fresh run refuses an occupied run directory; use a new
`--run-dir` or resume it.

### Optional BF16 on H100

Recipes default to FP32. After the FP32 smoke check, benchmark a separate
short run with `--precision bf16`. Model forward passes use autocast;
parameters, optimizer/EMA state, loss calculations, and the ordinary paired
validation/inference path remain FP32. This can improve throughput and memory
use, but compare reference-based metrics and fine features against an FP32
pilot before adopting it for metrology.

Increase batch size only after measuring peak memory, including validation.
Consistency training retains activations for two views and needs more memory
than N2N at the same configured batch. Use gradient accumulation to retain
the desired effective batch when fewer GPUs are available.

## 6. Compare four continuations from the same teacher

The teacher checkpoint is expected at
`runs/edge_denoise/sem_real_n2n/ckpt_latest.pt`. Use
`--init-checkpoint path/to/teacher.pt` if it is elsewhere.

| Recipe in `edge_denoise/configs/` | Target | Image / Sobel / consistency weights |
|---|---|---|
| `sem_real_ft_n2n.yml` | another noisy frame | 1 / 0 / 0 |
| `sem_real_ft_mean.yml` | leave-one-out registered mean | 1 / 0 / 0 |
| `sem_real_ft_avgfull.yml` | leave-one-out registered mean | 1 / 4 / 0 |
| `sem_real_ft_avgfull_consist.yml` | leave-one-out registered mean | 1 / 4 / 1 |

Each continuation allows 10,000 optimizer steps and starts fresh optimizer
and EMA state from the teacher's EMA weights. The N2N continuation controls
for the additional training budget. All five recipes use the same 512-pixel
backbone; the older 64-pixel MIIC checkpoints have a different architecture.

The smallest way to run all comparisons at once is one experiment per GPU.
After the teacher finishes, the following uses effective batch 16 in each arm:

```bash
CUDA_VISIBLE_DEVICES=0 python -m edge_denoise train --config edge_denoise/configs/sem_real_ft_n2n.yml --accumulation-steps 4 > runs/edge_denoise/ft_n2n.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python -m edge_denoise train --config edge_denoise/configs/sem_real_ft_mean.yml --accumulation-steps 4 > runs/edge_denoise/ft_mean.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 python -m edge_denoise train --config edge_denoise/configs/sem_real_ft_avgfull.yml --accumulation-steps 4 > runs/edge_denoise/ft_avgfull.log 2>&1 &
CUDA_VISIBLE_DEVICES=3 python -m edge_denoise train --config edge_denoise/configs/sem_real_ft_avgfull_consist.yml --accumulation-steps 4 > runs/edge_denoise/ft_avgfull_consist.log 2>&1 &
wait
```

When sharing the server, run only as many jobs as your allocation supports.
Alternatively, train each continuation with the DDP commands in step 5,
substituting its configuration. Use the same effective batch and common
teacher for the comparison. Consistency uses two views, so equal optimizer
steps and input-target counts do not imply equal GPU time.

Mean targets exclude the input frame's registered contribution before being
sampled back into its native coordinates. The consistency comparison also
accounts for relative subpixel displacement. A three-pixel loss border is
excluded to avoid interpolation and convolution boundary artifacts.
Interpolation can still soften targets, and motion estimated from noisy data
can affect noise independence. Assess the resulting edge profiles rather
than treating the registered mean as an exact unbiased clean image.

## 7. Monitor, stop, and resume

```bash
tensorboard --logdir runs/edge_denoise
```

Useful scalars are `train/loss`, its component losses,
`train/patches_per_sec`, `train/effective_batch`, `val/loss`, and
`val/consistency_sigma`. `val/input_pred_target` shows raw input, prediction,
and target. Real-data training deliberately does not report `val/psnr` against
a nonexistent clean image. A falling consistency score alone can reward
over-smoothing; pair it with the evaluation below.

To stop cooperatively after an optimizer step:

```bash
touch runs/edge_denoise/sem_real_n2n/stop
```

All ranks observe the stop request and save the latest checkpoint. Use the
corresponding run directory for a continuation. A subsequent launch removes
the old stop marker. Resume with the same recipe and batch settings:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc-per-node=4 \
  -m edge_denoise train --config edge_denoise/configs/sem_real_n2n.yml --resume
```

`--max-steps` is the desired **total** step count, including completed steps.
`--resume-from path/to/ckpt_0005000.pt` selects a milestone explicitly. A
fine-tune can resume with `init_checkpoint` still present in its YAML; resume
loads its saved model/optimizer/EMA state rather than reinitializing it.

When changing from four GPUs to two, preserve effective batch 16:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  -m edge_denoise train --config edge_denoise/configs/sem_real_n2n.yml \
  --resume --batch-size 4 --accumulation-steps 2
```

The model, optimizer, and EMA are restored. A change in world size starts new
rank-specific random streams and emits a warning; it is not bit-for-bit
replay. With an unchanged world size and recipe, each rank's sampler and
PyTorch RNG state are restored. CUDA kernels can still limit strict numerical
reproducibility.

## 8. Evaluate the validation sites

Run the models on the same raw input frames. A separate frame pool supplies
the registered reference average and fixed measurement boxes:

```bash
CUDA_VISIBLE_DEVICES=0 python -m edge_denoise evaluate-real \
  --config edge_denoise/configs/sem_real_n2n.yml \
  --checkpoint n2n=runs/edge_denoise/sem_real_ft_n2n/ckpt_latest.pt \
  --checkpoint mean=runs/edge_denoise/sem_real_ft_mean/ckpt_latest.pt \
  --checkpoint avgfull=runs/edge_denoise/sem_real_ft_avgfull/ckpt_latest.pt \
  --checkpoint consist=runs/edge_denoise/sem_real_ft_avgfull_consist/ckpt_latest.pt \
  --out output/sem_real_val \
  --split val --input-frames 32 --rois 5 --max-batch 4 --device cuda
```

With 128 accepted frames this uses 32 raw inference inputs and the other 96
as references. Pools are deterministic, spread across acquisition order, and
recorded in `results.json`; they share no image frames. At least two input
frames and two remaining reference frames are required. Registration itself
was estimated from the acquisition, so this is reference-based evaluation,
not a claim of fully independent ground truth.

The evaluator visits fixed center/corner windows inside common overlap. It
uses the saved acquisition transforms to compare outputs in common
coordinates, without re-registering each model output to hide edge shifts.
`results.json` contains per-site/per-ROI details, checkpoint hashes, and:

- `psnr_vs_reference` and `ssim_vs_reference`: similarity to the reference mean.
- `gradient_mse_vs_reference`: edge-field differences relative to that mean.
- `pixel_sigma`: RMS temporal output variation over the evaluated pixels.
- `cd_3sigma_px`: three times the sample CD standard deviation, summarized
  over measurable edge locations.
- `cd_bias_vs_reference_px`: systematic width difference from the reference.
- `cd_failure_fraction`: fraction of attempted edge measurements that failed.

All distance units are native pixels. Absolute accuracy against a truly clean
specimen is unavailable. A null CD result means insufficient measurable
locations or valid realizations, not perfect repeatability. Inspect failure
rates, small features, and edge profiles alongside variance.

The summary takes the median across sites after averaging each site's ROI
metrics. It does not treat overlapping patches or repeated images as new
independent sites. PNG comparisons are previews of the same fixed regions.

Registered averages of 4, 8, and 16 input frames are included when at least two
disjoint groups are available. These consume more frames per result than the
single-frame models. With 32 inputs, avg-of-16 has only two realizations and
its repeatability estimate is correspondingly uncertain. The full acquisition
mean in preparation previews is a QC reference, not a repeatability baseline.

Select the recipe on validation sites. Then run the chosen checkpoint once
with `--split test` and a separate output directory. Keep inference settings
and measurement settings fixed for that final comparison. The historical
`evaluate`/`repeatability` commands use synthetic clean references; use
`evaluate-real` for these prepared real datasets.

## 9. Test inference on a full raw frame

```bash
CUDA_VISIBLE_DEVICES=0 python -m edge_denoise denoise \
  --checkpoint runs/edge_denoise/sem_real_ft_avgfull_consist/ckpt_latest.pt \
  --input data/SEM-raw/site_01/frame_000.jpg \
  --out output/sem_real_inference \
  --stride 256 --tile-batch 4 --device cuda
```

The 1024×1024 image is processed in overlapping 512×512 tiles and blended
back to the original dimensions. It is not downscaled. The checkpoint
supplies the intensity scale, and inference requires only this one raw frame.
It does not need the prepared dataset, registration files, or additional
acquisitions.

Outputs are:

```text
frame_000_input.png        8-bit input preview
frame_000_denoised.png     8-bit output preview
frame_000_denoised.tif     float32 normalized output for quantitative use
```

The TIFF values are in `[0,1]`, not original detector counts. Convert back
with `output * (white_level - black_level) + black_level` using the checkpoint
normalization if your measurement software expects detector units. Keep the
float result for analysis; the PNG is only a preview.

Check dimensions, intensity range, tile boundaries, edge widths, and small
features. Try `--stride 384` only after confirming it gives acceptable seams
and measurements. Reduce `--tile-batch` if inference runs out of memory.
`--center-crop` explicitly tests one 512×512 tile instead of the full frame.

For example, inspect the quantitative output:

```bash
python -c "from PIL import Image; import numpy as np; a=np.asarray(Image.open('output/sem_real_inference/frame_000_denoised.tif')); print(a.shape, a.dtype, a.min(), a.max())"
```

## 10. Development verification and Windows notes

The relevant automated checks run on CPU, with filesystem/process/GPU
behavior isolated or mocked:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest tests/edge_denoise tests/burst_diffusion -q
```

They cover native precision, 128-frame sums, input exclusion, registration
geometry, fixed splits, data tampering, preparation failure cleanup, training,
resume, CLI inference/evaluation, gradient accumulation, and distributed
rank/writer/stop behavior. Run `python -m pytest` for the complete repository
suite. Actual H100 throughput, BF16 quality, and NCCL execution must be checked
on the server with the smoke runs above.

On Windows, use your activated development environment and set a GPU mask in
PowerShell before the Python command:

```powershell
$env:CUDA_VISIBLE_DEVICES = '0'
python -m edge_denoise train --config edge_denoise/configs/sem_real_n2n.yml
```

The single-process preparation/training/inference paths work on Windows.
The CUDA DDP examples target Linux/NCCL on the remote server. All generated
datasets, run bundles, checkpoints, and evaluation outputs should stay out of
version control.
