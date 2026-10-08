# Continuing ft_consist and the burst model on 8-frame sites

Use this when a new acquisition folder holds **consecutive repeats of several
sites in one flat folder**: images 1–8 are site 1, 9–16 are site 2, and so on
(natural filename order). Both models start from their latest checkpoints;
only the EMA weights are loaded, and the optimizer, EMA and step count start
fresh (`--init-checkpoint`, not `--resume`, because the dataset is new).

Run every command from the repository root on the server. The paths below are
examples; replace the `<...>` run folders with your own.

## 1. Split the flat folder into sites

```bash
python tools/average_site_groups.py \
  --source /data/20261002_162547 \
  --output /data/20261002_162547_sites
```

Only images directly inside `--source` are read; subfolders are ignored. The
image count must be a multiple of `--frames-per-site` (default 8), otherwise
nothing is written. The output folder must not exist yet.

```text
/data/20261002_162547_sites/
  sites/site001/   the 8 original files, byte-identical copies   <- training
  sites/site002/ ...
  average2/        4 lossless PNGs per site, no registration      <- viewing
  average4/        2 per site
  average8/        1 per site
```

## 2. Prepare the training dataset

```bash
CUDA_VISIBLE_DEVICES=0 python -m edge_denoise prepare-real \
  --source-dir /data/20261002_162547_sites/sites \
  --out /data/20261002_prep_data/train_align_none \
  --image-size 512 --white-level 255 --align none --device cuda
```

Sites are assigned to train/val/test automatically (10% val, 10% test) and the
split is printed. `--split-file splits.json` sets it explicitly.

## 3. Continue ft_consist

Use the `config.yml` saved in your ft_consist run folder, so the recipe is
exactly the one that run used:

```bash
DATA=/data/20261002_prep_data/train_align_none
FT=runs/edge_denoise/<your_ft_consist_run>
CUDA_VISIBLE_DEVICES=0 python -m edge_denoise train --config "$FT/config.yml" \
  --dataset-dir "$DATA" --init-checkpoint "$FT/ckpt_latest.pt" \
  --run-dir runs/edge_denoise/261008_ft_consist_newdata \
  --max-steps 10000 --device cuda:0 --cpu-threads 2
```

ft_consist needs 3 frames per site (input, target, consistency frame), so 8 is
enough.

## 4. Continue the burst model

`sem_real_burst_t16.yml` needs at least 17 frames per site, so use
`edge_denoise/configs/sem_real_burst_ft8.yml`: the same backbone, objective and
effective batch, with 8 frames per burst and frame counts 1–7. It can run on a
second GPU at the same time:

```bash
BURST=runs/edge_denoise/<your_burst_t16_run>
CUDA_VISIBLE_DEVICES=1 python -m edge_denoise train \
  --config edge_denoise/configs/sem_real_burst_ft8.yml \
  --dataset-dir "$DATA" --init-checkpoint "$BURST/ckpt_latest.pt" \
  --run-dir runs/edge_denoise/261008_burst_ft8_newdata \
  --device cuda:0 --cpu-threads 2
```

`CUDA_VISIBLE_DEVICES=1` exposes only that GPU, so inside the process it is
`cuda:0`.

The frame count reaches the network only as its timestep input, so the T = 16
weights load unchanged. Counts 8–16 get no further training on this data and
may get worse: compare `val/loss_m01` and `val/loss_m16` against the original
run.

## Checks and resuming

- First run steps 3 and 4 with `--max-steps 10` and a throwaway `--run-dir`.
  The log must show `warm start: weights from ...` and finite losses.
- If a run stops, rerun the same command with `--resume` added.
- Checkpoints are saved every 1000 steps in the `--run-dir` folders.
  `tensorboard --logdir runs/edge_denoise` shows the losses.
