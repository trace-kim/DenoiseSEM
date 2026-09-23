# Real-SEM next phase: implementation and remote handoff

Agreed on 2026-09-22. Source requirements: `AGENTS.md` and
`real_sem_model_plan.md`. This file persists the implementation plan and status.

## Fixed decisions

- Run `ft_noisy`, `ft_consist`, `grad`, `hybrid`, then `ft_grad_consist`.
- The image fine-tunes independently initialize from the same compatible real
  N2N teacher. Gradient reconstruction starts fresh. Hybrid copies the teacher's
  image input weights and zero-initializes its two extra input-channel weights.
- Choose gradient-only consistency for the fifth arm: map the second image
  prediction into the first acquisition's coordinates/brightness, then compare
  Sobel fields on valid support. Keep image/Sobel fidelity anchors.
- New recipes use affine registration and percentile target matching, native
  inputs, the same verified prepared `align none` dataset and fixed site splits.
  Preserve existing N2N controls and their actual recorded treatments.
- Blank/noise-only acquisitions skip geometry automatically and stay in training.
  Affine uses the existing block-contrast gate and first usable reference;
  unmeasurable/failed ECC fits skip by default without changing the next seed.
  Flat percentile distributions use native pair brightness. Per-frame reasons
  survive shared-cache reuse and checkpoint resume. See
  `real_sem_inline_matching.md` for the exact fallback policy.
- Measurements use decoded saved uint8 outputs and the agreed detector. Output
  brightness and geometry are never corrected during comparison.
- Use one allocated GPU and preserve scheduler visibility. Hardware throughput
  and CPU/GPU equivalence must be measured on the remote server.
- A failed pipeline is recorded and the next pipeline is attempted. Failed or
  unfinished work cannot count as completed. Explicit interruption stops the
  suite. Restart validates identities and resumes saved training state.

## Implementation checklist

- [x] Verify heterogeneous multi-model comparison on synthetic fixtures.
- [x] Implement complete prepared-real gradient/hybrid support, initialization,
      Sobel consistency, validation and saved-image inference tests.
- [x] Add all five affine/percentile recipes and CLI controls.
- [x] Add explicit verified registration-measurement reuse and stage timings.
- [x] Add ordered launcher, preflight, logs, failure continuation and safe resume.
- [x] Test actual training/resume/inference plus launcher failure/error paths.
- [x] Supply remote pilot, full-run, resume, equivalence and report commands.
- [x] Update older floating-output measurement guidance; record local validation.

## Acceptance criteria

Every method must train through `python -m edge_denoise train`, checkpoint,
resume, infer and export measurable uint8 images on synthetic fixtures. A
successful suite requires all requested step budgets to be reached with matching
dataset/configuration/teacher identities, not merely a zero process exit code.
The launcher must attempt later pipelines after an individual failure. Final
reports compare N2N and all five methods on held-out sites, including failures,
brightness changes, ECD/CD variation and mean shifts, with runtime and training
budgets recorded. No real server data, logs or screenshots are needed locally.

## Implemented methods

| Order / recipe | Representation | Image / Sobel / consistency weights | Initialization | Default steps |
|---|---|---|---|---|
| `sem_real_ft_noisy.yml` | image | 1 / 4 / 0 | N2N EMA | 10,000 |
| `sem_real_ft_consist.yml` | image | 1 / 4 / 1, image consistency | same N2N EMA | 10,000 |
| `sem_real_grad.yml` | gradient | 0 / 1 / 0 | fresh | 100,000 |
| `sem_real_hybrid.yml` | image + Sobel inputs | 1 / 4 / 0 | N2N EMA, explicit input-channel expansion | 10,000 |
| `sem_real_ft_grad_consist.yml` | image | 1 / 4 / 1, Sobel consistency | same N2N EMA | 10,000 |

These are starting budgets, not measured H100 duration estimates or a claim
that equal steps give equal compute. The suite copies the teacher's actual
backbone and crop size into every resolved recipe. All targets are a different
noisy acquisition; consistency uses a third distinct acquisition. The earlier
leave-one-out-mean recipes remain separate experiments.

For `ft_grad_consist`, both network inputs stay native. Map the second image
prediction using geometry and full-frame percentile coefficients measured on
raw acquisitions, then take Sobel derivatives. The consistency mask excludes
invalid neighbours in addition to interpolation/network margins. Image/Sobel
fidelity prevents the consistency term from being the sole objective. Its
initial weight is 1; validation loss components expose its scale.

Gradient reconstruction uses the existing Sobel inverse, with DC supplied by
the native input crop mean. The inverse's null space and low-frequency error
remain real limitations. FFT operators are cached with a four-entry bound.
Real gradient-output models currently require zero consistency weight; the
fifth arm uses an image-output model and does not pretend gradient channels
can receive an image brightness offset or an ordinary scalar affine warp.

## Remote preflight and short timing pilot

Run from the repository root in the server environment, inside the GPU
allocation. Change these two paths to the existing prepared dataset and teacher;
no YAML editing is needed. The dataset must preserve native uint8 acquisitions,
be prepared with `--align none`, and have the same original content/site splits
and normalization as the teacher. Registration history may differ and is recorded.

```bash
python -m pip install -e ".[analysis]"
DATASET=data/SEM-real-none
N2N=runs/edge_denoise/260921_real_n2n_affine_percentile/ckpt_latest.pt
RUN_DATE=260922

python tools/train_real_sem_suite.py \
  --dataset-dir "$DATASET" --n2n-checkpoint "$N2N" \
  --run-root runs/edge_denoise --date "$RUN_DATE" --dry-run

python tools/train_real_sem_suite.py \
  --dataset-dir "$DATASET" --n2n-checkpoint "$N2N" \
  --run-root runs/edge_denoise/pilots --date "$RUN_DATE" \
  --max-steps 100 --batch-size 4 --accumulation-steps 4 \
  --device cuda:0 --cpu-threads 2 --profile
```

Dry-run hashes/verifies input content and prints the resolved recipes; it writes
nothing and does not start training. `cuda:0` is the first **visible** allocated
GPU. Neither the launcher nor trainer rewrites `CUDA_VISIBLE_DEVICES`. The other
allocated GPUs remain available. PyTorch and OpenCV CPU threads are bounded;
matching remains the existing CPU ECC/OpenCV estimator and sampler. Forward,
backward, Sobel operations, differentiable prediction warps and reconstruction
run on the selected GPU. No CPU-estimator speedup is claimed.

Each run writes `timings.json` with dataset verification, matching/startup,
sampling, transfers, forward/loss/backward, optimizer/EMA, validation and
checkpoint timings, plus CUDA peak allocated memory. `--profile` synchronizes
CUDA around timed stages, adding overhead; use it for pilots. Nested stage
times overlap. Full-frame inference and saved-image analysis/I/O timings are
in the comparison report's `timings.json`. Choose production budgets after
checking the complete workflow, including validation and reconstruction.

Registration defaults to strict failure reporting. If an explicit skip policy
is appropriate, pass `--registration-failure skip` to the pilot and production
commands. This retains failed frames with unmeasured geometry under the existing
pair fallback rules; it never hides brightness, I/O or training failures.

## Unattended training and restart

After the pilot, run this command in a persistent terminal or scheduler job:

```bash
python tools/train_real_sem_suite.py \
  --dataset-dir "$DATASET" --n2n-checkpoint "$N2N" \
  --run-root runs/edge_denoise --date "$RUN_DATE" \
  --device cuda:0 --cpu-threads 2
```

All five pipelines are attempted in order even if an earlier pipeline fails.
The four continuations each start independently from the teacher. Recipes use
FP32, batch 4 and accumulation 4 by default (effective batch 16 on one worker).
Use `--batch-size`, `--accumulation-steps`, `--lr`, `--precision`,
`--checkpoint-every`, and `--steps grad=100000` to override execution/budgets;
`--max-steps N` overrides every budget for a pilot. Keep effective batch and
initialization/budget differences explicit in interpretation.

Generated layout:

```text
runs/edge_denoise/
  260922_real_suite/
    suite.json              resolved plan, content identities, attempts/status/times
    configs/*.yml           exact training CLI configurations
    real_matching.json      explicitly shared verified registration/percentile cache
    comparison.yml          N2N + all five methods; unchanged Otsu detector
  260922_real_ft_noisy_affine_percentile/
    attempt_001.log          separate stdout/stderr for this attempt
    config.yml
    ckpt_latest.pt
    provenance.json
    real_matching.json
    registration_report.html
    timings.json
    training_status.json
  ... other four dated model folders ...
```

To retry failed/unfinished pipelines and verify completed ones, repeat the exact
production command and add `--resume`:

```bash
python tools/train_real_sem_suite.py \
  --dataset-dir "$DATASET" --n2n-checkpoint "$N2N" \
  --run-root runs/edge_denoise --date "$RUN_DATE" \
  --device cuda:0 --cpu-threads 2 --resume
```

Resume restores model/optimizer/EMA/RNG/sampler state from a valid checkpoint.
If startup failed before any checkpoint, its generated artifacts are archived
under `failed_attempts/` and that pipeline starts again from its prescribed
initialization; prior logs remain. Existing unrelated runs are never claimed.
Changed plans, data, teachers or completed checkpoint hashes are rejected.
Choose a new date/run root for a different scientific recipe or budget. No
automatic retry loop repeatedly burns time on the same failed pipeline.

Exit codes: 0 only when all five reach their requested steps; 1 for an incomplete
suite (including individual failures); 130 for an explicit cooperative stop.
A zero child exit code or an existing directory cannot imply completion.
The manifest records errors and each attempt's wall time, including unsuccessful
attempts. A hard interruption with no recorded end time is explicitly marked
interrupted and counted in `attempts_without_wall_time`; its duration is not
invented. Logs contain tracebacks. An OS file lock prevents concurrent launches
of the same suite and releases automatically on process exit.

To stop the whole suite cooperatively:

```bash
touch "runs/edge_denoise/${RUN_DATE}_real_suite/stop"
```

Ctrl+C/SIGTERM also request a stop. Training can finish its current startup or
optimizer step before saving; a deliberate stop does not start another pipeline.
Explicit `--resume` clears the old suite stop marker. A checkpoint is saved every
1,000 steps by default, limiting work lost after an abrupt process/server failure.

### Retry a suite that failed during registration startup

After updating the server checkout, repeat the original suite command with
`--resume`, retaining its date, paths, budgets and other flags. For the default
full-run command above:

```bash
python tools/train_real_sem_suite.py \
  --dataset-dir "$DATASET" --n2n-checkpoint "$N2N" \
  --run-root runs/edge_denoise --date "$RUN_DATE" \
  --device cuda:0 --cpu-threads 2 --resume
```

The 2026-09-23 fix makes blank/low-contrast registration a recorded skip and
defaults affine ECC failures to skipping geometry. No new skip flag is needed.
Adding an override to an existing suite changes its plan identity, so keep the
original arguments. Failed startup artifacts/logs are retained; uncheckpointed
pipelines retry initialization and completed pipelines are verified. Review
each run's `registration_report.html`, `registration_frames.csv` and
`real_matching.json` on the server for the skipped frames and reasons.

## CPU/GPU checks and held-out reports

For a completed pilot or production suite, compare CPU/GPU inference entirely
on the server. The default is one centre crop per model to limit CPU cost:

```bash
python tools/check_real_sem_equivalence.py \
  --from-suite "runs/edge_denoise/pilots/${RUN_DATE}_real_suite/suite.json" \
  --site-dir /data/260904_raw_data/test/260904_0947-13 \
  --device cuda:0 --cpu-threads 2 \
  --output-dir "output/${RUN_DATE}_real_pilot_equivalence"
```

This saves both outputs, decodes their PNGs, and checks their pixel differences.
Defaults allow at most 1 DN difference affecting at most 1% of pixels; these are
explicit numerical tolerances, not proof of equal metrology. Use `--max-dn 0
--max-changed-fraction 0` for exact uint8 equivalence, or `--full-frame` to include
tiling/reconstruction seams at higher CPU cost. Nonfinite predictions fail before
export. `equivalence.json` contains timings, tolerances and per-model results.
No server artifacts need to be copied back here.

For model selection, use a raw site recorded in the prepared validation split:

```bash
VAL_SITE=/path/to/a/prepared-validation-sites-original-raw-folder
python tools/real_sem_compare.py \
  --config "runs/edge_denoise/${RUN_DATE}_real_suite/comparison.yml" \
  --split val --site-dir "$VAL_SITE" \
  --output-dir "output/${RUN_DATE}_real_models_validation"
```

The comparison expects the full 128-frame acquisition. Validation mode checks
every decoded acquisition against the checkpoint's recorded validation content.
The default test mode retains exclusion of **both** training and validation
content, while allowing new unseen test sites. Use the same detector/settings
for every model. After selecting settings, produce the final test report:

```bash
python tools/real_sem_compare.py \
  --config "runs/edge_denoise/${RUN_DATE}_real_suite/comparison.yml"

python tools/benchmark_sem_analysis.py \
  --from-comparison "output/${RUN_DATE}_real_models_comparison/comparison.json" \
  --device cuda:0 --frames-per-source 16 \
  --output-json "output/${RUN_DATE}_real_models_analysis_equivalence.json"
```

The second command times and checks the existing CPU/GPU measurement backends
on saved uint8 outputs, including image I/O. CUDA analysis uses the existing
CuPy installation described in `real_sem_comparison.md`; training uses PyTorch.
`--metrology-device cpu` explicitly selects CPU measurement when needed.

Review per-hole ECD mean/SD/3-sigma, usable/total counts and missing features,
brightness relative to the matching raw acquisition, clipping flags, and runtime.
The agreed Otsu detector supplies ECD; do not relabel it as a different CD detector
or infer accuracy from lower variation alone. Initialization, consistency domain,
steps and batch settings are exported with each model. Use `--contours-only` for
explicit contour remeasurement with other analysis reused, and `--render-only`
for presentation changes using existing measurements.

## Local validation status

Synthetic fixtures exercise actual affine/percentile matching, all five training
CLI paths, warm starts, resume, saved uint8 full-frame output, cache identity,
independent acquisitions and validity masks. Launcher tests mock subprocess
execution while exercising the real training CLI, plus failed/partial runs,
interruptions, unrelated directories and altered completed checkpoints.

Hardware performance, CUDA numerical equivalence and real-acquisition model
quality remain to be measured with the remote commands above.

Local CPU full-suite verification: `python -m pytest -q` passed **1,027 tests**,
with one optional browser test skipped because no browser executable was set.
An additional **93 focused regression tests passed** after GPU-RNG isolation,
full-frame export precision and restart-history changes; **8 final checks passed**
for stored-plan integrity and validation/test report labels. `git diff --check`
also passed. GPU selection/RNG and process behavior are mocked in tests; no H100
timing or real-data quality claim is made.

2026-09-23 patternless-frame regression fix: `python -m pytest -q` passed
**1,046 tests**, with the same optional browser test skipped. Coverage includes
saved uint8 noisy/constant acquisitions, all-blank train and validation sites,
first-usable affine references, failed-fit seed preservation, native brightness
fallback, all five training CLI paths after failed-suite restart, shared-cache
refresh/reuse and checkpoint restoration. Real H100 training remains a remote
check; no server artifacts were required for this local validation.
