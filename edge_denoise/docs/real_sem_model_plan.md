# Real-SEM model work agreed on 2026-09-22

Finish the multi-model comparison workflow first. The subsequent training work
is explicitly requested for the next phase; this document preserves it across
sessions. Do not assume that the synthetic benchmark's winning model is best
on the real acquisitions.

## Models and order

1. `ft_noisy`: warm-start a compatible real N2N checkpoint with image and Sobel
   fidelity against a different noisy acquisition.
2. `ft_consist`: the same continuation with cross-acquisition consistency.
3. Gradient-only reconstruction: implement and validate the prepared-real data
   path, registration and brightness semantics, and gradient-to-image recovery.
4. Hybrid image/Sobel inputs: implement and validate the prepared-real path and
   an appropriate initialization strategy for its different input channels.
5. A contour-position **or** gradient-only consistency loss: choose the simplest
   justified method after checking its loss, registration and validity support;
   implement it through the actual training entry point.

Missing methods must be implemented appropriately, not merely accepted by a
configuration validator. A comparison accepting a checkpoint does not establish
that its training pipeline is implemented. Preserve the working N2N controls.
All new model recipes must default to **affine registration and percentile
brightness correction**, while inputs stay native. Existing N2N experiments
retain their actual recorded settings. Measurements always use decoded saved
uint8 output images, the agreed detector, and no output correction.

## Unattended training deliverable

The remote server will soon train without supervision for several days. Before
that period (the user's end-of-day requirement), provide a ready-to-run script
or terminal command to run all agreed pipelines **in order**. Include explicit
dataset, N2N checkpoint and run-root arguments, separate dated run folders,
per-run logs, failure reporting, and safe restart/resume behavior. Do not
silently skip unfinished or failed training. Validate locally on synthetic
fixtures; remote artifacts cannot be copied to this PC.

User clarification on 2026-09-22: **continue to the next pipeline after a
pipeline fails**. Record the failure, attempt every remaining pipeline, and
return an unsuccessful suite status until every required pipeline completes.
An explicit user/scheduler stop should stop the suite, preserving restart state.
The implementation checklist and remote handoff are maintained in
[real_sem_next_phase.md](real_sem_next_phase.md).

Use one allocated H100 by default and preserve scheduler GPU visibility. Check
the full CPU/GPU/data-loading path before setting unattended budgets. Record
wall time and compare models on held-out sites using the multi-model report.
Do not label a launcher complete while any required training method is missing.
