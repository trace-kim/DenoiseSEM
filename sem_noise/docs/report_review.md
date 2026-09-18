# Report review

The report must explain acquisition stability and let a reader compare changes
to noisy target B while raw input A remains untouched. An acquisition trend,
a B-to-A correction, and noise about a repeat mean answer different questions.

## Fixed-reference brightness and blurred difference review

The cyclic pair means could not demonstrate acquisition-wide brightness
consistency: every B was matched to a different A. The report now starts with
raw, two-region gain/offset and percentile gain/offset means against the first
included raw acquisition, clearly identified by index. Both coefficient tracks
use that same reference. Each mean is measured from the actual unresampled
diagnostic image over every supplied pixel, including clipping bounds. No
extra mean normalization is used. Geometry affects two-region estimation only;
changing overlap, resampling or cropping cannot change the plotted pixel support.
Percentile estimation remains full-image. Excluded frames keep their raw means;
failed/excluded corrections leave gaps with independent reasons.

`acquisition_brightness.csv` and the JSON export contain all plotted values.
Three native NPZ examples retain the reference, raw frames and available
corrected copies. The graph appears in both the site and full pair reports,
while the original motion/raw-brightness tracks, histograms, scatter plots,
three pair examples and all residual/noise diagnostics remain available.

Interactive differences now represent images Gaussian-blurred at σ = 2 pixels
after each correction stage. Linearity allows the equivalent filtering of
signed differences. Full kernel support is required: an 8-pixel margin around
invalid pixels and borders is grey, and failed stages remain independent.
The viewer explicitly identifies its blur and its statistics as blurred;
static PNGs and residual tables remain unblurred. Colour entry, slider, full
range, pixel readout and PNG export are preserved.

Regression checks cover known gain/offset recovery, actual corrected-image
means, outliers outside geometric support that must prevent a flat track,
excluded frames, independent failures, and blur that suppresses noise while
retaining displaced edges without invalid-pixel leakage. The SEM suite and
real-pair training compatibility checks passed (120 tests, including the
offline browser controls). A complete 128-acquisition synthetic report was
also generated for workflow review. These checks validate implementation,
not physical estimator accuracy on the user's server dataset.

## Earlier report review

The review compared the current report, exported measurements and failure paths
with the reports before and after the registration rewrite (`5e42751` and
`6033065`). It found and addressed these problems:

| Finding | Correction |
|---|---|
| Acquisition brightness and x/y drift plots disappeared during the rewrite. | Restore them near the top, for all acquisitions, alongside affine centre drift and corner deformation. Use gaps for excluded/failed estimates. |
| Pair-correction tracks could be mistaken for acquisition evolution. | Keep them separate and explain the cyclic target change. Show both corrected means on the same comparison pixels. |
| The overview showed only the two-region gain range. | Include the percentile range in the site overview, summary JSON and CSV. |
| One failed two-region fit suppressed usable geometric and percentile comparisons. | Compute and report each stage independently. Failed stages stay grey; successful stages and their measurements remain available. |
| Intermediate image examples omitted the percentile-corrected target. | Show both corrected targets alongside the untouched input and geometric stages, using shared grayscale limits. |
| Display range choices were baked into coloured PNGs and a few presets. | Save unclipped signed difference values for every pair. Use a continuous slider and arbitrary positive DN entry in an offline viewer. Changing colour does not rerun or alter a fit. |
| Display changes were tested mainly as generated files rather than browser behaviour. | Add an executable offline browser regression for exact entry, slider, reset, recovery from saturation, invalid masks, failed stages, unchanged statistics, pixel readout and PNG export. |
| New browser code would be missing from source provenance. | Package the JavaScript asset and include its hash in provenance. |

The review also checked that raw histograms, individual pixel scatter plots,
intermediate images, both brightness estimators, all-pair tables, 4×4 residual
tables, noise distributions, mean/variance plots, temporal/spatial correlation,
Allan plots and early/late differences remain available. The main report keeps
three examples, with the full report separate. Early/late and legacy-fit
differences also link to the adjustable viewer. Removed tile estimators,
selection gates and SIFT registration have not been restored.

Neither brightness estimator was changed. Tests verify original fit values,
target-to-input direction, signed uint8 subtraction, full-image percentile
sampling and untouched input arrays. The viewer uses native float32 differences
for display, with float64 for values that would overflow or underflow;
analysis statistics retain their original precision. It does not
downsample or clip values when saving the data. Consequently interactive files
can be large: the five-panel 2048×2048 test produced a 98.7 MiB HTML viewer.
The report embeds only the three example viewers and links to the others.

Validation included the `sem_noise` suite and real-pair training compatibility
tests, an offline Chrome test, a complete 128-frame report with all local links
checked, a native 2048×2048 viewer, and a wheel build containing the browser
asset. Fixtures cover zero/sub-DN/8-bit/16-bit/large differences, outliers,
invalid pixels, excluded frames and independent method failures. These checks
establish implementation behaviour; they do not establish either estimator's
physical accuracy on the user's server dataset.

Run the browser check on Windows with an installed Chrome or Edge executable:

```powershell
$env:SEM_NOISE_TEST_BROWSER = 'C:\Program Files\Google\Chrome\Application\chrome.exe'
python -m pytest tests/sem_noise tests/edge_denoise/test_real_sem_experiment.py -q
```

On Linux, set the same environment variable to the browser executable. Without
it, only the browser-specific test is skipped; numerical/report tests still run.
