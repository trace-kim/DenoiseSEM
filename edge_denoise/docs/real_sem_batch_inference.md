# Denoise real SEM images and analyze the PNG outputs

Use this workflow for uint8 images and trained `edge_denoise` checkpoints
(ordinary N2N, registration-aligned N2N, `ft_avgfull_consist`, etc.).
Run commands in **bash on the remote server**, from the repository root inside
your Python environment/container. No `prepare-real` step is needed.

Analyze repeated acquisitions of **one site at a time**, with at least eight
frames by default. Keep filenames in acquisition order. Use separate output
folders for each site and checkpoint.

## 1. Set paths and run inference

`SRC` should contain only the raw frames for one site, directly in that folder.
Give every frame a unique filename stem (do not mix `frame_001.jpg` and
`frame_001.png`). If inference is already complete, set the paths and skip
the loop.

```bash
CKPT="runs/edge_denoise/<run_name>/ckpt_latest.pt"
SRC="data/SEM-test/site_01"
OUT="output/<run_name>/site_01-infer"
PNG="output/<run_name>/site_01-png"
RAW_REPORT="output/<run_name>/noise-raw"
PNG_REPORT="output/<run_name>/noise-png"

mkdir -p "$OUT"
: > "$OUT/failed.txt"
find "$SRC" -maxdepth 1 -type f \( -iname '*.png' -o -iname '*.tif' \
    -o -iname '*.tiff' -o -iname '*.bmp' -o -iname '*.jpg' -o -iname '*.jpeg' \) -print0 \
| sort -z \
| while IFS= read -r -d '' f; do
    if ! CUDA_VISIBLE_DEVICES=0 python -m edge_denoise denoise \
        --checkpoint "$CKPT" --input "$f" --out "$OUT" \
        --tile-batch 4 --device cuda; then
        echo "$f" >> "$OUT/failed.txt"
    fi
  done
cat "$OUT/failed.txt"   # Empty means no inference failures were recorded.
```

Resolve failures before comparing reports. On GPU memory errors, lower
`--tile-batch`. Leave stride and margin at their defaults.

Each frame produces:

| File | Contents |
|---|---|
| `*_input.png` | Input preview, uint8 |
| `*_denoised.png` | Denoised preview, uint8 |
| `*_denoised.tif` | Denoised values, float32 in [0, 1] |

## 2. Make PNGs in the original intensity units

Normalization uses **fixed black/white levels**, not each frame's minimum and
maximum. With levels 0 and 255, an input range of 60–180 stays 60–180 after
scaling back; it is not stretched to fill the range. A decreasing brightness
trend is preserved by this scaling, although the model may alter it.

**If the checkpoint uses black = 0 and white = 255, the existing denoised
PNGs already have the correct scale.** Copy only those files into a new folder:

```bash
mkdir "$PNG" && cp "$OUT"/*_denoised.png "$PNG"/
```

**If the levels differ or you are unsure, use this instead of the copy step.**
It reads the levels from the same checkpoint used for inference, restores the
original intensity units from the TIFFs, and rounds to uint8. It requires no
original images. This conversion is specifically for originally uint8 data.

```bash
python - "$CKPT" "$OUT" "$PNG" <<'PY'
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from edge_denoise.train import load_checkpoint

checkpoint, source, destination = map(Path, sys.argv[1:])
data = load_checkpoint(checkpoint, map_location="cpu")["config"]["data"]
black, white = data.get("black_level"), data.get("white_level")
if black is None and white is None:
    black, white = 0.0, 255.0  # Legacy uint8 normalization.
if black is None or white is None or not np.isfinite([black, white]).all() or white <= black:
    raise ValueError("Checkpoint must have valid fixed normalization levels")
files = sorted(source.glob("*_denoised.tif"))
if not files:
    raise ValueError(f"No denoised TIFFs in {source}")
destination.mkdir(parents=True, exist_ok=False)  # Use a new folder.
for path in files:
    with Image.open(path) as image:
        normalized = np.asarray(image, dtype=np.float32)
    counts = normalized * (white - black) + black
    pixels = np.rint(np.clip(counts, 0, 255)).astype(np.uint8)
    Image.fromarray(pixels).save(destination / f"{path.stem}.png")
print(f"Wrote {len(files)} PNGs; black={black}, white={white}")
PY
```

Do not use per-image auto-contrast or min/max scaling: that would change the
brightness trend. Inverse scaling cannot recover clipped values or undo
brightness changes caused by the model. Keep the TIFFs for future analysis.

## 3. Analyze raw frames and denoised PNGs

```bash
python -m pip install -e ".[analysis]"
python -m sem_noise analyze --input "$SRC" --output "$RAW_REPORT"
python -m sem_noise analyze --input "$PNG" --output "$PNG_REPORT"
```

Use new report directories on reruns. Open `index.html` in each report folder.
Check that both reports contain the same acquisitions. **Do not analyze
`$OUT` directly:** it mixes inputs, previews, and TIFFs.

Compare:

- **Brightness over time:** did the denoiser preserve the decreasing intensity
  trend, flatten it, or introduce an offset?
- **Temporal sigma and adjacent-frame differences:** how much variation remains
  between repeated denoised frames?
- **Drift:** do estimated movements agree? Registration is estimated separately
  for each report, so accepted frames and comparison regions can differ.

These PNG results measure the **final uint8 output**, including rounding.
Rounding can hide or exaggerate small fluctuations; lower noise alone does
not prove that edges or absolute intensity are accurate. For precise residual
noise measurements, use float32 TIFF copies converted to the same original
units (`normalized * (white - black) + black`), without rounding.
`sem_noise` does not automatically rescale normalized TIFFs to match raw data.

Treat mean–variance fits on denoised data as properties of the denoiser output,
not detector calibration. A brightness trend or temporal correlation also
affects noise estimates and averaging/Allan curves.

Pixel size and frame interval are optional. For multiple sites, acquisition
order manifests, regions of interest, and other options, see the
[sem_noise guide](../../sem_noise/README.md).
