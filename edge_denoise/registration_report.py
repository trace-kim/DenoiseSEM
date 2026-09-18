"""Local per-frame registration audit; never remove a failed frame from training."""

from __future__ import annotations

from collections import Counter
import csv
from html import escape
import json
from pathlib import Path

from burst_diffusion.data import BurstCache


def write_registration_report(directory: Path, cache: BurstCache, measurements: dict, method: str) -> None:
    """Write all measured train/validation frames, with original relative filenames."""
    sites = {site["source_index"]: site for site in cache.real_metadata["sites"]}
    rows = []
    for split, sources in (("train", cache.train_sources), ("val", cache.val_sources)):
        for source in sources:
            site = sites[source.source_index]
            record = measurements[str(source.source_index)]
            diagnostics = record.get("diagnostics") or [
                {"status": "disabled" if method == "none" else "registered"} for _ in source.frames]
            for i, diagnostic in enumerate(diagnostics):
                status = diagnostic["status"]
                unavailable = status.startswith("skipped")
                reason = diagnostic.get("reason", "")
                if status == "skipped_low_contrast":
                    reason = "contrast below the configured minimum; geometry is unmeasured"
                row = {"site": site["name"], "source_index": source.source_index, "split": split,
                       "frame_index": i, "filename": site["frames"][i]["name"], "method": method,
                       "status": status, "reason": reason, "contrast": diagnostic.get("contrast", ""),
                       "retained_in_split": True, "training_eligible": split == "train",
                       "pair_geometry": "disabled if either frame is unavailable" if unavailable else
                                        ("disabled by configuration" if method == "none" else "enabled when both frames are available"),
                       "estimated_shift_yx": json.dumps(diagnostic["estimated_shift"]) if "estimated_shift" in diagnostic else "",
                       "estimated_uncertainty_yx": json.dumps(diagnostic["estimated_uncertainty"]) if "estimated_uncertainty" in diagnostic else ""}
                row.update({f"stored_m{r}{c}": record["matrices"][i][r][c] for r in range(2) for c in range(3)})
                rows.append(row)
    with (directory / "registration_frames.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = Counter(row["status"] for row in rows)
    failures = [row for row in rows if row["status"].startswith("skipped")]
    columns = ("site", "split", "frame_index", "filename", "status", "reason", "contrast", "training_eligible")

    def table(selected: list[dict]) -> str:
        headers = ''.join(f'<th>{escape(name.replace("_", " "))}</th>' for name in columns)
        content = ''.join('<tr>' + ''.join(f'<td>{escape(str(row[name]))}</td>' for name in columns) + '</tr>'
                          for row in selected)
        return f'<div class="scroll"><table><thead><tr>{headers}</tr></thead><tbody>{content}</tbody></table></div>'

    body = '<h1>Training registration report</h1>'
    body += f'<p>Method: {escape(method)}. {len(rows)} train/validation frames measured; {len(failures)} have unavailable registration. No frames were removed by registration.</p>'
    body += '<p>A failed or low-contrast frame remains in its original split. If either member of a sampled pair has unavailable geometry, both crops use the same native coordinates: no registration correction is applied. This also covers each leave-one-out target and the consistency pair. Brightness matching is independent.</p>'
    body += '<p>Frame index is zero-based within the prepared site. Filename is the original path relative to the raw dataset. Validation frames remain validation-only; test frames are not measured here. Reference status denotes a coordinate anchor, not a fitted zero-motion measurement. Stored identity fallback matrices are not successful estimates.</p>'
    body += '<p>Created after startup measurements (or checkpoint restoration), before the first optimizer update, and retained after training. This is registration eligibility, not a log of how often a frame was drawn.</p>'
    body += '<p><a href="registration_frames.csv">Every frame and stored matrix (CSV)</a> | <a href="real_matching.json">Complete measurements (JSON)</a></p>'
    body += '<p>' + '; '.join(f'{escape(status)}: {count}' for status, count in sorted(counts.items())) + '</p>'
    body += '<h2>Frames retained without registration</h2>' + (table(failures) if failures else '<p>None.</p>')
    body += '<details><summary>All measured frames</summary>' + table(rows) + '</details>'
    html = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>Training registration report</title><style>body{font:15px/1.5 system-ui,sans-serif;margin:24px;max-width:1400px}'
            'table{border-collapse:collapse}th,td{padding:7px;border:1px solid #ccc;text-align:left}'
            '.scroll{overflow-x:auto}summary{cursor:pointer}</style></head><body>' + body + '</body></html>')
    (directory / "registration_report.html").write_text(html, encoding="utf-8")
