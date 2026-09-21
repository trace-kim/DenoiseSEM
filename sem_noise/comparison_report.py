"""Static comparison reports and TensorBoard export; no training imports.

Consumes the coordinator's plain results and saved images. Both outputs share
one scalar table and the exact same rendered PNG figures.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
from PIL import Image


def comparison_metrics(record: dict) -> list[dict]:
    metrics = []
    for site in record["sites"]:
        for name, series in site["series"].items():
            frames = series["frames"]
            ranges = [f for f in frames if "minimum_dn" in f]
            values = {"frames": len(frames), "clipped_images": sum(f.get("clipped", False) for f in frames),
                      "registration_failures": sum(f["registration_status"] == "failed" for f in frames),
                      "unmatched_regions": sum(f.get("unmatched_regions", 0) for f in frames),
                      "unregistered_temporal_sigma_dn": series["noise"].get("unregistered_temporal_sigma_dn")}
            for mode, measures in series["noise"].get("modes", {}).items():
                for key, value in measures.items():
                    if isinstance(value, (float, int)) and np.isfinite(value):
                        values[f"noise/{mode}/{key}"] = value
            for stage, seconds in series.get("segmentation_stage_totals_s", {}).items():
                if frames:
                    values[f"time/{stage}_s_per_image"] = seconds / len(frames)
            for key, value in series["noise"].get("registration", {}).items():
                if isinstance(value, (float, int)) and np.isfinite(value):
                    values[f"noise/registration/{key}"] = value
            if name in record["models"]:
                values["output_registration_failures"] = sum(f["output_registration_status"] == "failed" for f in frames)
                differences = np.asarray([[f["output_minus_raw_dy_px"], f["output_minus_raw_dx_px"]]
                                          for f in frames if f["output_minus_raw_dy_px"] is not None]).reshape(-1, 2)
                values["output_vs_raw_registration_count"] = len(differences)
                values["output_vs_raw_translation_rms_px"] = float(np.sqrt(np.mean(np.sum(differences**2, axis=1)))) if len(differences) else None
            if ranges:
                count = sum(f["pixels"] for f in ranges)
                values.update(minimum_dn=min(f["minimum_dn"] for f in ranges),
                              maximum_dn=max(f["maximum_dn"] for f in ranges),
                              below_zero=sum(f["below_zero"] for f in ranges),
                              above_255=sum(f["above_255"] for f in ranges),
                              out_of_range_fraction=sum(f["below_zero"] + f["above_255"] for f in ranges) / count)
            for summary in site["repeatability"]:
                if summary["series"] == name:
                    values.update({f"{summary['method']}/{key}": summary[key] for key in
                                   ("median_cd_std", "median_cd_3sigma", "common_hole_count",
                                    "contributing_observations", "valid_count", "failed_count")})
            arm = record["models"].get(name, {}).get("arm", {})
            metrics.append({"site": site["name"], "series": name, "step": series["step"],
                            "refinement_device": series.get("refinement_backend", {}).get("device", "cpu"),
                            "training_registration": arm.get("registration"),
                            "training_brightness": arm.get("brightness"), "values": values})
    return metrics


def comparison_arms(record: dict) -> list[dict]:
    """One experiment identity table shared by HTML, exports and TensorBoard."""
    rows = []
    for name, model in record["models"].items():
        arm = model["arm"]
        training = model["config"]["training"]
        rows.append({"arm": name, "registration": arm["registration"], "brightness": arm["brightness"],
                     "step": model["step"], "settings_source": arm["settings_source"], "ema": model["ema"],
                     "checkpoint": model["checkpoint"], "checkpoint_sha256": model["sha256"],
                     "raw_content_split_sha256": arm["raw_content_split_sha256"],
                     "prepared_manifest_sha256": arm["prepared_manifest_sha256"],
                     "black_level": model["black_level"], "white_level": model["white_level"],
                     "log_every": training["log_every"], "val_every": training["val_every"],
                     "batch_size_per_rank": training["batch_size"], "accumulation_steps": training["accumulation_steps"]})
    return rows


def _table(rows: list[dict], columns: list[str]) -> str:
    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            # Keep the shortest round-trip representation: formatting a tiny
            # excursion as exactly "255" would contradict the range warning.
            return repr(value)
        return str(value)
    return ("<table><thead><tr>" + "".join(f"<th>{escape(k)}</th>" for k in columns) + "</tr></thead><tbody>"
            + "".join("<tr>" + "".join(f"<td>{escape(cell(r.get(k)))}</td>" for k in columns) + "</tr>" for r in rows)
            + "</tbody></table>")


def _warning(record: dict, rows: list[dict]) -> str:
    if not rows:
        return ""
    return ('<div class="warning"><strong>' + escape(record["range_warning"]) + "</strong>"
            + _table(rows, ["site", "model", "filename", "minimum_dn", "maximum_dn", "below_zero", "above_255",
                            "below_percent", "above_percent"]) + "</div>")


def render_comparison(root: Path, record: dict) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from .pipeline import write_csv, write_json

    root = Path(root)
    record["metrics"] = comparison_metrics(record)
    record["arms"] = comparison_arms(record)
    record["artifacts"] = []
    write_json(root / "metrics.json", record["metrics"])
    write_csv(root / "metrics.csv", [{k: v for k, v in r.items() if k != "values"} | r["values"]
                                     for r in record["metrics"]])
    write_csv(root / "arms.csv", record["arms"])
    write_json(root / "arms.json", record["arms"])
    clipped = [r for r in record["prediction_ranges"] if r.get("clipped")]
    body = ["<h1>Real N2N: registration and brightness comparison</h1>", _warning(record, clipped),
            f"<p>Status: {escape(record['status'])}. Intensities: DN (0–255). Dimensions: {record['unit']}.</p>",
            "<p>128 raw acquisitions and 128 individual outputs per model; 16 nonoverlapping eight-frame means. "
            "Means use raw pixels without registration or brightness correction. The first eight-frame mean "
            "defines correspondence, not ground truth. All measurements start from saved uint8 images; "
            "contours retain subpixel coordinates. No accuracy, PSNR or SSIM is calculated.</p>",
            "<p>Precision is the median within-hole sample SD over holes with at least two valid observations "
            "in every series, separately for each contour method. No temporal detrending is applied. "
            "Counts and measurement failures accompany every estimate. Noise reports may include separately "
            "labelled registration diagnostics; metrology never warps pixels.</p>",
            '<p><a href="comparison.json">Complete JSON</a> · <a href="metrics.json">Metrics JSON</a> · '
            '<a href="metrics.csv">Metrics CSV</a> · <a href="prediction_ranges.csv">Prediction range CSV</a> · '
            '<a href="arms.csv">Training arms CSV</a> · <a href="arms.json">Training arms JSON</a></p>',
            "<h2>Training treatments</h2>",
            "<p>These are registration/brightness variants of the same real-data Noise2Noise objective. "
            "The table describes training target preparation. Every checkpoint receives the same native test acquisitions; "
            "no arm-specific registration or brightness correction is applied at inference. Legacy preparation settings "
            "are read from the original manifest, rather than inferred from the absence of inline matching.</p>",
            _table(record["arms"], ["arm", "registration", "brightness", "step", "settings_source", "ema", "log_every", "val_every"]),
            "<p>Loss plateaus cannot rank denoising or CD accuracy across treatments: target resampling, brightness scaling "
            "and valid support alter the loss being measured. Compare these saved outputs with common analysis settings. "
            "Training logging already averages windows; TensorBoard smoothing adds a separate display filter. "
            "Checkpoint steps are reported explicitly; select checkpoints on validation sites before evaluating test sites.</p>"]
    body.extend(f'<p class="warning">{escape(w)}</p>' for w in record.get("warnings", []))

    def figure(fig, relative: str, site: str, label: str, series: str | None = None) -> str:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        record["artifacts"].append({"path": relative, "site": site, "label": label, "series": series})
        return f'<figure><a href="{relative}"><img src="{relative}" loading="lazy"></a><figcaption>{escape(label)}</figcaption></figure>'

    def pixels(path: str) -> np.ndarray:
        with Image.open(root / path) as image:
            return np.asarray(image).copy()[..., 0]

    crop = record["segmentation_settings"]["input"]["crop"]
    offset = np.array([crop[0], crop[2]]) if crop else np.zeros(2)
    for site in record["sites"]:
        site_name = site["name"]
        body.append(f"<h2>{escape(site_name)}</h2>")
        body.extend(f'<p class="warning">{escape(w)}</p>' for w in site["warnings"])
        body.append(f"<p>Matching gate: {site['match_gate_px']:.5g} px. Translations measured from raw acquisitions "
                    "are reused for model correspondence. Output translations are independently measured for the diagnostics "
                    "below, using the same first eight-frame average. They never replace the raw correspondence shifts. "
                    "Output-minus-input drift is a geometric diagnostic, not reference-based accuracy. "
                    "Registration failures yield missing values, not zero drift.</p>")
        body.append(_table([{**r, "clipped_images": sum(f.get("clipped", False) for f in site["series"][r["series"]]["frames"]),
                             "frames": len(site["series"][r["series"]]["frames"])} for r in site["repeatability"]],
                           ["series", "method", "frames", "clipped_images", "common_hole_count", "contributing_observations",
                            "median_cd_std", "median_cd_3sigma", "valid_count", "failed_count"]))
        body.append("<p>" + " · ".join(f'<a href="{site_name}/{name}">{name}</a>' for name in
                    ("observations.csv", "per_hole.csv", "repeatability.csv", "frames.csv", "contours.json")) + "</p>")
        fig, axes = plt.subplots(3, 1, figsize=(12, 11), layout="constrained")
        for name, series in site["series"].items():
            frames = series["frames"]
            x = [f["timestamp_s"] if f["timestamp_s"] is not None else f["order"] for f in frames]
            axes[0].plot(x, [f["mean_dn"] for f in frames], label=name)
            for component, style in (("dy_px", "-"), ("dx_px", "--")):
                axes[1].plot(x, [f.get(f"output_{component}", f[component]) for f in frames], style,
                             label=f"{name} {component}")
                if name in record["models"]:
                    axes[2].plot(x, [f[f"output_minus_raw_{component}"] for f in frames], style,
                                 label=f"{name} {component}")
        for ax in axes:
            ax.legend(fontsize=7, ncol=4)
            ax.grid(alpha=.2)
        axes[0].set_ylabel("Mean brightness (DN)")
        axes[1].set_ylabel("Independently measured drift (px)")
        axes[2].set_ylabel("Output minus raw drift (px)")
        axes[2].set_xlabel("Acquisition time (s)" if frames[0]["timestamp_s"] is not None else "Acquisition order / block center")
        body.append(figure(fig, f"{site_name}/figures/tracks.png", site_name, "Brightness and translation tracks; charging trends retained"))

        # Every block shows exactly its first raw observation and corresponding predictions.
        names = ["raw", *record["models"], "average8"]
        for block in range(16):
            fig, axes = plt.subplots(2, (len(names) + 1) // 2, figsize=(15, 7), squeeze=False, layout="constrained")
            links = []
            for ax, name in zip(axes.flat, names):
                frame = site["series"][name]["frames"][block if name == "average8" else block * 8]
                ax.imshow(pixels(frame["path"]), cmap="gray", vmin=0, vmax=255, interpolation="nearest")
                ax.set_title(name + (" [CLIPPED]" if frame.get("clipped") else ""), fontsize=9)
                links.append(f'<a href="{frame["path"]}">{escape(name)} native PNG</a>')
            for ax in axes.flat:
                ax.axis("off")
            body.append(figure(fig, f"{site_name}/figures/block_{block + 1:02d}.png", site_name,
                               f"Block {block + 1}: acquisitions {block * 8 + 1}–{block * 8 + 8}; models show acquisition {block * 8 + 1}"))
            body.append("<p>" + " · ".join(links) + "</p>")
        body.append(f'<h3>Full average — visual reference only</h3><a href="{site["full_average"]}">'
                    f'<img class="reference" src="{site["full_average"]}" loading="lazy"></a>'
                    '<p>Excluded from noise, registration, correspondence, metrology and repeatability calculations.</p>')

        for name, series in site["series"].items():
            body.append(f"<h3>{escape(site_name)} / {escape(name)}</h3>")
            body.append(_warning(record, [r for r in clipped if r["site"] == site_name and r["model"] == name]))
            metric = next(r for r in record["metrics"] if r["site"] == site_name and r["series"] == name)
            body.append(f"<p>Contour refinement device: {escape(metric['refinement_device'])}. "
                        "Classical masks and polygon geometry use the CPU. Timings include device transfers.</p>")
            body.append(_table([{"metric": k, "value": v} for k, v in metric["values"].items()], ["metric", "value"]))
            body.append(f'<p>Noise analysis: {escape(series["noise_status"])}. <a href="{series["noise_report"]}">Detailed noise report</a></p>')
            if series["noise"].get("error"):
                body.append(f'<p class="warning">{escape(series["noise"]["error"])}</p>')
            for warning in sorted({w for f in series["frames"] for w in f.get("segmentation_warnings", [])}):
                body.append(f'<p>{escape(warning)}</p>')
            observations = [r for r in site["observations"] if r["series"] == name]
            fig, axes = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
            for ax, method in zip(axes, ("coarse", "refined")):
                for hole in range(1, len(site["template_centroids"]) + 1):
                    points = [r for r in observations if r["hole"] == hole and r["method"] == method]
                    ax.plot([r["timestamp_s"] if r["timestamp_s"] is not None else r["order"] for r in points],
                            [r["cd"] if r["status"] == "valid" else np.nan for r in points], linewidth=.8)
                ax.set_ylabel(f"{method} ECD ({record['unit']})")
                ax.grid(alpha=.2)
            axes[-1].set_xlabel("Acquisition time (s)" if series["frames"][0]["timestamp_s"] is not None else "Acquisition order / block center")
            body.append(figure(fig, f"{site_name}/figures/{name}_cd.png", site_name,
                               f"{name}: one trace per hole; gaps are failed measurements", name))
            centers = np.asarray(site["template_centroids"][:4]).reshape(-1, 2) + offset
            if len(centers):
                fig, axes = plt.subplots(1, len(centers), figsize=(4 * len(centers), 4), squeeze=False, layout="constrained")
                first = series["frames"][0]
                image = pixels(first["path"])
                contours = [r for r in site["contours"] if r["series"] == name]
                for hole, (ax, center) in enumerate(zip(axes.flat, centers), 1):
                    # Display native pixels in template coordinates by changing extent,
                    # without resampling. Every overlaid contour gets its raw drift removed.
                    shift = np.array([first["dy_px"], first["dx_px"]], dtype=float)
                    if not np.isfinite(shift).all():
                        ax.set_title(f"Hole {hole}: registration failed")
                        continue
                    ax.imshow(image, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
                              extent=(-.5-shift[1], image.shape[1]-.5-shift[1], image.shape[0]-.5-shift[0], -.5-shift[0]))
                    for contour in contours:
                        if contour["hole"] != hole:
                            continue
                        for method, color in (("coarse", "cyan"), ("refined", "orange")):
                            points = np.asarray(contour[method]).reshape(-1, 2) + offset - contour["shift_yx"]
                            if len(points):
                                points = np.vstack((points, points[0]))
                            ax.plot(points[:, 1], points[:, 0], color=color, alpha=.18, linewidth=.5)
                    radius = min(32, max(12, site["match_gate_px"]))
                    ax.set_xlim(center[1] - radius, center[1] + radius)
                    ax.set_ylim(center[0] + radius, center[0] - radius)
                    ax.set_title(f"Fixed hole {hole}")
                body.append(figure(fig, f"{site_name}/figures/{name}_contours.png", site_name,
                                   f"{name}: fixed crops; mask (cyan), gradient refined (orange), all matched observations", name))
    html = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>SEM comparison</title>'
            '<style>body{font:15px system-ui;margin:2em;color:#192532}table{border-collapse:collapse;display:block;overflow:auto}'
            'th,td{border:1px solid #ccc;padding:.4em;text-align:left}.warning{border:3px solid #ba2323;background:#fff0df;padding:1em;margin:1em 0}'
            'figure{margin:1em 0}img{max-width:100%;height:auto}.reference{max-width:700px}h2{border-top:2px solid #456;padding-top:1em}</style>'
            + "\n".join(body) + "</html>")
    destination = root / "index.html"
    destination.write_text(html, encoding="utf-8")
    return destination


def write_tensorboard(root: Path, record: dict, *, writer_factory=None) -> None:
    """Log precisely the report's scalars and figure PNGs at checkpoint steps."""
    if writer_factory is None:
        from torch.utils.tensorboard import SummaryWriter
        writer_factory = SummaryWriter
    writer = writer_factory(log_dir=str(root / "tensorboard_comparison"))
    try:
        if record.get("warnings"):
            writer.add_text("comparison/provenance_warnings", "\n\n".join(record["warnings"]), 0)
        for arm in record["arms"]:
            writer.add_text(f"comparison/{arm['arm']}/training_treatment", _table([arm], list(arm)), arm["step"])
        for row in record["metrics"]:
            prefix = f"{row['site']}/{row['series']}"
            writer.add_text(f"{prefix}/refinement_device", row.get("refinement_device", "cpu"), row["step"])
            for key, value in row["values"].items():
                if value is not None and np.isfinite(value):
                    writer.add_scalar(f"{prefix}/{key}", value, row["step"])
            flagged = [r for r in record["prediction_ranges"] if r["site"] == row["site"] and r["model"] == row["series"] and r.get("clipped")]
            if flagged:
                writer.add_text(f"{prefix}/range_warning", record["range_warning"] + "\n\n" +
                                _table(flagged, ["filename", "minimum_dn", "maximum_dn", "below_zero", "above_255"]), row["step"])
        for artifact in record["artifacts"]:
            with Image.open(root / artifact["path"]) as image:
                array = np.asarray(image.convert("RGB")).copy()
            series = [artifact["series"]] if artifact["series"] else ["raw", *record["models"]]
            for name in series:
                step = record["models"].get(name, {}).get("step", 0)
                writer.add_image(f"{artifact['site']}/{name}/{Path(artifact['path']).stem}", array, step, dataformats="HWC")
        for site in record["sites"]:
            with Image.open(root / site["full_average"]) as image:
                reference = np.asarray(image.convert("RGB")).copy()
            for name, model in record["models"].items():
                writer.add_image(f"{site['name']}/{name}/full_average_visual_reference_only", reference,
                                 model["step"], dataformats="HWC")
        writer.add_text("comparison/protocol", "Saved uint8 measurements. Full average: visual reference only. "
                        "Eight-frame raw nonoverlapping means; no detrending. See comparison.json for settings and counts.", 0)
    finally:
        writer.close()
