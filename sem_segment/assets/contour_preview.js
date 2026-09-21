/* Position within each saved source is distinct from its acquisition ID. */
(() => {
  "use strict";
  const data = window.CONTOUR_PREVIEW;
  const $ = id => document.getElementById(id);
  const labels = {raw: "Raw (raw)", average8: "Average of 8 (average8)", average128: "Average of 128 (average128)"};
  const site = () => data.sites[Number($("site").value)];
  const frames = () => site().series[$("source").value]?.frames || [];
  $("kind").textContent = data.synthetic ? "SYNTHETIC EXAMPLE · NO REAL ACQUISITIONS OR TRAINED MODEL OUTPUTS" : "SEGMENTATION PREVIEW";
  $("study").textContent = data.study;
  document.title = data.synthetic ? "Synthetic SEM contour baseline preview" : "SEM contour baseline preview";
  const crop = data.measurement_crop_y0_y1_x0_x1;
  $("crop-caption").textContent = crop ?
    `Measurement crop: rows [${crop[0]}, ${crop[1]}), columns [${crop[2]}, ${crop[3]}). Outlines are placed back on the full image.` :
    "Measurement crop: full image. Baseline paths at the measurement border stay open.";

  function render() {
    const rows = frames();
    const position = Math.max(0, Math.min(Number($("acquisition").value), rows.length - 1));
    const frame = rows[position];
    $("image-status").textContent = "";
    $("frame-note").textContent = frame?.note || "";
    $("frame-label").textContent = !frame ? "No saved images in this source" :
      `Saved image ${position + 1} of ${rows.length} · ` +
      (frame.first_acquisition != null && frame.last_acquisition != null ?
        `Acquisitions ${frame.first_acquisition}–${frame.last_acquisition} (index ${frame.index})` :
        `Acquisition ${frame.index}`);
    ["original", "current", "baseline"].forEach(panel => {
      const image = $(panel + "-image"), link = $(panel + "-link");
      image.hidden = link.hidden = !frame;
      if (frame) {
        image.src = panel === "original" ? frame.display : frame[panel];
        link.href = frame[panel];
      } else {
        image.removeAttribute("src"); link.removeAttribute("href");
      }
    });
    $("current-status").textContent = frame?.current_status || "";
    const settings = data.settings;
    $("baseline-caption").textContent = frame ?
      `Otsu threshold ${frame.threshold_dn.toFixed(3)} DN · ${settings.polarity} foreground · σ ${settings.sigma_px} px · minimum area ${settings.min_area_px} px. ${frame.retained_count} retained regions.` : "";
  }

  function selectSource() {
    $("acquisition").max = Math.max(0, frames().length - 1);
    $("acquisition").value = 0;
    $("acquisition").disabled = frames().length < 2;
    render();
  }

  function selectSite() {
    $("source").replaceChildren(...Object.keys(site().series).map(name => new Option(labels[name] || name, name)));
    selectSource();
  }

  ["original", "current", "baseline"].forEach(panel => {
    $(panel + "-image").addEventListener("error", () => {
      $("image-status").textContent = "An image could not be loaded. Keep the preview and its images folder together.";
    });
  });
  $("site").replaceChildren(...data.sites.map((entry, index) => new Option(entry.name, String(index))));
  $("site").addEventListener("change", selectSite);
  $("source").addEventListener("change", selectSource);
  $("acquisition").addEventListener("input", render);
  selectSite();
})();
