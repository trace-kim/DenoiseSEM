/* Native-coordinate image inspection. No image normalization or registration. */
(() => {
  "use strict";
  const report = window.SEM_REPORT;
  window.SEM_CONTOURS = Object.create(null);
  const $ = id => document.getElementById(id);
  const ns = "http://www.w3.org/2000/svg";
  const letters = ["a", "b"], colors = ["#14778d", "#d47732"];
  const state = {site: 0, names: ["raw", ""], indices: [0, 0], acquisition: 1,
    active: 1, view: [], contours: [[], []], selected: null, revision: 0, playing: null};
  const pending = new Map(), cacheOrder = [];
  const site = () => report.sites[state.site];
  const series = p => site().series[state.names[p]];
  const frame = p => series(p).frames[state.indices[p]];
  const fmt = (v, digits = 3) => Number.isFinite(v) ? Number(v).toFixed(digits) : "Unavailable";
  const label = name => ({raw: "Raw input", average8: "Average of 8", average128: "Average of 128"}[name]
    || name.replaceAll("_", " + "));
  const svg = (tag, attrs = {}, text) => {
    const node = document.createElementNS(ns, tag);
    for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const option = (value, text) => new Option(text, value);
  const ringPath = points => points.length ? "M" + points.map(p => `${p[1]},${p[0]}`).join("L") + "Z" : "";
  const time = f => f.timestamp_s ?? f.order;
  const timeTitle = () => site().series.raw.frames[0].timestamp_s == null ? "Acquisition / block center" : "Acquisition time (s)";
  function frameLabel(p) {
    const f = frame(p), name = state.names[p];
    if (name === "average128") return "Reference · acquisitions 1–128";
    if (name === "average8") return `Block ${f.index}/16 · acquisitions ${f.first_acquisition}–${f.last_acquisition}`;
    return `Acquisition ${f.index}/${series(p).frames.length}`;
  }
  function atAcquisition(name, acquisition) {
    if (name === "average128") return 0;
    return Math.max(0, Math.min(site().series[name].frames.length - 1,
      name === "average8" ? Math.floor((acquisition - 1) / 8) : acquisition - 1));
  }
  function setFrame(p, index) {
    state.active = p;
    state.indices[p] = Math.max(0, Math.min(series(p).frames.length - 1, index));
    if (state.names[p] !== "average128") {
      state.acquisition = frame(p).first_acquisition ?? frame(p).index;
      if ($("linked").checked) state.indices[1 - p] = atAcquisition(state.names[1 - p], state.acquisition);
    }
    render();
  }
  function stop() {
    if (state.playing) clearInterval(state.playing);
    state.playing = null;
    letters.forEach(l => $("play-" + l).textContent = "Play");
  }
  function loadContours(f) {
    if (window.SEM_CONTOURS[f.overlay_key]) return Promise.resolve(window.SEM_CONTOURS[f.overlay_key]);
    if (pending.has(f.overlay_key)) return pending.get(f.overlay_key);
    const promise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = f.overlay;
      script.onload = () => {
        script.remove(); pending.delete(f.overlay_key);
        cacheOrder.push(f.overlay_key);
        while (cacheOrder.length > 24) {
          const old = cacheOrder.shift();
          if (!letters.some((_, p) => frame(p).overlay_key === old)) delete window.SEM_CONTOURS[old];
        }
        resolve(window.SEM_CONTOURS[f.overlay_key] || []);
      };
      script.onerror = () => {script.remove(); pending.delete(f.overlay_key); reject(new Error("Contour asset unavailable. Check that the report folder is complete."));};
      document.head.append(script);
    });
    pending.set(f.overlay_key, promise);
    return promise;
  }
  function selected(c, p) {
    const s = state.selected;
    return s && (s.hole != null ? c.hole === s.hole :
      s.pane === p && s.name === state.names[p] && s.frame === frame(p).index && s.region === c.region_id);
  }
  function layer(p) {
    const group = svg("g", {"data-source": state.names[p], "data-frame": frame(p).index});
    const [height, width] = site().shape;
    const f = frame(p);
    const path = $("display").value === "difference" && f.difference_path ? f.difference_path : f.path;
    const image = svg("image", {href: path, x: -.5, y: -.5, width, height});
    image.addEventListener("error", () => {$("status-" + letters[p]).textContent = "Image unavailable. Check the report assets.";});
    group.append(image);
    const mode = $("contours").value, measure = $("measure").value;
    if (mode === "off") return group;
    for (const c of state.contours[p]) {
      const isSelected = selected(c, p);
      const selectedRing = measure === "refined" && c.refined.length ? c.refined : c.coarse;
      const ring = mode === "refined" && c.refined.length ? c.refined : c.coarse;
      if (isSelected) group.append(svg("path", {d: [ringPath(selectedRing), ...c.holes.map(ringPath)].join(" "),
        "fill-rule": "evenodd", fill: "#f9dc6d", "fill-opacity": .2, "pointer-events": "none"}));
      const outline = svg("path", {d: [ringPath(ring), ...c.holes.map(ringPath)].join(" "), "fill-rule": "evenodd",
        stroke: mode === "refined" ? "none" : c.status.coarse === "border" ? "#d7ad56" : "#43dbe2",
        class: `boundary${isSelected ? " selected" : ""}${c.status.coarse !== "valid" ? " partial" : ""}`,
        "data-pane": p, "data-region": c.region_id, "aria-label": c.hole ? `Hole ${c.hole}` : `Unmatched region ${c.region_id}`});
      outline.append(svg("title", {}, `${c.hole ? "Hole " + c.hole : "Local region " + c.region_id} · ${c.status[measure].replaceAll("_", " ")}`));
      group.append(outline);
      if ((mode === "both" || mode === "refined") && c.refined.length) {
        group.append(svg("path", {d: ringPath(c.refined), class: "refined-line fallback"}));
        let d = "";
        c.refined.forEach((point, i) => {
          const j = (i + 1) % c.refined.length;
          if (c.refined_valid[i] && c.refined_valid[j]) d += `M${point[1]},${point[0]}L${c.refined[j][1]},${c.refined[j][0]}`;
        });
        group.append(svg("path", {d, class: "refined-line"}));
      }
      if (isSelected && ring.length) {
        const x = ring.reduce((sum, q) => sum + q[1], 0) / ring.length;
        const y = ring.reduce((sum, q) => sum + q[0], 0) / ring.length;
        const size = Math.max(3, state.view[2] / 65);
        group.append(svg("text", {x, y, fill: "#fff", "font-size": size, "text-anchor": "middle", "pointer-events": "none"}, c.hole ? `#${c.hole}` : `R${c.region_id}`));
      }
    }
    return group;
  }
  function drawImages() {
    const wipe = $("layout").value === "wipe";
    $("images").classList.toggle("wipe", wipe);
    $("pane-b").hidden = wipe;
    $("wipe-control").hidden = !wipe;
    letters.forEach((l, p) => {
      const root = $("image-" + l);
      root.replaceChildren(); root.setAttribute("viewBox", state.view.join(" "));
      root.append(layer(p));
    });
    if (wipe) {
      const root = $("image-a"), [x, y, width, height] = state.view;
      const divider = x + width * Number($("wipe").value) / 100;
      const defs = svg("defs"), clip = svg("clipPath", {id: "wipe-clip", clipPathUnits: "userSpaceOnUse"});
      clip.append(svg("rect", {x: divider, y, width: x + width - divider, height})); defs.append(clip); root.append(defs);
      const second = layer(1); second.setAttribute("clip-path", "url(#wipe-clip)"); root.append(second);
      root.append(svg("line", {x1: divider, x2: divider, y1: y, y2: y + height, stroke: "white", "stroke-width": 2, "vector-effect": "non-scaling-stroke", "pointer-events": "none"}));
    }
    drawOverview();
  }
  function drawOverview() {
    const root = $("overview"), [height, width] = site().shape;
    const p = state.selected?.pane ?? state.active;
    root.replaceChildren(); root.setAttribute("viewBox", `0 0 ${width} ${height}`);
    root.append(svg("image", {href: frame(p).path, x: -.5, y: -.5, width, height}));
    root.append(svg("rect", {x: state.view[0], y: state.view[1], width: state.view[2], height: state.view[3],
      fill: "none", stroke: "#f9dc6d", "stroke-width": 2, "vector-effect": "non-scaling-stroke"}));
  }
  function clampView() {
    const [h, w] = site().shape;
    const viewport = $("image-a").getBoundingClientRect();
    state.view[2] = Math.max(8, Math.min(Math.max(w * 2, viewport.width), state.view[2]));
    state.view[3] = Math.max(8, Math.min(Math.max(h * 2, viewport.height), state.view[3]));
    state.view[0] = state.view[2] > w ? (w - state.view[2]) / 2 : Math.max(0, Math.min(w - state.view[2], state.view[0]));
    state.view[1] = state.view[3] > h ? (h - state.view[3]) / 2 : Math.max(0, Math.min(h - state.view[3], state.view[1]));
  }
  function fitHole() {
    const p = state.selected?.pane;
    if (p == null) return;
    const c = state.contours[p].find(c => selected(c, p));
    if (!c) return;
    const all = [...c.coarse, ...c.refined];
    if (!all.length) return;
    const xs = all.map(q => q[1]), ys = all.map(q => q[0]);
    const minX = Math.min(...xs), minY = Math.min(...ys), w = Math.max(...xs) - minX, h = Math.max(...ys) - minY;
    const pad = Math.max(4, Math.max(w, h) * .2);
    state.view = [minX - pad, minY - pad, w + pad * 2, h + pad * 2]; clampView(); drawImages();
  }
  function inspect(p, region) {
    const c = state.contours[p].find(c => c.region_id === region);
    if (!c) return;
    state.selected = {pane: p, name: state.names[p], frame: frame(p).index, region, hole: c.hole};
    state.active = p; $("fit-hole").disabled = false;
    fitHole(); drawMeasurements(); drawCharts();
  }
  function drawMeasurements() {
    const target = $("measurement-values"); target.replaceChildren();
    if (!state.selected) {$("selection").textContent = "Select a contour in either image to see exactly what was measured."; return;}
    const s = state.selected, method = $("measure").value;
    $("selection").textContent = s.hole != null ? `Hole ${s.hole} · shaded area uses the selected ${method === "coarse" ? "mask" : "refined"} measurement.` : `Local region ${s.region} · no cross-frame identity is available.`;
    letters.forEach((l, p) => {
      const c = state.contours[p].find(c => selected(c, p));
      const paragraph = document.createElement("p");
      if (!c) paragraph.textContent = `${l.toUpperCase()} · ${label(state.names[p])}: this hole is unavailable on the displayed frame.`;
      else {
        const value = c.measures[method];
        paragraph.textContent = `${l.toUpperCase()} · ${frameLabel(p)}: ` + (c.status[method] === "valid" && value ?
          `area ${fmt(value.area_px2, 2)} px² → ECD ${fmt(value.ecd)} ${report.unit}.` :
          `ECD unavailable (${c.status[method].replaceAll("_", " ")}).`);
        if (method === "refined") paragraph.textContent += ` Refined coverage ${fmt(c.refined_fraction * 100, 1)}%; remaining spans use the mask.`;
        if (c.review) paragraph.textContent += ` Needs inspection: sampled edge strength changed ${fmt(c.edge_strength.change * 100, 1)}%.`;
      }
      target.append(paragraph);
    });
  }
  function chart(id, tracks, {zero = false, selectedX = null, message = "No measurements available."} = {}) {
    const root = $(id); root.replaceChildren(); root.setAttribute("viewBox", "0 0 1100 185");
    const all = tracks.flatMap(t => t.points).filter(q => Number.isFinite(q.y));
    if (!all.length) {root.append(svg("text", {x: 20, y: 75}, message)); return;}
    const raw = site().series.raw.frames, x0 = time(raw[0]), x1 = time(raw.at(-1));
    const values = all.map(p => p.y);
    let lo = Math.min(...values), hi = Math.max(...values);
    if (zero) {const a = Math.max(Math.abs(lo), Math.abs(hi), .1); lo = -a; hi = a;}
    const pad = Math.max((hi - lo) * .1, .05); lo -= pad; hi += pad;
    const x = v => 65 + (v - x0) / (x1 - x0 || 1) * 1015;
    const y = v => 146 - (v - lo) / (hi - lo) * 128;
    for (let i = 0; i < 4; i++) {
      const value = lo + (hi - lo) * i / 3;
      root.append(svg("line", {x1: 65, x2: 1080, y1: y(value), y2: y(value), stroke: "#e1e7eb"}));
      root.append(svg("text", {x: 57, y: y(value) + 4, "text-anchor": "end"}, fmt(value, 2)));
    }
    for (let i = 0; i < 5; i++) {
      const value = x0 + (x1 - x0) * i / 4;
      root.append(svg("text", {x: x(value), y: 166, "text-anchor": "middle"}, fmt(value, x1 - x0 < 10 ? 2 : 0)));
    }
    root.append(svg("text", {x: 570, y: 183, "text-anchor": "middle"}, timeTitle()));
    if (zero) root.append(svg("line", {x1: 65, x2: 1080, y1: y(0), y2: y(0), stroke: "#929da7", "stroke-dasharray": "3 3"}));
    if (selectedX != null) root.append(svg("line", {x1: x(selectedX), x2: x(selectedX), y1: 15, y2: 147, class: "cursor"}));
    for (const track of tracks) {
      let d = "", connected = false;
      for (const q of track.points) {
        if (!Number.isFinite(q.y)) {connected = false; continue;}
        d += `${connected ? "L" : "M"}${x(q.x)},${y(q.y)}`; connected = true;
      }
      root.append(svg("path", {d, fill: "none", stroke: track.color, "stroke-width": 1.5, "stroke-dasharray": track.dashed ? "4 3" : "none"}));
      for (const q of track.points) {
        if (!Number.isFinite(q.y)) continue;
        const dot = svg("circle", {cx: x(q.x), cy: y(q.y), r: 2.6, fill: track.color, class: "dot"});
        dot.append(svg("title", {}, `${track.label} · ${q.description} · ${fmt(q.y)}`));
        dot.addEventListener("click", () => {
          if (track.pane != null) setFrame(track.pane, q.index);
          else {state.names[0] = "raw"; $("source-a").value = "raw"; setFrame(0, q.index);}
        }); root.append(dot);
      }
    }
  }
  function points(name, key) {
    return site().series[name].frames.map((f, index) => ({x: time(f), y: f[key], index,
      description: name === "average8" ? `block ${f.index} (${f.first_acquisition}–${f.last_acquisition})` : `acquisition ${f.index}`}));
  }
  function drawCharts() {
    const tracks = [{label: "Raw", color: "#8995a1", points: points("raw", "mean_dn")}];
    letters.forEach((_, p) => {if (state.names[p] !== "raw") tracks.push({label: label(state.names[p]), color: colors[p], pane: p, points: points(state.names[p], "mean_dn")});});
    $("brightness-legend").textContent = "Gray: raw input · Teal: A · Orange: B. Eight-frame means are located at block centers.";
    chart("brightness-chart", tracks, {selectedX: time(frame(state.active))});
    const delta = letters.map((_, p) => ({label: label(state.names[p]), color: colors[p], pane: p, points: points(state.names[p], "brightness_delta_dn")}));
    chart("delta-chart", delta, {zero: true, selectedX: time(frame(state.active)), message: "Select a model output to compare brightness with its corresponding raw input."});
    const drift = letters.flatMap((_, p) => ["dy", "dx"].map((axis, i) => ({label: `${label(state.names[p])} ${axis}`, color: colors[p], dashed: Boolean(i), pane: p, points: points(state.names[p], `output_minus_raw_${axis}_px`)})));
    $("drift-legend").textContent = "Teal: A · Orange: B · Solid: y displacement · Dashed: x displacement. Missing fits leave gaps.";
    chart("drift-chart", drift, {zero: true, selectedX: time(frame(state.active)), message: "Select a model output to inspect output-minus-raw motion."});
    const hole = state.selected?.hole, method = $("measure").value;
    const ecd = letters.map((_, p) => {
      const values = new Map(site().traces[state.names[p]]?.[hole]?.[method] || []);
      return {label: `${letters[p].toUpperCase()} ${label(state.names[p])}`, color: colors[p], pane: p,
        points: series(p).frames.map((f, index) => ({x: time(f), y: values.get(f.index), index, description: `acquisition ${f.order}`}))};
    });
    $("ecd-title").textContent = hole != null ? `Hole ${hole} · ${method === "coarse" ? "Mask" : "Refined"} ECD (${report.unit})` : "Diameter across acquisitions";
    $("ecd-coverage").textContent = hole != null ? ecd.map((t, p) => `${letters[p].toUpperCase()}: ${t.points.filter(q => Number.isFinite(q.y)).length}/${series(p).frames.length} usable observations`).join(" · ") : "";
    chart("ecd-chart", ecd, {selectedX: time(frame(state.active)), message: "Select a matched hole to inspect its diameter across acquisitions."});
  }
  function drawCoverage() {
    const target = $("coverage"); target.replaceChildren();
    const table = document.createElement("table");
    const head = document.createElement("tr");
    ["Images", "Boundary", "Common holes", "Usable measurements", "Unavailable", `ECD sample SD (${report.unit})`].forEach(text => {const th = document.createElement("th"); th.textContent = text; head.append(th);});
    table.append(head);
    const names = [...new Set(state.names)];
    const models = names.filter(n => report.arms.some(a => a.arm === n));
    const rows = [];
    for (const method of ["coarse", "refined"]) {
      const holes = Object.fromEntries(names.map(n => [n, Object.fromEntries(Object.entries(site().traces[n] || {}).map(([id, t]) =>
        [id, t[method].map(p => p[1]).filter(Number.isFinite)]).filter(([, v]) => v.length >= 2))]));
      const common = models.length ? Object.keys(holes[models[0]]).filter(id => models.every(n => id in holes[n])) : [];
      for (const name of names) {
        const ids = models.includes(name) ? common : Object.keys(holes[name]);
        const sds = ids.map(id => {const v = holes[name][id], mean = v.reduce((s, x) => s + x, 0) / v.length;
          return Math.sqrt(v.reduce((s, x) => s + (x - mean) ** 2, 0) / (v.length - 1));}).sort((a, b) => a - b);
        const used = ids.reduce((sum, id) => sum + holes[name][id].length, 0);
        const median = sds.length ? (sds[Math.floor((sds.length - 1) / 2)] + sds[Math.floor(sds.length / 2)]) / 2 : null;
        rows.push({series: name, method, common_hole_count: ids.length, valid_count: used,
          failed_count: ids.length ? ids.length * site().series[name].frames.length - used : "No matched sample",
          median_cd_std: median});
      }
    }
    for (const r of rows) {
      const tr = document.createElement("tr");
      [label(r.series), r.method === "coarse" ? "Mask" : "Refined", r.common_hole_count, r.valid_count, r.failed_count, fmt(r.median_cd_std)].forEach(text => {const td = document.createElement("td"); td.textContent = text; tr.append(td);});
      table.append(tr);
    }
    target.append(table);
  }
  async function render() {
    const revision = ++state.revision;
    letters.forEach((l, p) => {
      const f = frame(p), s = series(p), arm = report.arms.find(a => a.arm === state.names[p]);
      $("frame-" + l).max = s.frames.length; $("frame-" + l).value = state.indices[p] + 1;
      ["frame-", "prev-", "next-", "play-"].forEach(prefix => $(prefix + l).disabled = s.frames.length < 2);
      $("label-" + l).textContent = frameLabel(p) + (f.clipped ? " · CLIPPED" : "");
      $("file-" + l).href = f.path;
      $("meta-" + l).textContent = arm ? `Training: ${arm.registration} registration / ${arm.brightness} brightness · checkpoint step ${arm.step}${arm.ema ? " · EMA" : ""}` : "Uncorrected raw acquisitions · uint8 measurement pixels";
      const c = f.contour_counts || {};
      $("status-" + l).className = !c.complete || c.review ? "attention" : "";
      $("status-" + l).textContent = !c.detected ? `${l.toUpperCase()}: No contours detected; ECD unavailable.` :
        !c.complete ? `${l.toUpperCase()}: No complete holes. ${c.detected} detected; ${c.border || 0} partial at the border.` :
        `${l.toUpperCase()}: ${c.complete} complete holes; ${c.refined || 0} with usable refinement. ${c.border || 0} partial; ${c.review || 0} need edge inspection.`;
      if (c.detected && f.correspondence_status !== "available") $("status-" + l).textContent += " Cross-frame matching unavailable; local contours remain visible.";
      $("variation-" + l).hidden = !s.temporal_image;
      if (s.temporal_image) $("variation-" + l).src = s.temporal_image;
      $("variation-label-" + l).textContent = `${l.toUpperCase()} · ${label(state.names[p])}: ` + (s.temporal_image ?
        `native temporal variation; shared display 0–${fmt(s.temporal_limit_dn, 2)} DN. ${s.frames.length} observations.` : "Single reference: no temporal variation estimate.");
      state.contours[p] = []; // Never show the previous image's contour while loading.
    });
    $("scale-note").textContent = $("display").value === "pixels" ? "Shared display scale: 0–255 DN. Native coordinates; no brightness matching or image registration applied." :
      "Model panes show output minus the same raw acquisition (blue: negative; white: zero; red: positive). Baseline panes retain their pixels. Contours still measure the original saved image. " +
      letters.map((_, p) => series(p).difference_limit_dn ? `${letters[p].toUpperCase()}: ±${series(p).difference_limit_dn} DN; larger differences saturate for display.` : "").join(" ");
    drawImages(); drawMeasurements(); drawCharts(); drawCoverage();
    const loaded = await Promise.allSettled(letters.map((_, p) => loadContours(frame(p))));
    if (revision !== state.revision) return;
    loaded.forEach((result, p) => {
      if (result.status === "fulfilled") state.contours[p] = result.value;
      else $("status-" + letters[p]).textContent = result.reason.message;
    });
    drawImages(); drawMeasurements();
  }
  function changeSite() {
    stop(); state.selected = null; state.indices = [0, 0]; state.acquisition = 1;
    const names = Object.keys(site().series);
    state.names = ["raw", names.find(n => report.arms.some(a => a.arm === n)) || "average8"];
    letters.forEach((l, p) => {$("source-" + l).replaceChildren(...names.map(n => option(n, label(n)))); $("source-" + l).value = state.names[p];});
    state.view = [0, 0, site().shape[1], site().shape[0]];
    $("fit-hole").disabled = true;
    $("site-status").textContent = `Report generated · contour availability: ${site().contour_status} · ${site().hole_count} reference hole IDs. ${site().warnings.join(" ")}`;
    $("site-exports").replaceChildren();
    ["observations.csv", "per_hole.csv", "repeatability.csv", "frames.csv", "contours.json"].forEach(file => {const a = document.createElement("a"); a.href = `${site().name}/${file}`; a.textContent = file; $("site-exports").append(a, " · ");});
    drawCoverage(); render();
  }
  letters.forEach((l, p) => {
    $("source-" + l).addEventListener("change", e => {stop(); state.names[p] = e.target.value; state.indices[p] = atAcquisition(state.names[p], state.acquisition); render();});
    $("frame-" + l).addEventListener("input", e => {stop(); setFrame(p, Number(e.target.value) - 1);});
    $("prev-" + l).onclick = () => {stop(); setFrame(p, state.indices[p] - 1);};
    $("next-" + l).onclick = () => {stop(); setFrame(p, state.indices[p] + 1);};
    $("play-" + l).onclick = () => {
      const wasPlaying = state.playing && state.active === p; stop();
      if (wasPlaying) return;
      state.active = p; $("play-" + l).textContent = "Pause";
      state.playing = setInterval(() => setFrame(p, (state.indices[p] + 1) % series(p).frames.length), 350);
    };
    const root = $("image-" + l);
    let down = null;
    root.addEventListener("wheel", e => {
      e.preventDefault(); const point = new DOMPoint(e.clientX, e.clientY).matrixTransform(root.getScreenCTM().inverse());
      const k = e.deltaY > 0 ? 1.15 : 1 / 1.15;
      state.view = [point.x + (state.view[0] - point.x) * k, point.y + (state.view[1] - point.y) * k, state.view[2] * k, state.view[3] * k];
      clampView(); drawImages();
    }, {passive: false});
    root.addEventListener("pointerdown", e => {if (e.button === 0) down = {x: e.clientX, y: e.clientY, view: [...state.view], target: e.target, moved: false};});
    root.addEventListener("pointermove", e => {
      if (!down || !e.buttons) return;
      const dx = e.clientX - down.x, dy = e.clientY - down.y;
      if (Math.abs(dx) + Math.abs(dy) < 4) return;
      down.moved = true; root.setPointerCapture(e.pointerId);
      const rect = root.getBoundingClientRect(), scale = Math.min(rect.width / down.view[2], rect.height / down.view[3]);
      state.view = [down.view[0] - dx / scale, down.view[1] - dy / scale, down.view[2], down.view[3]];
      clampView(); drawImages();
    });
    root.addEventListener("pointerup", e => {
      if (!down) return;
      const target = down.target.closest?.("[data-region]");
      if (!down.moved && target) inspect(Number(target.dataset.pane), Number(target.dataset.region));
      down = null; if (root.hasPointerCapture(e.pointerId)) root.releasePointerCapture(e.pointerId);
    });
    root.addEventListener("pointercancel", () => {down = null;});
  });
  $("site").replaceChildren(...report.sites.map((s, i) => option(i, s.name)));
  $("site").onchange = e => {state.site = Number(e.target.value); changeSite();};
  ["layout", "display", "contours"].forEach(id => $(id).onchange = () => render());
  $("linked").onchange = () => {if ($("linked").checked) {state.indices[1 - state.active] = atAcquisition(state.names[1 - state.active], state.acquisition); render();}};
  $("wipe").oninput = drawImages;
  $("measure").onchange = () => {drawImages(); drawMeasurements(); drawCharts();};
  $("fit").onclick = () => {state.view = [0, 0, site().shape[1], site().shape[0]]; drawImages();};
  $("native").onclick = () => {const r = $("image-a").getBoundingClientRect(); const [x, y, w, h] = state.view;
    state.view = [x + w / 2 - r.width / 2, y + h / 2 - r.height / 2, r.width, r.height]; clampView(); drawImages();};
  $("fit-hole").onclick = fitHole;
  document.addEventListener("keydown", e => {if (["INPUT", "SELECT", "BUTTON"].includes(e.target.tagName)) return;
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {e.preventDefault(); stop(); setFrame(state.active, state.indices[state.active] + (e.key === "ArrowLeft" ? -1 : 1));}});
  changeSite();
})();
