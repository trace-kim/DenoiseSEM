/* Native-coordinate image inspection. No image normalization or registration. */
(() => {
  "use strict";
  const report = window.SEM_REPORT;
  const maskOnly = report.contour_method === "otsu";
  window.SEM_CONTOURS = Object.create(null);
  const $ = id => document.getElementById(id);
  const ns = "http://www.w3.org/2000/svg";
  const letters = ["a", "b"], colors = ["#14778d", "#d47732"];
  const modelColors = ["#14778d", "#d47732", "#7254a1", "#258049", "#bb406b", "#376ec0", "#99751b", "#745548"];
  const baselineColors = {raw: "#76818d", average8: "#323e49", average128: "#9a8f7c"};
  const state = {site: 0, names: ["raw", ""], indices: [0, 0], acquisition: 1,
    active: 1, view: [], contours: [[], []], selected: null, revision: 0, playing: null, ready: false, ecdSources: new Set()};
  // Requested controls are separate from the fully decoded pair on screen.
  const wanted = {site: 0, names: ["raw", ""], indices: [0, 0], acquisition: 1, display: "pixels"};
  let rendering = false;
  const imageCache = new Map();
  const imageBudget = 64 * 1024 * 1024;
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
  const openPath = points => points.length ? "M" + points.map(p => `${p[1]},${p[0]}`).join("L") : "";
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
    return Math.max(0, Math.min(report.sites[wanted.site].series[name].frames.length - 1,
      name === "average8" ? Math.floor((acquisition - 1) / 8) : acquisition - 1));
  }
  function setFrame(p, index) {
    state.active = p;
    const frames = report.sites[wanted.site].series[wanted.names[p]].frames;
    wanted.indices[p] = Math.max(0, Math.min(frames.length - 1, index));
    if (wanted.names[p] !== "average128") {
      const f = frames[wanted.indices[p]];
      wanted.acquisition = f.first_acquisition ?? f.index;
      if ($("linked").checked) wanted.indices[1 - p] = atAcquisition(wanted.names[1 - p], wanted.acquisition);
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
          // Displayed arrays have their own references; evict cache entries
          // even when their frame is visible, keeping the cache bounded.
          delete window.SEM_CONTOURS[cacheOrder.shift()];
        }
        resolve(window.SEM_CONTOURS[f.overlay_key] || []);
      };
      script.onerror = () => {script.remove(); pending.delete(f.overlay_key); reject(new Error("Contour asset unavailable. Check that the report folder is complete."));};
      document.head.append(script);
    });
    pending.set(f.overlay_key, promise);
    return promise;
  }
  function loadImage(path) {
    if (imageCache.has(path)) {
      const entry = imageCache.get(path);
      imageCache.delete(path); imageCache.set(path, entry);
      return entry.promise;
    }
    const image = new Image();
    const entry = {bytes: 0, promise: null};
    entry.promise = new Promise((resolve, reject) => {
      image.onload = async () => {
        try {
          await image.decode();
          entry.bytes = image.naturalWidth * image.naturalHeight * 4;
          let bytes = [...imageCache.values()].reduce((n, e) => n + e.bytes, 0);
          for (const [key, old] of imageCache) {
            if (bytes <= imageBudget) break;
            bytes -= old.bytes; imageCache.delete(key);
          }
          resolve(image);
        } catch (error) {imageCache.delete(path); reject(error);}
      };
      image.onerror = () => {imageCache.delete(path); reject(new Error(`Image unavailable: ${path}. Check the report assets.`));};
      image.src = path;
    });
    imageCache.set(path, entry);
    return entry.promise;
  }
  function surface(root) {
    const group = svg("g"), raster = svg("foreignObject", {x: -.5, y: -.5});
    const canvas = document.createElement("canvas"), overlay = svg("g");
    raster.append(canvas); group.append(raster, overlay); root.append(group);
    return {group, raster, canvas, overlay, path: null};
  }
  const surfaces = letters.map(l => surface($("image-" + l)));
  const overview = surface($("overview"));
  const overviewBox = svg("rect", {fill: "none", stroke: "#f9dc6d", "stroke-width": 2, "vector-effect": "non-scaling-stroke"});
  overview.overlay.append(overviewBox);
  const wipeDefs = svg("defs"), wipeClip = svg("clipPath", {id: "wipe-clip", clipPathUnits: "userSpaceOnUse"});
  const wipeRect = svg("rect"), wipeLine = svg("line", {stroke: "white", "stroke-width": 2, "vector-effect": "non-scaling-stroke", "pointer-events": "none"});
  wipeClip.append(wipeRect); wipeDefs.append(wipeClip); $("image-a").append(wipeDefs, wipeLine);
  function paint(target, image, path) {
    if (target.path === path) return;
    const [height, width] = site().shape;
    if (target.canvas.width !== width) target.canvas.width = width;
    if (target.canvas.height !== height) target.canvas.height = height;
    target.raster.setAttribute("width", width); target.raster.setAttribute("height", height);
    // Synchronous replacement of opaque, decoded pixels; never clear a frame.
    target.canvas.getContext("2d").drawImage(image, 0, 0, width, height);
    target.path = path;
  }
  function selected(c, p) {
    const s = state.selected;
    return s && (s.hole != null ? c.hole === s.hole :
      s.pane === p && s.name === state.names[p] && s.frame === frame(p).index && s.region === c.region_id);
  }
  function layer(p) {
    const group = svg("g", {"data-source": state.names[p], "data-frame": frame(p).index});
    const mode = $("contours").value, measure = $("measure").value;
    if (mode === "off") return group;
    for (const c of state.contours[p]) {
      const isSelected = selected(c, p);
      const selectedRing = measure === "refined" && c.refined.length ? c.refined : c.coarse;
      const ring = mode === "refined" && c.refined.length ? c.refined : c.coarse;
      if (isSelected && selectedRing.length) group.append(svg("path", {d: [ringPath(selectedRing), ...c.holes.map(ringPath)].join(" "),
        "fill-rule": "evenodd", fill: "#f9dc6d", "fill-opacity": .2, "pointer-events": "none"}));
      const outline = svg("path", {d: [ringPath(ring), ...c.holes.map(ringPath), ...(c.open_paths || []).map(openPath)].join(" "), "fill-rule": "evenodd",
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
  function drawImages(overlays = true) {
    if (!state.ready) return;
    const wipe = $("layout").value === "wipe";
    $("images").classList.toggle("wipe", wipe);
    $("pane-b").hidden = wipe;
    $("wipe-control").hidden = !wipe;
    letters.forEach((l, p) => {
      const root = $("image-" + l);
      root.setAttribute("viewBox", state.view.join(" "));
      if (overlays) surfaces[p].overlay.replaceChildren(layer(p));
    });
    const second = surfaces[1].group, parent = $(wipe ? "image-a" : "image-b");
    if (second.parentNode !== parent) parent.append(second);
    wipeLine.style.display = wipe ? "" : "none";
    if (wipe) {
      const root = $("image-a"), [x, y, width, height] = state.view;
      const divider = x + width * Number($("wipe").value) / 100;
      for (const [key, value] of Object.entries({x: divider, y, width: x + width - divider, height})) wipeRect.setAttribute(key, value);
      second.setAttribute("clip-path", "url(#wipe-clip)");
      for (const [key, value] of Object.entries({x1: divider, x2: divider, y1: y, y2: y + height})) wipeLine.setAttribute(key, value);
      root.append(wipeLine);
    } else second.removeAttribute("clip-path");
    drawOverview();
  }
  function drawOverview() {
    const root = $("overview"), [height, width] = site().shape;
    const p = state.selected?.pane ?? state.active;
    root.setAttribute("viewBox", `0 0 ${width} ${height}`);
    paint(overview, state.nativeImages[p], frame(p).path);
    for (const [key, value] of Object.entries({x: state.view[0], y: state.view[1], width: state.view[2], height: state.view[3]})) overviewBox.setAttribute(key, value);
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
    const all = [...c.coarse, ...c.refined, ...c.holes.flat(), ...(c.open_paths || []).flat()];
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
    if (!state.ready) return;
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
  function chart(id, tracks, {zero = false, selectedX = null, bands = true, message = "No measurements available."} = {}) {
    const root = $(id); root.replaceChildren(); root.setAttribute("viewBox", "0 0 1100 185");
    const all = tracks.flatMap(t => t.points).filter(q => Number.isFinite(q.y));
    if (!all.length) {root.append(svg("text", {x: 20, y: 75}, message)); return;}
    const raw = site().series.raw.frames, x0 = time(raw[0]), x1 = time(raw.at(-1));
    const values = all.map(p => p.y);
    for (const t of tracks) if (bands && t.stats?.sd != null) values.push(t.stats.mean - 3 * t.stats.sd, t.stats.mean + 3 * t.stats.sd);
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
    for (const track of tracks) {
      if (track.stats?.mean == null) continue;
      const {mean, sd} = track.stats;
      if (bands && sd != null) {
        const low = mean - 3 * sd, high = mean + 3 * sd;
        root.append(svg("rect", {x: 65, y: y(high), width: 1015, height: y(low) - y(high), fill: track.color, "fill-opacity": .07, class: "sigma-band"}));
        for (const v of [low, high]) root.append(svg("line", {x1: 65, x2: 1080, y1: y(v), y2: y(v), stroke: track.color, "stroke-dasharray": "6 4", class: "sigma-limit"}));
      }
      const line = svg("line", {x1: 65, x2: 1080, y1: y(mean), y2: y(mean), stroke: track.color, class: "mean-line"});
      line.append(svg("title", {}, `${track.label}: mean ${fmt(mean)}; 3σ ${fmt(sd == null ? null : 3 * sd)} ${report.unit}`));
      root.append(line);
    }
    if (zero) root.append(svg("line", {x1: 65, x2: 1080, y1: y(0), y2: y(0), stroke: "#929da7", "stroke-dasharray": "3 3"}));
    if (selectedX != null) root.append(svg("line", {x1: x(selectedX), x2: x(selectedX), y1: 15, y2: 147, class: "cursor"}));
    for (const track of tracks) {
      let d = "", connected = false;
      for (const q of track.points) {
        if (!Number.isFinite(q.y)) {connected = false; continue;}
        d += `${connected ? "L" : "M"}${x(q.x)},${y(q.y)}`; connected = true;
      }
      root.append(svg("path", {d, fill: "none", stroke: track.color, "stroke-width": 1.5, "stroke-dasharray": track.dashed ? "4 3" : "none",
        "data-series": track.name || track.label}));
      for (const q of track.points) {
        if (!Number.isFinite(q.y)) continue;
        const dot = svg("circle", {cx: x(q.x), cy: y(q.y), r: 2.6, fill: track.color, class: "dot", "data-series": track.name || track.label,
          "aria-label": `${track.label} · ${q.description} · ${fmt(q.y)}`});
        dot.append(svg("title", {}, `${track.label} · ${q.description} · ${fmt(q.y)}`));
        dot.addEventListener("click", () => {
          stop();
          if (track.name != null) {
            wanted.names[1] = track.name; $("source-b").value = track.name; setFrame(1, q.index);
          } else if (track.pane != null) setFrame(track.pane, q.index);
          else {wanted.names[0] = "raw"; $("source-a").value = "raw"; setFrame(0, q.index);}
        }); root.append(dot);
      }
    }
  }
  function points(name, key) {
    return site().series[name].frames.map((f, index) => ({x: time(f), y: f[key], index,
      description: name === "average8" ? `block ${f.index} (${f.first_acquisition}–${f.last_acquisition})` : `acquisition ${f.index}`}));
  }
  function statistics(points) {
    const values = points.map(p => p.y).filter(Number.isFinite), n = values.length;
    const mean = n ? values.reduce((sum, v) => sum + v, 0) / n : null;
    const sd = n > 1 ? Math.sqrt(values.reduce((sum, v) => sum + (v - mean) ** 2, 0) / (n - 1)) : null;
    return {n, mean, sd};
  }
  function ecdNames() {
    return [...report.arms.map(a => a.arm), ...Object.keys(baselineColors)].filter(n => site().series[n]);
  }
  function ecdColor(name) {
    if (name in baselineColors) return baselineColors[name];
    const index = report.arms.findIndex(a => a.arm === name);
    return modelColors[index] || `hsl(${(index * 137.5) % 360} 55% 38%)`;
  }
  function ecdControls() {
    $("ecd-hole").replaceChildren(option("", "Select a matched hole"),
      ...Array.from({length: site().hole_count}, (_, i) => option(String(i + 1), `Hole ${i + 1}`)));
    state.ecdSources = new Set(ecdNames().filter(n => !(n in baselineColors)));
    const root = $("ecd-sources"); root.replaceChildren();
    for (const name of ecdNames()) {
      const control = document.createElement("label"), input = document.createElement("input"), swatch = document.createElement("span");
      input.type = "checkbox"; input.checked = state.ecdSources.has(name);
      input.setAttribute("data-series", name); input.setAttribute("aria-label", `ECD: ${label(name)}`);
      swatch.className = "ecd-swatch"; swatch.style.backgroundColor = ecdColor(name);
      input.onchange = () => {
        if (input.checked) state.ecdSources.add(name); else state.ecdSources.delete(name);
        drawCharts();
      };
      control.append(input, swatch, label(name)); root.append(control);
    }
  }
  function drawStatistics(tracks) {
    const root = $("ecd-statistics"); root.replaceChildren();
    if (state.selected?.hole == null) return;
    const table = document.createElement("table"), head = document.createElement("tr");
    ["Source", "Usable / total", `Mean (${report.unit})`, `σ, sample SD (${report.unit})`, `3σ (${report.unit})`, `Mean ±3σ (${report.unit})`].forEach(value => {
      const th = document.createElement("th"); th.textContent = value; head.append(th);
    }); table.append(head);
    for (const t of [...tracks].sort((a, b) => (a.stats.sd ?? Infinity) - (b.stats.sd ?? Infinity))) {
      const {n, mean, sd} = t.stats, tr = document.createElement("tr");
      const range = sd == null ? "Unavailable (n < 2)" : `${fmt(mean - 3 * sd)} … ${fmt(mean + 3 * sd)}`;
      [t.label, `${n} / ${t.points.length}`, fmt(mean), fmt(sd), fmt(sd == null ? null : 3 * sd), range].forEach((value, i) => {
        const td = document.createElement("td"); td.textContent = value;
        if (!i) td.style.color = t.color;
        tr.append(td);
      }); table.append(tr);
    }
    root.append(table);
  }
  function drawCharts() {
    if (!state.ready) return;
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
    const ecd = ecdNames().filter(n => state.ecdSources.has(n)).map(name => {
      const values = new Map(site().traces[name]?.[hole]?.[method] || []);
      return {name, label: label(name), color: ecdColor(name), dashed: name in baselineColors,
        points: site().series[name].frames.map((f, index) => ({x: time(f), y: values.get(f.index), index,
          description: name === "average8" ? `block ${f.index} (${f.first_acquisition}–${f.last_acquisition})` : `acquisition ${f.order}`}))};
    });
    for (const track of ecd) track.stats = statistics(track.points);
    $("ecd-hole").value = hole == null ? "" : String(hole);
    $("ecd-title").textContent = hole != null ? `Hole ${hole} · ${method === "coarse" ? "Mask" : "Refined"} ECD (${report.unit})` : "Diameter across acquisitions";
    $("ecd-coverage").textContent = hole != null ? ecd.map(t => `${t.label}: ${t.stats.n}/${t.points.length} usable observations`).join(" · ") : "";
    chart("ecd-chart", ecd, {selectedX: time(frame(state.active)), bands: $("ecd-bands").checked,
      message: hole == null ? "Select a matched hole to compare every model." : !ecd.length ? "Select at least one ECD source." : "No usable ECD measurements for this hole and these sources."});
    drawStatistics(ecd);
  }
  function drawCoverage() {
    const target = $("coverage"); target.replaceChildren();
    const table = document.createElement("table");
    const head = document.createElement("tr");
    ["Images", "Boundary", "Contributing holes", "Usable measurements", "Unavailable", `ECD sample SD (${report.unit})`, `ECD 3σ (${report.unit})`].forEach(text => {const th = document.createElement("th"); th.textContent = text; head.append(th);});
    table.append(head);
    const rows = site().repeatability.filter(r => !maskOnly || r.method === "coarse");
    for (const r of rows) {
      const tr = document.createElement("tr");
      [label(r.series), r.method === "coarse" ? "Mask" : "Refined", r.common_hole_count, r.valid_count, r.failed_count,
        fmt(r.median_cd_std), fmt(r.median_cd_std == null ? null : 3 * r.median_cd_std)].forEach(text => {const td = document.createElement("td"); td.textContent = text; tr.append(td);});
      table.append(tr);
    }
    target.append(table);
  }
  function colorbar(id, title, low, high, difference = false) {
    const root = $(id); root.replaceChildren(); root.className = "colorbar";
    const heading = document.createElement("span"), ramp = document.createElement("div"), ticks = document.createElement("div");
    heading.className = "colorbar-title"; heading.textContent = title;
    ramp.className = "colorbar-ramp" + (difference ? " difference" : "");
    ramp.setAttribute("role", "img"); ramp.setAttribute("aria-label", `${title}: ${low} to ${high} DN${difference ? "; blue negative, white zero, red positive" : ""}`);
    ticks.className = "colorbar-ticks";
    for (const value of [low, (low + high) / 2, high]) {
      const tick = document.createElement("span"); tick.textContent = fmt(value, Number.isInteger(value) ? 0 : 2); ticks.append(tick);
    }
    root.append(heading, ramp, ticks);
  }
  function drawBand() {
    if (!state.ready || !$("band-section").open) return;
    const name = $("band-source").value, s = site().series[name], method = $("measure").value;
    const path = s?.contour_bands?.[method];
    $("contour-band").hidden = !path; $("band-file").hidden = !path;
    if (path) {
      if ($("contour-band").getAttribute("src") !== path) $("contour-band").src = path;
      $("band-file").href = path;
    }
    const count = s?.frames.reduce((n, f) => n + (f.contour_counts?.detected || 0), 0) || 0;
    $("band-label").textContent = path ? `${label(name)} · ${method === "coarse" ? "Mask" : "Refined"} boundaries · ${count} detected regions across ${s.frames.length} images.` +
      (s.frames.length === 1 ? " Single reference: no temporal band." : "") + (!count ? " No contours detected." : "") : "Contour band unavailable; regenerate this report with --render-only.";
  }
  async function render() {
    ++state.revision;
    wanted.display = $("display").value;
    letters.forEach((l, p) => {
      const s = report.sites[wanted.site].series[wanted.names[p]];
      $("frame-" + l).max = s.frames.length; $("frame-" + l).value = wanted.indices[p] + 1;
      ["frame-", "prev-", "next-", "play-"].forEach(prefix => $(prefix + l).disabled = s.frames.length < 2);
    });
    $("load-status").textContent = "Loading requested images… Displayed labels describe the pair currently on screen.";
    if (rendering) return;
    rendering = true;
    try {
      let revision;
      do {
        revision = state.revision;
        const next = {...wanted, names: [...wanted.names], indices: [...wanted.indices]};
        const nextSite = report.sites[next.site];
        const loaded = await Promise.allSettled(letters.map(async (_, p) => {
          const f = nextSite.series[next.names[p]].frames[next.indices[p]];
          const path = next.display === "difference" && f.difference_path ? f.difference_path : f.path;
          const [native, image, contours] = await Promise.all([loadImage(f.path), loadImage(path),
            loadContours(f).then(data => ({data}), error => ({data: [], error: error.message}))]);
          for (const img of [native, image]) if (img.naturalWidth !== nextSite.shape[1] || img.naturalHeight !== nextSite.shape[0]) throw new Error(`Unexpected image dimensions: ${path}`);
          return {native, image, contours, path};
        }));
        if (revision !== state.revision) continue;
        const failure = loaded.find(r => r.status === "rejected");
        if (failure) {
          stop(); $("load-status").textContent = `${failure.reason.message} Previous images and their labels remain displayed.`;
          continue;
        }
        await new Promise(resolve => requestAnimationFrame(resolve));
        if (revision !== state.revision) continue;
        const changedSite = !state.ready || next.site !== state.site;
        Object.assign(state, next);
        if (changedSite) {
          state.selected = null; $("fit-hole").disabled = true;
          state.view = [0, 0, site().shape[1], site().shape[0]];
          $("band-source").replaceChildren(...Object.keys(site().series).map(n => option(n, label(n))));
          $("band-source").value = state.names[1];
          ecdControls();
        }
        state.nativeImages = loaded.map(r => r.value.native);
        state.contours = loaded.map(r => r.value.contours.data);
        state.ready = true;
        loaded.forEach((r, p) => paint(surfaces[p], r.value.image, r.value.path));
        present();
        loaded.forEach((r, p) => {if (r.value.contours.error) $("status-" + letters[p]).textContent = r.value.contours.error;});
        $("load-status").textContent = "";
      } while (revision !== state.revision);
    } finally {rendering = false;}
  }
  function present() {
    letters.forEach((l, p) => {
      const f = frame(p), s = series(p), arm = report.arms.find(a => a.arm === state.names[p]);
      $("label-" + l).textContent = `${frameLabel(p)}${f.clipped ? " · CLIPPED" : ""} · ${label(state.names[p])}`;
      $("label-" + l).title = $("label-" + l).textContent;
      $("file-" + l).href = f.path;
      $("meta-" + l).textContent = arm ? `Training: ${arm.registration} registration / ${arm.brightness} brightness · checkpoint step ${arm.step}${arm.ema ? " · EMA" : ""}` : "Uncorrected raw acquisitions · uint8 measurement pixels";
      const c = f.contour_counts || {};
      $("status-" + l).className = !c.complete || c.review ? "attention" : "";
      $("status-" + l).textContent = !c.detected ? `${l.toUpperCase()}: No contours detected; ECD unavailable.` :
        !c.complete ? `${l.toUpperCase()}: No complete holes. ${c.detected} detected; ${c.border || 0} partial at the border.` :
        `${l.toUpperCase()}: ${c.complete} complete holes; ${c.refined || 0} with usable refinement. ${c.border || 0} partial; ${c.review || 0} need edge inspection.`;
      if (maskOnly) $("status-" + l).textContent =
        `${l.toUpperCase()}: ${c.complete || 0} complete Otsu regions; ${c.border || 0} partial at the border. Otsu threshold ${fmt(f.otsu_threshold_dn)} DN. No edge refinement.`;
      if (c.detected && f.correspondence_status !== "available") $("status-" + l).textContent += " Cross-frame matching unavailable; local contours remain visible.";
      $("variation-" + l).hidden = !s.temporal_image;
      if (s.temporal_image && $("variation-" + l).getAttribute("src") !== s.temporal_image) $("variation-" + l).src = s.temporal_image;
      $("variation-label-" + l).textContent = `${l.toUpperCase()} · ${label(state.names[p])}: ` + (s.temporal_image ?
        `native temporal variation; shared display 0–${fmt(s.temporal_limit_dn, 2)} DN. ${s.frames.length} observations.` : "Single reference: no temporal variation estimate.");
      const difference = state.display === "difference" && Boolean(f.difference_path);
      const limit = s.difference_limit_dn;
      colorbar("scale-" + l, `${l.toUpperCase()} · ${difference ? "Output − raw (DN)" : "Saved intensity (DN)"}`, difference ? -limit : 0, difference ? limit : 255, difference);
      $("scale-variation-" + l).hidden = !s.temporal_image;
      if (s.temporal_image) colorbar("scale-variation-" + l, "Temporal sample SD (DN)", 0, s.temporal_limit_dn);
    });
    colorbar("scale-overview", "Saved intensity (DN)", 0, 255);
    $("scale-note").textContent = state.display === "pixels" ? "Shared display scale: 0–255 DN. Native coordinates; no brightness matching or image registration applied." :
      "Model panes show output minus the same raw acquisition (blue: negative; white: zero; red: positive). Baseline panes retain their pixels. Contours still measure the original saved image. " +
      letters.map((_, p) => series(p).difference_limit_dn ? `${letters[p].toUpperCase()}: ±${series(p).difference_limit_dn} DN; larger differences saturate for display.` : "").join(" ");
    $("site-status").textContent = `Report generated · contour availability: ${site().contour_status} · ${site().hole_count} reference hole IDs. ${site().warnings.join(" ")}`;
    $("site-exports").replaceChildren();
    ["observations.csv", "per_hole.csv", "repeatability.csv", "frames.csv", "contours.json"].forEach(file => {const a = document.createElement("a"); a.href = `${site().name}/${file}`; a.textContent = file; $("site-exports").append(a, " · ");});
    drawImages(); drawMeasurements(); drawCharts(); drawCoverage(); drawBand();
  }
  function changeSite() {
    stop(); wanted.indices = [0, 0]; wanted.acquisition = 1;
    const names = Object.keys(report.sites[wanted.site].series);
    wanted.names = ["raw", names.find(n => report.arms.some(a => a.arm === n)) || "average8"];
    letters.forEach((l, p) => {$("source-" + l).replaceChildren(...names.map(n => option(n, label(n)))); $("source-" + l).value = wanted.names[p];});
    render();
  }
  letters.forEach((l, p) => {
    $("source-" + l).addEventListener("change", e => {stop(); wanted.names[p] = e.target.value; wanted.indices[p] = atAcquisition(wanted.names[p], wanted.acquisition); render();});
    $("frame-" + l).addEventListener("input", e => {stop(); setFrame(p, Number(e.target.value) - 1);});
    $("prev-" + l).onclick = () => {stop(); setFrame(p, wanted.indices[p] - 1);};
    $("next-" + l).onclick = () => {stop(); setFrame(p, wanted.indices[p] + 1);};
    $("play-" + l).onclick = () => {
      const wasPlaying = state.playing && state.active === p; stop();
      if (wasPlaying) return;
      state.active = p; $("play-" + l).textContent = "Pause";
      state.playing = setInterval(() => {if (!rendering) setFrame(p, (wanted.indices[p] + 1) % series(p).frames.length);}, 350);
    };
    const root = $("image-" + l);
    let down = null;
    root.addEventListener("wheel", e => {
      if (!state.ready) return;
      e.preventDefault(); const point = new DOMPoint(e.clientX, e.clientY).matrixTransform(root.getScreenCTM().inverse());
      const k = e.deltaY > 0 ? 1.15 : 1 / 1.15;
      state.view = [point.x + (state.view[0] - point.x) * k, point.y + (state.view[1] - point.y) * k, state.view[2] * k, state.view[3] * k];
      clampView(); drawImages(false);
    }, {passive: false});
    root.addEventListener("pointerdown", e => {if (state.ready && e.button === 0) down = {x: e.clientX, y: e.clientY, view: [...state.view], target: e.target, moved: false};});
    root.addEventListener("pointermove", e => {
      if (!down || !e.buttons) return;
      const dx = e.clientX - down.x, dy = e.clientY - down.y;
      if (Math.abs(dx) + Math.abs(dy) < 4) return;
      down.moved = true; root.setPointerCapture(e.pointerId);
      const rect = root.getBoundingClientRect(), scale = Math.min(rect.width / down.view[2], rect.height / down.view[3]);
      state.view = [down.view[0] - dx / scale, down.view[1] - dy / scale, down.view[2], down.view[3]];
      clampView(); drawImages(false);
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
  $("site").onchange = e => {wanted.site = Number(e.target.value); changeSite();};
  $("display").onchange = () => render();
  $("layout").onchange = () => drawImages(false);
  $("contours").onchange = () => drawImages();
  $("linked").onchange = () => {if ($("linked").checked) {wanted.indices[1 - state.active] = atAcquisition(wanted.names[1 - state.active], wanted.acquisition); render();}};
  $("wipe").oninput = () => drawImages(false);
  $("measure").onchange = () => {drawImages(); drawMeasurements(); drawCharts(); drawBand();};
  $("ecd-hole").onchange = () => {
    if (!state.ready) return;
    const hole = $("ecd-hole").value;
    state.selected = hole ? {hole: Number(hole), pane: state.active} : null;
    $("fit-hole").disabled = !hole;
    drawImages(); drawMeasurements(); drawCharts();
  };
  $("ecd-bands").onchange = drawCharts;
  $("ecd-all").onclick = () => {
    if (!state.ready) return;
    state.ecdSources = new Set(ecdNames().filter(n => !(n in baselineColors)));
    for (const control of $("ecd-sources").children) {
      const input = control.children[0]; input.checked = state.ecdSources.has(input.getAttribute("data-series"));
    }
    drawCharts();
  };
  $("band-source").onchange = drawBand;
  $("band-section").ontoggle = drawBand;
  $("fit").onclick = () => {if (state.ready) {state.view = [0, 0, site().shape[1], site().shape[0]]; drawImages(false);}};
  $("native").onclick = () => {if (!state.ready) return; const r = $("image-a").getBoundingClientRect(); const [x, y, w, h] = state.view;
    state.view = [x + w / 2 - r.width / 2, y + h / 2 - r.height / 2, r.width, r.height]; clampView(); drawImages(false);};
  $("fit-hole").onclick = fitHole;
  document.addEventListener("keydown", e => {if (["INPUT", "SELECT", "BUTTON"].includes(e.target.tagName)) return;
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") {e.preventDefault(); stop(); setFrame(state.active, wanted.indices[state.active] + (e.key === "ArrowLeft" ? -1 : 1));}});
  if (maskOnly) {
    $("contours").value = "coarse";
    for (const entry of $("contours").options) entry.disabled = ["both", "refined"].includes(entry.value);
    $("measure").value = "coarse";
    for (const entry of $("measure").options) entry.disabled = entry.value === "refined";
    $("refinement-legend").hidden = true;
    const settings = report.otsu_settings;
    $("detector-note").textContent = `Contours: Gaussian + Otsu · ${settings.polarity} foreground · σ ${settings.sigma_px} px · minimum area ${settings.min_area_px} px. Same settings for every source; each image has its own threshold.`;
    $("measurement-note").textContent = "ECD uses the area enclosed by the Otsu mask, subtracting interior rings. These are segmentation boundaries, not subpixel edge measurements. Open border paths are displayed but excluded from area/ECD.";
  } else {
    $("detector-note").textContent = "Contours: current segmentation and edge-refinement pipeline.";
  }
  changeSite();
})();
