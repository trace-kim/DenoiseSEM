/* Exercise asynchronous navigation without a GPU, browser, or network. */
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const root = process.argv[2];
const elements = new Map(), held = new Map(), failures = new Set(), images = [];
let context;
class Element {
  constructor(tag = "div") {
    this.tagName = tag.toUpperCase(); this.children = []; this.attributes = {};
    this.events = {}; this.style = {}; this.value = ""; this.textContent = "";
    this.classList = {toggle() {}}; this.dataset = {};
  }
  append(...children) {
    for (const child of children) {
      if (typeof child !== "object") continue;
      child.remove(); this.children.push(child); child.parentNode = this;
      if (child.tagName === "SCRIPT") setImmediate(async () => {
        try {
          if (held.has(child.src)) await held.get(child.src).promise;
          if (failures.has(child.src)) throw Error("Missing contour fixture");
          vm.runInContext(fs.readFileSync(path.join(root, child.src), "utf8"), context); child.onload();
        }
        catch {child.onerror();}
      });
    }
  }
  remove() {if (this.parentNode) this.parentNode.children = this.parentNode.children.filter(c => c !== this); this.parentNode = null;}
  replaceChildren(...children) {
    // A redraw must not detach an existing image surface, even transiently.
    assert(!descendants(this).some(e => e.tagName === "CANVAS"), "live raster removed");
    for (const child of [...this.children]) child.remove();
    this.append(...children);
    if (this.tagName === "SELECT") this.value = children[0]?.value || "";
  }
  setAttribute(key, value) {this.attributes[key] = String(value); if (key.startsWith("data-")) this.dataset[key.slice(5)] = String(value);}
  getAttribute(key) {return this.attributes[key] ?? null;}
  removeAttribute(key) {delete this.attributes[key];}
  addEventListener(event, callback) {this.events[event] = callback;}
  closest(selector) {return selector === "[data-region]" && this.dataset.region ? this : this.parentNode?.closest(selector);}
  getBoundingClientRect() {return {width: 600, height: 500};}
  hasPointerCapture() {return false;}
  get options() {return this.children;}
  set src(value) {this.attributes.src = value;}
  get src() {return this.attributes.src;}
  getContext() {
    return {drawImage: image => {assert(image.decoded, "undecoded image displayed"); this.draws = (this.draws || 0) + 1; this.pixels = image.src;},
      clearRect() {throw Error("visible pixels erased");}};
  }
}
const descendants = e => e.children.flatMap(c => [c, ...descendants(c)]);
const get = id => {
  if (!elements.has(id)) elements.set(id, new Element(["site", "source-a", "source-b", "band-source", "ecd-hole"].includes(id) ? "select" : "div"));
  return elements.get(id);
};
class FakeImage {
  constructor() {this.naturalWidth = 64; this.naturalHeight = 64; images.push(this);}
  set src(value) {this.path = value; setImmediate(() => failures.has(value) ? this.onerror() : this.onload());}
  get src() {return this.path;}
  async decode() {if (held.has(this.path)) await held.get(this.path).promise; this.decoded = true;}
}
const hold = name => {
  let release;
  const promise = new Promise(resolve => {release = resolve;});
  held.set(name, {promise, release});
};
const settle = async () => {for (let i = 0; i < 40; i++) await new Promise(setImmediate);};
const event = (id, type, value) => {
  const e = get(id); if (value !== undefined) e.value = value;
  (e.events[type] || e["on" + type])({target: e});
};
const text = e => e.textContent + e.children.map(text).join("|");
const canvas = id => descendants(get(id)).find(e => e.tagName === "CANVAS");
const classes = (id, cls) => descendants(get(id)).filter(e => e.attributes.class === cls);
get("layout").value = "side"; get("display").value = "pixels"; get("contours").value = "coarse";
get("measure").value = "coarse"; get("linked").checked = true; get("wipe").value = 50;
get("ecd-bands").checked = false;
get("contours").append(...["both", "refined", "coarse", "off"].map(value => Object.assign(new Element("option"), {value})));
context = vm.createContext({window: {}, document: {getElementById: get, head: new Element("head"),
  createElement: tag => new Element(tag), createElementNS: (_, tag) => new Element(tag), addEventListener() {}},
  Image: FakeImage, Option: function(text, value) {return Object.assign(new Element("option"), {textContent: text, value});},
  requestAnimationFrame: fn => setImmediate(fn), setInterval, clearInterval});
vm.runInContext(fs.readFileSync(path.join(root, "viewer/data.js"), "utf8"), context);
const report = context.window.SEM_REPORT;
report.sites[0].traces.raw[1].coarse = [[1, 10], [2, 12], [3, null], [4, 14]];
report.sites[0].traces.model[1].coarse = [[1, 15]];
report.sites[0].traces.ft_noisy[1].coarse = [[1, 10], [2, 12], [3, null], [4, 14]];
report.sites[0].traces.ft_consist[1].coarse = [[1, 11], [2, 12], [3, null], [4, 13]];
const ecdSources = () => descendants(get("ecd-chart")).filter(e => e.tagName === "PATH").map(e => e.dataset.series);
const ecdToggle = (name, checked) => {
  const input = descendants(get("ecd-sources")).find(e => e.dataset.series === name);
  input.checked = checked; input.onchange();
};
vm.runInContext(fs.readFileSync(path.join(root, "viewer/comparison.js"), "utf8"), context);
(async () => {
  await settle();
  const a = canvas("image-a"), b = canvas("image-b");
  assert.match(a.pixels, /raw\/frame_001/); assert.match(b.pixels, /model\/frame_001/);
  const region = descendants(get("image-a")).find(e => e.dataset.region);
  get("image-a").events.pointerdown({button: 0, clientX: 10, clientY: 10, target: region});
  get("image-a").events.pointerup({});
  assert.match(text(get("ecd-statistics")), /3 \/ 8\|12.000\|2.000\|6.000\|6.000 … 18.000/);
  assert.match(text(get("ecd-statistics")), /3 \/ 8\|12.000\|1.000\|3.000\|9.000 … 15.000/);
  assert.match(text(get("ecd-statistics")), /1 \/ 8\|15.000\|Unavailable\|Unavailable/);
  assert.deepEqual(ecdSources(), ["model", "ft_noisy", "ft_consist"]);
  assert.equal(classes("ecd-chart", "mean-line").length, 3);
  assert.equal(classes("ecd-chart", "sigma-limit").length, 0);
  assert.equal(get("ecd-hole").value, "1");
  const originalColor = descendants(get("ecd-chart")).find(e => e.dataset.series === "ft_consist").attributes.stroke;
  const loadedCount = images.length;
  get("ecd-bands").checked = true; event("ecd-bands", "change");
  assert.equal(classes("ecd-chart", "sigma-limit").length, 4);
  ecdToggle("ft_noisy", false);
  assert.deepEqual(ecdSources(), ["model", "ft_consist"]);
  event("ecd-all", "click");
  assert.deepEqual(ecdSources(), ["model", "ft_noisy", "ft_consist"]);
  assert.equal(images.length, loadedCount, "chart-only controls load no images or measurements");
  ecdToggle("raw", true);
  assert.equal(classes("ecd-chart", "mean-line").length, 4);
  assert.equal(classes("ecd-chart", "sigma-limit").length, 6);
  // A point from a model outside A/B opens the correct saved acquisition.
  const candidate = classes("ecd-chart", "dot").find(e => e.dataset.series === "ft_consist" && /acquisition 2/.test(text(e)));
  candidate.events.click(); await settle();
  assert.match(b.pixels, /ft_consist\/frame_002/);
  assert.match(a.pixels, /raw\/frame_002/);
  assert.equal(get("source-b").value, "ft_consist");
  assert.equal(descendants(get("ecd-chart")).find(e => e.dataset.series === "ft_consist").attributes.stroke, originalColor);
  assert.match(text(get("coverage")), /ft \+ noisy/);
  assert.match(text(get("coverage")), /ft \+ consist/);
  event("frame-a", "input", "1"); await settle();
  event("source-b", "change", "model"); await settle();
  // While either half is decoding, pixels, labels and overlays stay together.
  hold("site/model/frame_002.png");
  event("frame-a", "input", "2"); await settle();
  assert.equal(canvas("image-a"), a); assert.equal(canvas("image-b"), b);
  assert.match(a.pixels, /frame_001/); assert.match(b.pixels, /frame_001/);
  assert.match(get("label-a").textContent, /Acquisition 1\//);
  assert.equal(descendants(get("image-a")).find(e => e.dataset.frame).dataset.frame, "1");
  // Coalesce rapid changes: decoded frame 2 must never flash before frame 4.
  hold("site/model/frame_004.png");
  event("frame-a", "input", "3"); event("frame-a", "input", "4");
  held.get("site/model/frame_002.png").release(); await settle();
  assert.match(a.pixels, /frame_001/); assert.match(b.pixels, /frame_001/);
  assert(!images.some(i => /frame_003/.test(i.src)), "superseded frame loaded");
  held.get("site/model/frame_004.png").release(); await settle();
  assert.match(a.pixels, /frame_004/); assert.match(b.pixels, /frame_004/);
  assert.match(get("label-a").textContent, /Acquisition 4\//);
  assert.equal(descendants(get("image-a")).find(e => e.dataset.frame).dataset.frame, "4");
  const draws = [a.draws, b.draws];
  event("layout", "change", "wipe"); event("wipe", "input", "70"); event("native", "click");
  assert.equal(canvas("image-a"), a); assert(descendants(get("image-a")).includes(b));
  assert.deepEqual([a.draws, b.draws], draws);
  event("layout", "change", "side"); assert.equal(canvas("image-b"), b);
  // A missing image retains the previous pair and measurement provenance.
  failures.add("site/model/frame_006.png");
  event("frame-a", "input", "6"); await settle();
  assert.match(get("load-status").textContent, /Image unavailable/);
  assert.match(a.pixels, /frame_004/); assert.match(b.pixels, /frame_004/);
  assert.match(get("label-b").textContent, /Acquisition 4\//);
  event("frame-a", "input", "1"); await settle();
  hold("site/viewer/model/007.js");
  event("frame-a", "input", "7"); await settle();
  assert.match(a.pixels, /frame_001/); assert.match(b.pixels, /frame_001/);
  held.get("site/viewer/model/007.js").release(); await settle();
  assert.match(b.pixels, /frame_007/);
  // Missing contours must never reuse another frame's boundary.
  failures.add("site/viewer/model/008.js");
  event("frame-a", "input", "8"); await settle();
  assert.match(b.pixels, /frame_008/);
  assert.equal(descendants(get("image-b")).filter(e => e.dataset.region).length, 0);
  assert.match(get("status-b").textContent, /Contour asset unavailable/);
  event("frame-a", "input", "1"); await settle();
  event("display", "change", "difference"); await settle();
  assert.match(text(get("scale-a")), /Saved intensity.*0\|127.50\|255/);
  assert.match(text(get("scale-b")), /Output − raw.*-32\|0\|32/);
  assert(get("scale-b").children.some(e => e.className === "colorbar-ramp difference"));
  assert.match(text(get("scale-overview")), /0\|127.50\|255/);
  assert.match(text(get("scale-variation-b")), /Temporal sample SD/);
  get("band-section").open = true; event("band-section", "toggle");
  assert.match(get("contour-band").src, /model\/coarse_band.svg/);
  event("band-source", "change", "average128");
  assert.match(get("band-label").textContent, /Single reference/);
  event("source-b", "change", "average128"); await settle();
  assert(get("frame-b").disabled); assert.match(text(get("scale-b")), /Saved intensity/);
  assert(get("scale-variation-b").hidden);
  event("source-b", "change", "average8"); await settle();
  assert.match(get("label-b").textContent, /acquisitions 1–8/);
  assert.deepEqual(ecdSources(), ["model", "ft_noisy", "ft_consist", "raw"]);
  event("ecd-hole", "change", "");
  assert.match(text(get("ecd-chart")), /Select a matched hole/);
  event("ecd-hole", "change", "1");
  for (const name of ["model", "ft_noisy", "ft_consist", "raw", "average8"]) report.sites[0].traces[name][1].coarse = [];
  event("measure", "change", "coarse");
  assert.match(text(get("ecd-statistics")), /0 \/ 8\|Unavailable\|Unavailable/);
  assert.equal(classes("ecd-chart", "mean-line").length, 0);
  assert.match(text(get("ecd-chart")), /No usable ECD/);
  for (const name of ["model", "ft_noisy", "ft_consist", "raw"]) ecdToggle(name, false);
  assert.match(text(get("ecd-chart")), /Select at least one/);
  console.log("Atomic image swaps, stale requests, image failures, wipe, all-model ECD statistics, bands and colorbars passed.");
})().catch(error => {console.error(error); process.exitCode = 1;});
