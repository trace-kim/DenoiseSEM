const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const root = process.argv[2];
const elements = new Map();
class Element {
  constructor() {this.value = "0"; this.events = {}; this.textContent = "";}
  replaceChildren(...children) {this.children = children; this.value = children[0]?.value || "";}
  addEventListener(event, callback) {this.events[event] = callback;}
  removeAttribute(name) {delete this[name];}
}
const get = id => {if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id);};
const window = {};
const context = vm.createContext({window, document: {getElementById: get},
  Option: function(text, value) {this.text = text; this.value = value;}});
vm.runInContext(fs.readFileSync(path.join(root, "data.js"), "utf8"), context);
vm.runInContext(fs.readFileSync(path.join(root, "preview.js"), "utf8"), context);
assert.match(get("kind").textContent, /SYNTHETIC/);
assert.equal(get("acquisition").max, 127);
for (let i = 0; i < 128; i++) {
  get("acquisition").value = String(i);
  get("acquisition").events.input();
  assert.match(get("frame-label").textContent, new RegExp(`Saved image ${i + 1} of 128 · Acquisition ${i + 1}$`));
  const expected = window.CONTOUR_PREVIEW.sites[0].series.raw.frames[i];
  assert.equal(get("original-image").src, expected.display);
  assert.equal(get("current-image").src, expected.current);
  assert.equal(get("baseline-link").href, expected.baseline);
}
get("source").value = "average8";
get("source").events.change();
assert.equal(get("acquisition").max, 1);
assert.match(get("frame-label").textContent, /Acquisitions 9–16 \(index 2\)/);
get("acquisition").value = "1";
get("acquisition").events.input();
assert.match(get("frame-label").textContent, /Acquisitions 25–32 \(index 4\)/);
get("source").value = "average128";
get("source").events.change();
assert.equal(get("acquisition").disabled, true);
assert.match(get("frame-label").textContent, /Acquisitions 1–128/);
get("source").value = "model_a";
get("source").events.change();
assert.equal(get("acquisition").value, 0);
assert.match(get("frame-label").textContent, /Acquisition 9$/);
get("acquisition").value = "1";
get("acquisition").events.input();
assert.match(get("frame-label").textContent, /Acquisition 29$/);
get("source").value = "model_b";
get("source").events.change();
assert.equal(get("acquisition").disabled, true);
assert.match(get("frame-label").textContent, /Saved image 1 of 1/);
get("site").value = "1";
get("site").events.change();
assert.equal(get("acquisition").disabled, true);
assert.match(get("frame-label").textContent, /No saved images/);
assert.equal(get("original-image").hidden, true);
