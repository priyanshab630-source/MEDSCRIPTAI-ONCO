// The Docking tab must render in all three states, and must not reference a
// function that no longer exists. Run with: node tests/docking_tab_check.js
const fs = require("fs"), vm = require("vm");

function node() {
  return { innerHTML: "", textContent: "", value: "all", dataset: {}, style: {},
    classList: { toggle() {}, add() {}, remove() {} },
    querySelector: () => node(), querySelectorAll: () => [], appendChild() {},
    addEventListener() {},
    getContext: () => ({ clearRect(){}, beginPath(){}, arc(){}, fill(){}, moveTo(){},
      lineTo(){}, stroke(){}, fillText(){}, fillRect(){}, setLineDash(){}, save(){},
      restore(){} }),
    width: 820, height: 520 };
}
const elements = {};
global.document = {
  getElementById: (id) => (elements[id] = elements[id] || node()),
  querySelectorAll: () => [], querySelector: () => node(),
};
global.window = global;

const sample = {
  available: true, engine: "AutoDock Vina 1.2", poses: [[-7.19, 0, 0]],
  redocking_rmsd: 0.73, crystal_pose_score: -6.76,
  own_analysis: { counts: { hydrogen_bonds: 5, binding_site_residues: 18 }, residues: ["ASP51"] },
  plip: { counts: { hydrogen_bonds: 4 } },
};
const emptyScene = { centre: [0,0,0], protein: [], ligand: [], bonds: [], hbonds: [],
  labels: [], backbone: [], sticks: [], counts: { protein_atoms: 0, ligand_atoms: 0, hbonds: 0 } };
global.fetch = async (url) => ({
  ok: true, status: 200,
  json: async () => (String(url).includes("scene") ? emptyScene : sample),
});

const html = fs.readFileSync("api/static/index.html", "utf8");
vm.runInThisContext(html.match(/<script>([\s\S]*)<\/script>/)[1]);

const checks = {};
const entryPoints = ["renderDocking", "bindPoseCanvas", "drawPose", "loadDocking", "selectTab"];
for (const name of entryPoints) {
  checks[`${name} is defined`] = typeof global[name] === "function";
}

function el(id) {                       // the stub only creates on first access
  return (elements[id] = elements[id] || node());
}

(async () => {
  // A: nothing analysed yet -> the sample, clearly labelled
  window.lastReport = null;
  el("dock-interactions").innerHTML = "Loading…";
  try {
    await loadDocking();
    checks["sample path renders"] = el("dock-interactions").innerHTML.includes("table");
  } catch (error) { checks["sample path renders"] = false; console.log("A threw:", error.message); }

  // B: analysed, not docked -> an explanation, never the sample
  window.lastReport = { hypothesis: { target: "SHP2" }, docking: {},
    trace: [{ agent: "analysis_agent",
              message: "real docking skipped - AutoDock Vina is not installed" }] };
  el("dock-interactions").innerHTML = "Loading…";
  try {
    await loadDocking();
    const markup = el("dock-interactions").innerHTML;
    checks["skip path explains why"] = markup.includes("not docked") && markup.includes("Vina");
    checks["skip path hides the sample"] = !markup.includes("-7.19");
  } catch (error) { checks["skip path explains why"] = false; console.log("B threw:", error.message); }

  // C: analysed with a pose -> the compound's own numbers
  window.lastReport = { hypothesis: { target: "SHP2" },
    docking: { target: "SHP2", engine: "AutoDock Vina 1.2", best_affinity: -6.22,
      poses: [[-6.22]],
      own_analysis: { counts: { hydrogen_bonds: 6, binding_site_residues: 19 }, residues: ["ASP425"] },
      plip: {}, scene: emptyScene } };
  el("dock-interactions").innerHTML = "Loading…";
  try {
    await loadDocking();
    checks["own pose renders"] = el("dock-kpis").innerHTML.includes("-6.22");
    checks["missing PLIP shows a dash"] = el("dock-interactions").innerHTML.includes("—");
  } catch (error) { checks["own pose renders"] = false; console.log("C threw:", error.message); }

  for (const [name, ok] of Object.entries(checks)) console.log((ok ? "PASS " : "FAIL ") + name);
  if (Object.values(checks).some((ok) => !ok)) process.exit(1);
})();
