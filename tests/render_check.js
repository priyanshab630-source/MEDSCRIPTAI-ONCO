// Render the dashboard's own functions against a real report, with a DOM stub.
const fs = require("fs");
const vm = require("vm");

function node() {
  return {
    innerHTML: "", textContent: "", value: "all", dataset: {}, classList: { toggle() {} },
    querySelector: () => null, querySelectorAll: () => [], appendChild() {},
    onclick: null, onchange: null,
  };
}
const elements = {};
global.document = {
  getElementById: (id) => (elements[id] = elements[id] || node()),
  querySelectorAll: () => [],
  querySelector: () => null,
};
global.fetch = () => Promise.reject(new Error("network disabled in the harness"));
global.window = global;

const html = fs.readFileSync("api/static/index.html", "utf8");
const script = html.match(/<script>([\s\S]*)<\/script>/)[1];
vm.runInThisContext(script);

const reportPath = process.argv[2] || "/scratch/work/report.json";
const report = JSON.parse(fs.readFileSync(reportPath, "utf8"));
renderVerdict(report);
renderUncertainty(report);
renderTools(report);
renderEvidence(report);
renderTrace(report);

const checks = {
  "verdict badge": elements["verdict-card"].innerHTML.includes(report.verdict),
  "six dimensions": (elements["dims"].innerHTML.match(/class="dim/g) || []).length === 6,
  "radar svg": elements["radar"].innerHTML.startsWith("<svg"),
  "tool timeline": elements["tools-body"].innerHTML.includes("timeline"),
  "evidence rows": (elements["evidence-body"].innerHTML.match(/<tr>/g) || []).length > 1,
  "trace steps": (elements["trace-body"].innerHTML.match(/class="step"/g) || []).length > 3,
};
for (const [name, ok] of Object.entries(checks)) console.log((ok ? "PASS " : "FAIL ") + name);
if (Object.values(checks).some((ok) => !ok)) process.exit(1);
