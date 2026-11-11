"""Render the dashboard's Docking tab to a PNG, without a JavaScript browser.

The dashboard is a live single-page app; headless Chromium is not available in every
environment, so this script rebuilds the *same* markup the tab's JavaScript produces —
reusing the dashboard's own stylesheet — and screenshots it with wkhtmltoimage. The
result is what the tab looks like, generated from live API data rather than mocked.

    python scripts/ui_snapshot.py --out docs/screenshots/docking-tab.png
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from api.app import app  # noqa: E402
from src.tools import render_svg  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def dashboard_css() -> str:
    html = (ROOT / "api" / "static" / "index.html").read_text(encoding="utf-8")
    match = re.search(r"<style>(.*?)</style>", html, re.S)
    return match.group(1) if match else ""


def kpi(label: str, value: str, hint: str = "", size: str = "") -> str:
    style = f' style="font-size:{size}"' if size else ""
    return (f'<div class="kpi" style="margin-bottom:10px"><div class="box">'
            f'<div class="label">{label}</div><div class="value"{style}>{value}</div>'
            f'<div class="hint">{hint}</div></div></div>')


def build_html(drug: str = None, target: str = None, smiles: str = None) -> str:
    client = TestClient(app)
    if drug and target:
        # screenshot a real per-compound run rather than the shipped sample
        report = client.post("/analyze", json={"drug": drug, "target": target,
                                               "disease": "", "smiles": smiles or ""}).json()
        docking = report.get("docking") or {}
        if not docking.get("scene"):
            raise SystemExit(f"{drug} -> {target} produced no docked pose: "
                             f"{docking.get('reason', 'see the analysis trace')}")
        docking = {**docking, "poses": docking.get("poses", [[docking.get("best_affinity")]]),
                   "crystal_pose_score": None, "redocking_rmsd": None,
                   "own_analysis": docking.get("own_analysis", {}),
                   "plip": docking.get("plip", {})}
        geometry = docking["scene"]
        label = f"{drug} → {target}"
    else:
        docking = client.get("/docking").json()
        geometry = client.get("/docking/scene").json()
        label = "4AJL sample"
    svg = render_svg.render(geometry, width=900, height=560)

    plip = (docking.get("plip") or {}).get("counts", {})
    own = docking["own_analysis"]["counts"]

    def row(label, left, right):
        return (f"<tr><td>{label}</td><td class='mono'>{left}</td>"
                f"<td class='mono'>{right}</td></tr>")

    table = "".join([
        row("hydrogen bonds", plip.get("hydrogen_bonds", 0), own.get("hydrogen_bonds", 0)),
        row("hydrophobic", plip.get("hydrophobic_contacts", 0), own.get("hydrophobic_contacts", 0)),
        row("salt bridges", plip.get("salt_bridges", 0), own.get("salt_bridges", 0)),
        row("pi-stacks", plip.get("pi_stacks", 0), own.get("pi_stacks", 0)),
        row("binding-site residues", plip.get("binding_site_residues", 0),
            own.get("binding_site_residues", 0)),
    ])

    residues = ", ".join(docking["own_analysis"]["residues"][:14])

    return f"""<!doctype html><html><head><meta charset="utf-8">
<style>{dashboard_css()}
.snap{{max-width:1500px;margin:0 auto;padding:26px 30px 34px}}
.tabs{{display:flex;gap:8px;flex-wrap:wrap;margin:18px 0 20px}}
.tab{{padding:8px 16px;border:1px solid var(--line);border-radius:999px;font-size:13px;
color:var(--muted)}} .tab.on{{background:rgba(77,163,255,.14);border-color:#2f5d8a;color:#cfe4ff}}
</style></head><body><div class="snap">
  <div style="display:flex;align-items:baseline;gap:14px">
    <h1 style="margin:0;font-size:26px">MEDSCRIPTAI-ONCO</h1>
    <span class="tag">prototype console</span>
    <span class="tag" style="margin-left:auto">{docking.get('engine')}</span>
  </div>
  <div class="tabs">
    <span class="tab">Verdict</span><span class="tab">Uncertainty</span>
    <span class="tab">Tools (RL)</span><span class="tab">Evidence</span>
    <span class="tab on">Docking</span><span class="tab">Datasets</span>
    <span class="tab">Calibration</span><span class="tab">Compare</span>
  </div>

  <div class="card">
    <div class="row"><h3 style="margin:0">Docked pose</h3><div class="spacer"></div>
      <span class="tag">{docking.get('engine')}</span>
      {f'<span class="tag">redocking {docking["redocking_rmsd"]} Å</span>'
        if docking.get("redocking_rmsd") is not None else '<span class="tag">per-compound run</span>'}</div>
    <div style="display:flex;gap:18px;align-items:flex-start;margin-top:14px">
      <div style="flex:1.6;min-width:0">{svg}</div>
      <div style="flex:1;min-width:0">
        {kpi("best pose", f"{docking['poses'][0][0]} kcal/mol",
             "AutoDock Vina" + (f", exhaustiveness {docking['exhaustiveness']}"
                                if docking.get("exhaustiveness") else ""))}
        {kpi("crystal pose", f"{docking['crystal_pose_score']} kcal/mol",
             f"redocking {docking['redocking_rmsd']} Å RMSD")
         if docking.get("crystal_pose_score") is not None else ""}
        {kpi("binding site", f"{len(docking['own_analysis']['residues'])} residues",
             "within 5 Å of the ligand", size="17px")}
        <div class="note" style="margin-top:10px">drag to rotate · scroll to zoom ·
          ligand in yellow, interacting residues in green, hydrogen bonds dashed</div>
      </div>
    </div>
  </div>

  <div class="card" style="margin-top:16px">
    <div class="row"><h3 style="margin:0">Interactions</h3><div class="spacer"></div>
      <span class="tag">PLIP</span><span class="tag">built-in</span></div>
    <table style="margin-bottom:12px">
      <tr><th>interaction</th><th>PLIP</th><th>built-in</th></tr>{table}
    </table>
    <div class="note">PLIP applies angle-aware criteria; the built-in analyser uses plain
      heavy-atom cutoffs, so it reports more contacts. Both are shown.</div>
    <div class="note" style="margin-top:8px">Residues: {residues}</div>
  </div>
</div></body></html>"""


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Screenshot the Docking tab")
    parser.add_argument("--out", default="docs/screenshots/docking-tab.png")
    parser.add_argument("--width", type=int, default=1440)
    parser.add_argument("--drug", help="screenshot this compound instead of the sample")
    parser.add_argument("--target", help="its target (must have a receptor)")
    parser.add_argument("--smiles", help="optional SMILES override")
    args = parser.parse_args(argv)

    html_path = Path(args.out).with_suffix(".html")
    html_path.parent.mkdir(parents=True, exist_ok=True)
    html_path.write_text(build_html(args.drug, args.target, args.smiles), encoding="utf-8")

    subprocess.run(["wkhtmltoimage", "--width", str(args.width), "--quality", "92",
                    str(html_path), str(args.out)], check=True,
                   capture_output=True, timeout=180)
    print(f"wrote {args.out} ({Path(args.out).stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
