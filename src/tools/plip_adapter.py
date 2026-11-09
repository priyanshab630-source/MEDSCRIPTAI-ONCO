"""PLIP interaction profiling, with the XML parsed correctly.

PLIP (Protein–Ligand Interaction Profiler) is the reference implementation for
this step, and it is what Discovery Studio's interaction view is usually compared
against. Two things matter when driving it from code:

  * it needs a **complex** — protein ATOM records plus the ligand as HETATM;
  * its report nests the interaction lists one level deeper than you expect:
    `bindingsite/interactions/hydrogen_bonds`, not `bindingsite/hydrogen_bonds`.

This module runs it and normalises the result into the same shape that
`src/tools/interactions.py` returns, so the API and the dashboard do not care
which engine produced the numbers.
"""

import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

SECTIONS = {
    "hydrogen_bonds": "hydrogen_bonds",
    "hydrophobic_contacts": "hydrophobic_interactions",
    "salt_bridges": "salt_bridges",
    "pi_stacks": "pi_stacks",
    "water_bridges": "water_bridges",
    "halogen_bonds": "halogen_bonds",
    "metal_complexes": "metal_complexes",
}


def available() -> bool:
    """PLIP usable as a library or as the `plip` CLI."""
    if shutil.which("plip"):
        return True
    try:
        import plip  # noqa: F401
        return True
    except ImportError:
        return False


def _text(element, *names, default=""):
    for name in names:
        value = element.findtext(name)
        if value not in (None, ""):
            return value.strip()
    return default


def _distance(element) -> float:
    for tag in ("dist", "distance", "dist_h-a", "dist_d-a"):
        value = element.findtext(tag)
        if value:
            try:
                return round(float(value), 2)
            except ValueError:
                continue
    return 0.0


def parse_report(xml_path: Path) -> dict:
    """Normalise a PLIP XML report into the project's interaction shape."""
    root = ET.parse(str(xml_path)).getroot()
    site = root.find("bindingsite")
    if site is None:
        return {"engine": "plip", "error": "no binding site in the report"}

    identifiers = site.find("identifiers")
    interactions = site.find("interactions")

    result = {
        "engine": "plip",
        "ligand": _text(identifiers, "longname", "hetid", default="ligand"),
        "smiles": _text(identifiers, "smiles"),
        "inchikey": _text(identifiers, "inchikey"),
        "binding_site_residues": [],
        "hydrogen_bonds": [],
        "hydrophobic_contacts": [],
        "salt_bridges": [],
        "pi_stacks": [],
        "water_bridges": [],
        "halogen_bonds": [],
        "counts": {},
    }

    residues = site.find("bs_residues")
    if residues is not None:
        for residue in list(residues):
            name = _text(residue, "restype")
            number = _text(residue, "resnr")
            if name and number:
                result["binding_site_residues"].append(f"{name}{number}")

    if interactions is None:
        result["error"] = "report has no interactions section"
        return result

    for key, tag in SECTIONS.items():
        node = interactions.find(tag)
        if node is None:
            continue
        for item in list(node):
            entry = {
                "residue": f"{_text(item, 'restype')}{_text(item, 'resnr')}",
                "distance": _distance(item),
            }
            donor = _text(item, "donortype")
            acceptor = _text(item, "acceptortype")
            if donor or acceptor:
                entry["donor"] = donor
                entry["acceptor"] = acceptor
            if _text(item, "sidechain"):
                entry["sidechain"] = _text(item, "sidechain") == "True"
            stack_type = _text(item, "stack_type")
            if stack_type:
                entry["stack_type"] = stack_type
            result[key].append(entry)

    for key in SECTIONS:
        result.setdefault(key, [])

    # PLIP 3 leaves bs_residues empty in some builds; derive the list from the
    # interactions instead, which is the same information.
    if not result["binding_site_residues"]:
        seen = []
        for key in SECTIONS:
            for entry in result[key]:
                residue = entry.get("residue", "")
                if residue and residue not in seen:
                    seen.append(residue)
        result["binding_site_residues"] = seen
    result["counts"] = {key: len(result[key]) for key in SECTIONS}
    result["counts"]["binding_site_residues"] = len(result["binding_site_residues"])
    return result


def analyse(complex_pdb: str | Path, workdir: str | Path = None) -> dict:
    """Run PLIP on a complex PDB and return the normalised report."""
    if not available():
        return {"engine": "plip", "available": False,
                "error": "plip is not installed (pip install plip)"}

    complex_path = Path(complex_pdb)
    output_dir = Path(workdir) if workdir else Path(tempfile.mkdtemp())
    output_dir.mkdir(parents=True, exist_ok=True)

    command = ["plip", "-f", str(complex_path), "-x", "-o", str(output_dir / "plip"),
               "--quiet"]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=600)
    if completed.returncode != 0:
        return {"engine": "plip", "available": True, "error":
                (completed.stderr or completed.stdout or "plip failed").strip()[:400]}

    reports = sorted(output_dir.rglob("*report.xml"))
    if not reports:
        return {"engine": "plip", "available": True, "error": "plip wrote no XML report"}

    result = parse_report(reports[-1])
    result["available"] = True
    result["report_xml"] = str(reports[-1])
    return result


def summary_line(report: dict) -> str:
    if report.get("error"):
        return f"PLIP: {report['error']}"
    counts = report["counts"]
    return (f"PLIP: {counts['hydrogen_bonds']} H-bonds, "
            f"{counts['hydrophobic_contacts']} hydrophobic, "
            f"{counts['pi_stacks']} pi-stacks, {counts['salt_bridges']} salt bridges "
            f"across {counts['binding_site_residues']} residues")
