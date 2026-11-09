"""The ChEMBL activity importer: filters, dedupes by compound, takes medians."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import import_chembl_activities as importer  # noqa: E402


def record(molecule, pchembl, relation="=", smiles="CCO", stype="IC50",
           validity=None, value=100.0):
    return {"molecule_chembl_id": molecule, "canonical_smiles": smiles,
            "standard_type": stype, "standard_relation": relation,
            "standard_value": value, "standard_units": "nM",
            "pchembl_value": pchembl, "data_validity_comment": validity}


def test_filters_each_reason_and_counts_it():
    rows, dropped = importer.clean([
        record("A", 7.0),
        record("B", None),                       # no pChEMBL
        record("C", 5.0, relation=">"),          # censored
        record("D", 6.0, validity="Potential author error"),
        record("E", 6.0, smiles=None),           # no structure
        record("F", 6.0, stype="Ki"),            # wrong type
    ])
    assert [r["molecule_chembl_id"] for r in rows] == ["A"]
    assert dropped == {"wrong_type": 1, "no_pchembl": 1, "censored": 1,
                       "flagged_invalid": 1, "no_smiles": 1}


def test_repeated_measurements_collapse_to_the_median():
    rows, _ = importer.clean([record("A", 6.0), record("A", 8.0), record("A", 7.0)])
    assert len(rows) == 1
    assert rows[0]["pchembl_value"] == 7.0
    assert rows[0]["n_measurements"] == 3


def test_accepts_an_object_wrapping_the_activities(tmp_path):
    path = tmp_path / "export.json"
    path.write_text('{"activities": [{"molecule_chembl_id": "A"}]}', encoding="utf-8")
    assert importer.load_activities(path) == [{"molecule_chembl_id": "A"}]
