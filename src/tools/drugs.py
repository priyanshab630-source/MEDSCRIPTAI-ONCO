"""Seed compound table: name -> SMILES (+ which target it was designed for)."""

import json

import config

_CACHE = {}


def all_drugs() -> dict:
    if not _CACHE:
        with open(config.DATA_DIR / "drugs.json", encoding="utf-8") as handle:
            _CACHE.update(json.load(handle))
    return _CACHE


def get(name: str) -> dict:
    """Case-insensitive lookup, empty dict when unknown."""
    for key, value in all_drugs().items():
        if key.lower() == name.lower():
            return {"name": key, **value}
    return {}
