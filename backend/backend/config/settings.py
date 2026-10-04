"""
config/settings.py — Chemins partagés
=====================================
Localise la racine du projet (dossier contenant notebooks/ et data/),
quelle que soit la profondeur du dossier backend.
"""

import os
from pathlib import Path


def _find_project_root() -> Path:
    if os.getenv("PROJECT_ROOT"):
        return Path(os.environ["PROJECT_ROOT"]).resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "notebooks").is_dir():
            return parent
    raise FileNotFoundError(
        "Racine du projet introuvable (aucun dossier parent ne contient 'notebooks/'). "
        "Définissez PROJECT_ROOT dans le .env."
    )


PROJECT_ROOT: Path = _find_project_root()
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
CLEAN_DATA_PATH: Path = PROJECT_ROOT / "data" / "processed" / "ntsb_clean_final.csv"
