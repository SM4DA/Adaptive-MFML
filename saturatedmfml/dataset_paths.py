"""Central resolution of dataset and results locations for the adaptive-MFML
experiments.

All datasets live under ``DATASETS_ROOT`` (``~/Datasets`` by default).  Point
the scripts at another data layout with the ``SATMFML_DATASETS`` environment
variable instead of editing paths in individual files::

    SATMFML_DATASETS=/scratch/datasets python qemfi_experiments.py

Experiment outputs go under ``RESULTS_ROOT`` (``<repo>/results`` by default),
overridable with ``SATMFML_RESULTS``.  Each experiment has one subdirectory;
round-1 artefacts are kept under ``archive/round1``.
"""
import os
from pathlib import Path

DATASETS_ROOT = Path(os.environ.get('SATMFML_DATASETS', '~/Datasets')).expanduser()

VIB5_DIR = DATASETS_ROOT / 'VIB5'
VIB5_RAWDATA_DIR = VIB5_DIR / 'RAWDATA'

QEMFI_DIR = DATASETS_ROOT / 'QeMFi'
ANI_DIR = DATASETS_ROOT / 'ANI1x'

# The QeMFi CM descriptors were left behind when the datasets were moved.
_LEGACY_QEMFI_REPS = Path('~/2024/QeMFi/MFML/Reps').expanduser()


def _default_qemfi_reps() -> Path:
    """Prefer a copy inside the dataset; fall back to the pre-move location."""
    for candidate in (QEMFI_DIR / 'Reps', _LEGACY_QEMFI_REPS):
        if candidate.is_dir():
            return candidate
    return QEMFI_DIR / 'Reps'


# Override with SATMFML_QEMFI_REPS if the descriptors live elsewhere.
QEMFI_REPS_DIR = Path(
    os.environ.get('SATMFML_QEMFI_REPS', _default_qemfi_reps())
).expanduser()

REPO_ROOT = Path(__file__).resolve().parent.parent  # this file lives in saturatedmfml/
RESULTS_ROOT = Path(
    os.environ.get('SATMFML_RESULTS', REPO_ROOT / 'results')
).expanduser()

RESULTS_SINGLE_PASS = RESULTS_ROOT / 'single_pass'
RESULTS_AVERAGED = RESULTS_ROOT / 'averaged'
RESULTS_THRESHOLD = RESULTS_ROOT / 'threshold'
RESULTS_QEMFI = RESULTS_ROOT / 'qemfi_per_molecule'
RESULTS_HYPEROPT = RESULTS_ROOT / 'hyperopt'
RESULTS_ROBUSTNESS = RESULTS_ROOT / 'tolerance_robustness'
RESULTS_VAL32 = RESULTS_ROOT / 'validation32'
RESULTS_ARCHIVE = RESULTS_ROOT / 'archive' / 'round1'


def results_dir(path) -> Path:
    """Create (if needed) and return a results directory."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
