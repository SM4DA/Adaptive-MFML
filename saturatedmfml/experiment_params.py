"""Access to the hyperparameters and presets stored in ``params.json``.

Keeping these in one JSON file means the per-molecule regularisation/length
scales and the tolerance presets can be edited without touching the scripts.
"""
import json
from pathlib import Path

PARAMS_FILE = Path(__file__).resolve().parent.parent / 'params.json'


def load_params() -> dict:
    with open(PARAMS_FILE) as handle:
        return json.load(handle)


def get_hyperparams(dataset: str, name: str):
    """Return ``(reg, sigma)`` for a dataset and molecule/property."""
    params = load_params()
    try:
        entry = params[dataset][name]
    except KeyError:
        raise KeyError(
            f"No hyperparameters for {dataset!r}/{name!r} in {PARAMS_FILE.name}. "
            f"Available: { {k: list(v) for k, v in params.items() if isinstance(v, dict) and not k.startswith('_')} }"
        ) from None
    return entry['reg'], entry['sigma']


def get_preset(name: str) -> dict:
    return load_params()['presets'][name]


def get_sweep(name: str) -> dict:
    return load_params()['sweeps'][name]
