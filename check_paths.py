#!/usr/bin/env python3
"""Verify that every dataset file used by the loaders exists.

Run before starting a long experiment:

    python check_paths.py

Exits with status 1 if anything is missing.  Override the dataset root with
``SATMFML_DATASETS`` (see ``dataset_paths.py``).
"""
import sys

from saturatedmfml.dataset_paths import (
    ANI_DIR, QEMFI_DIR, QEMFI_REPS_DIR, VIB5_DIR, VIB5_RAWDATA_DIR,
)

VIB5_MOLECULES = ['CH3Cl', 'CH3F']
VIB5_FIDELITIES = ['CCSD-T', 'HF-QZ', 'HF-TZ', 'MP2']

QEMFI_MOLECULES = ['urea', 'acrolein', 'alanine', 'sma', 'nitrophenol',
                   'urocanic', 'dmabn', 'thymine', 'o-hbdi']

ANI_FILES = [
    'ANI1x_50k_SLATM_features.npy',
    'ANI1x_multifidelity_50k_allfids.npz',
    'ANI1x_multifidelity_50k.npz',
    'ANI-1x-release.h5',
    'train_set.npz',
    'val_set.npz',
    'test_set.npz',
    'ANI_proxy_time_costs.npy',
]


def required_files():
    """Yield ``(path, description)`` for every file a loader reads."""
    for mol in VIB5_MOLECULES:
        yield VIB5_DIR / f'{mol}_CM.npy', f'VIB5 {mol} descriptors'
        for fid in VIB5_FIDELITIES:
            yield VIB5_RAWDATA_DIR / f'{mol}_{fid}.dat', f'VIB5 {mol} {fid}'

    for mol in QEMFI_MOLECULES:
        yield QEMFI_DIR / f'QeMFi_{mol}.npz', f'QeMFi {mol} properties'
        yield QEMFI_REPS_DIR / f'{mol}_CM.npy', f'QeMFi {mol} descriptors'

    for name in ANI_FILES:
        yield ANI_DIR / name, f'ANI1x {name}'


def check(verbose=True):
    missing = []
    for path, description in required_files():
        exists = path.exists()
        if verbose:
            print(f'{"OK  " if exists else "MISS"} {description:<32} {path}')
        if not exists:
            missing.append(path)
    return missing


def main():
    missing = check()
    if missing:
        print(f'\n{len(missing)} required file(s) missing.')
        return 1
    print('\nAll dataset files present.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
