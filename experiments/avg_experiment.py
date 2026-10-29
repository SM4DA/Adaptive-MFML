import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from saturatedmfml.MFMLManager import MFMLConfig, MFMLExperimentManager
from saturatedmfml.dataloaders import generate_indexes, get_dataset
from saturatedmfml.dataset_paths import RESULTS_AVERAGED, RESULTS_THRESHOLD, results_dir
from saturatedmfml.experiment_params import get_hyperparams, get_preset, get_sweep

def main(source='VIB5', name='CH3Cl', reg=None, sigma=None,
         global_tol=[1e-3], window=5, 
         local_tol=[1e-4, 1e-4, 1e-3, 1e-2], 
         initial_size=[64, 32, 16, 8], batch_size=[128,8,4,2], 
         seed:int = 42, locname=None, globname=None, val_size=2):
    if reg is None or sigma is None:
        reg, sigma = get_hyperparams('VIB5' if source.upper() == 'VIB5' else ('QeMFi' if source.upper() == 'QEMFI' else 'ANI'), name)
    # Initial Config
    config = MFMLConfig(
        reg=reg,
        sigma=sigma,
        scale=2,
        navg=1,  
        window=window, 
        global_tol=np.asarray(global_tol),
        local_tol=np.asarray(local_tol),
        max_passes=20,
        maxiter=100, 
        batch_size=batch_size,#[128, 8, 4, 2], 
        initial_size=initial_size,#[16, 8, 4, 2], 
        nmax_basic=9, 
        nmax_sf=11,
        seed=seed,
        relative=True
    )

    #Load Data
    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset(source, name, seed=42, center=True)
    # print(y_train.shape)
    #Load indices
    try:
        indexes = np.load(f'{name}_indexes.npy', allow_pickle=True)
        indexes = np.copy(indexes[0][:, 0])
    except FileNotFoundError:
        print(f"Indexes file '{name}_indexes.npy' not found! Generating.")
        indexes = generate_indexes(nfids=y_train.shape[1],size=y_train.shape[0])
        indexes = np.copy(indexes[0][:,0])
        #return

    # init manager
    manager = MFMLExperimentManager(
        X_train=X_train, 
        y_trains=y_train.T, 
        index=indexes, 
        X_val=X_val[:val_size,:], 
        y_val_target=y_val[:val_size, -1], 
        X_test=X_test, 
        y_test_target=y_test[:, -1], 
        config=config
    )

    out_threshold = results_dir(RESULTS_THRESHOLD)
    out_averaged = results_dir(RESULTS_AVERAGED)

    # Multi-Pass saturation
    m_glob, v_glob, n_glob, a_glob = manager.global_multipass_MFML_wrapper()
    
    np.save(f'{out_threshold}/{val_size}_{name}_globalpass_mae.npy', m_glob)
    np.save(f'{out_threshold}/{val_size}_{name}_globalpass_valmae.npy', v_glob)
    np.save(f'{out_threshold}/{val_size}_{name}_globalpass_trainsizes.npy', n_glob)
    np.save(f'{out_threshold}/{val_size}_{name}_globalpass_diffs.npy', a_glob)

    # # Cascading Saturation
    # m_casc, v_casc, n_casc, a_casc = manager.see_saw_saturation_MFML()

    # np.save(f'{out_averaged}/{seed}_{name}_cascading_mae.npy', m_casc)
    # np.save(f'{out_averaged}/{seed}_{name}_cascading_valmae.npy', v_casc)
    # np.save(f'{out_averaged}/{seed}_{name}_cascading_trainsizes.npy', n_casc)
    # np.save(f'{out_averaged}/{seed}_{name}_cascading_diffs.npy', a_casc)
    

if __name__=='__main__':
    preset = get_preset('vib5_threshold')
    for val_size in get_sweep('vib5_threshold_val_sizes'):
        print('validation set size: ', val_size, '\n')
        main(source='VIB5', name='CH3Cl', global_tol=preset['global_tol'],
             local_tol=preset['local_tol'], window=preset['window'],
             initial_size=preset['initial_size'], batch_size=preset['batch_size'],
             seed=0, val_size=val_size)
