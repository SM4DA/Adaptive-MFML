import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from saturatedmfml.MFMLManager import MFMLConfig, MFMLExperimentManager
import pandas as pd

from saturatedmfml.dataloaders import generate_indexes, partition_QeMFi_data
from saturatedmfml.dataset_paths import RESULTS_HYPEROPT, RESULTS_QEMFI, results_dir

def main(molname='urea', reg=1e-1, 
         sigma=170.0, global_tol=[1e-3], window=5, 
         local_tol=[1e-4, 1e-4, 1e-3, 1e-2], 
         initial_size=[64, 32, 16, 8], batch_size=[128,8,4,2], 
         seed:int = 42):
    # Initial Config
    config = MFMLConfig(
        reg=reg,
        sigma=sigma,
        scale=2,
        navg=10,  
        window=window, 
        global_tol=np.asarray(global_tol),
        local_tol=np.asarray(local_tol),
        max_passes=20,
        maxiter=100, 
        batch_size=batch_size, 
        initial_size=initial_size,
        nmax_basic=9, 
        nmax_sf=9,
        seed=seed
    )

    out = results_dir(RESULTS_QEMFI)

    #Load Data
    X_train, X_val, X_test, y_train, y_val, y_test = partition_QeMFi_data(
        prop='EV', seed=42, center=True, molname=molname)
    
    try:
        indexes = np.load(f'qemfi_indexes.npy', allow_pickle=True)
        indexes = np.copy(indexes[0][:, 0])
    except FileNotFoundError:
        print(f"Indexes file 'qemfi_indexes.npy' not found! Generating with {y_train.shape[1]} and {y_train.shape[0]}.")
        indexes = generate_indexes(nfids=y_train.shape[1],size=y_train.shape[0])
        np.save('qemfi_indexes.npy',indexes, allow_pickle=True)
        indexes = np.copy(indexes[0][:,0])
        #return

    # init manager
    manager = MFMLExperimentManager(
        X_train=X_train, 
        y_trains=y_train.T, 
        index=indexes, 
        X_val=X_val, 
        y_val_target=y_val[:, -1], 
        X_test=X_test, 
        y_test_target=y_test[:, -1], 
        config=config
    )
    # #single fidelity
    # sf_mae = manager.sf_LC()
    # np.save(f'{out}/{molname}_sf_mae.npy', sf_mae)
    
    # # #Basic MFML
    # basic_mae, basic_sizes = manager.basic_MFML()
    # np.save(f'{out}/{molname}_{config.scale}_basic_mfml.npy', basic_mae)
    # np.save(f'{out}/{molname}_{config.scale}_basic_mfml_sizes.npy', basic_sizes)

    # Multi-Pass saturation
    m_glob, v_glob, n_glob, a_glob = manager.global_multipass_MFML_wrapper()
    
    np.save(f'{out}/{seed}_{molname}_globalpass_mae.npy', m_glob)
    np.save(f'{out}/{seed}_{molname}_globalpass_valmae.npy', v_glob)
    np.save(f'{out}/{seed}_{molname}_globalpass_trainsizes.npy', n_glob)
    np.save(f'{out}/{seed}_{molname}_globalpass_diffs.npy', a_glob)

    # # # Cascading Saturation
    # m_casc, v_casc, n_casc, a_casc = manager.see_saw_saturation_MFML()

    # np.save(f'{out}/{seed}_{molname}_cascading_mae.npy', m_casc)
    # np.save(f'{out}/{seed}_{molname}_cascading_valmae.npy', v_casc)
    # np.save(f'{out}/{seed}_{molname}_cascading_trainsizes.npy', n_casc)
    # np.save(f'{out}/{seed}_{molname}_cascading_diffs.npy', a_casc)
    

if __name__=='__main__':
    molecules = ['urea','acrolein','alanine','sma','nitrophenol',
                'urocanic','dmabn','thymine','o-hbdi']
    hyperopts =  pd.read_csv(f'{RESULTS_HYPEROPT}/matern_hyperopt_results_ex.csv')

    
    for i in range(9):
        
        reg, sig = hyperopts.loc[hyperopts['Molname'] == molecules[i], ['HP_1', 'HP_2']].values[0]
        ##################
        # print(molecules[i])
        # main(molname=molecules[i],reg=reg, 
        #      sigma=sig, 
        #      seed = 42)
        ##################
        for s in range(10):
            print(molecules[i],s)
            main(molname=molecules[i],reg=reg, 
                 sigma=sig, global_tol=[1e-1], 
                 local_tol=[1e-1, 1e-1, 1e-1, 1e-1], initial_size=[32,16,8,4], 
                 batch_size=[32, 4, 2, 2], window=5, 
                 seed = s)
        
    # for s in range(10):
    #     print('\nSeed: ',s,'\n')
        # main(source='VIB5', name='CH3Cl', reg=6e-10, sigma=170.0, global_tol=[0.1,0.1,0.1,0.1], 
        #      local_tol=[1e-2, 1e-2, 1e-2, 1e-2], window=10, 
        #      initial_size=[32, 16, 8, 4], batch_size=[32,8,4,2],seed=s)
        # main(source='VIB5', name='CH3F', reg=5e-11, sigma=50.0, global_tol=[0.1,0.1,0.1,0.1], 
        #      local_tol=[1e-3, 1e-3, 1e-3, 1e-3],window=10,initial_size=[32, 16, 8,4], batch_size=[32,8,4,2], seed=s)
        # ###multi-pass
        # main(source='QEMFI', name='SCF', 
        #      reg=6e-10, sigma=70.0, 
        #      global_tol=[0.5], local_tol=[1,1,2,2], initial_size=[32,16,8,4], 
        #      batch_size=[64, 16, 8, 2],
        #      window=5,seed=s)
        # ###see-saw
        # main(source='QEMFI', name='SCF', reg=6e-10, sigma=70.0, 
        #      global_tol=[2,2,0.5,0.5], local_tol=[1,1,1,1], initial_size=[32,16,8,4], 
        #      batch_size=[64, 16, 8, 2],window=5,seed=s)
        # ####see-saw
        # main(source='QEMFI', name='EV', reg=1e-14, sigma=1600.0, 
        #      global_tol=[1,1,1e-1,1e-1], local_tol=[1e-1, 1e-1, 1e-1, 1e-1], 
        #      initial_size=[32,16,8,4], batch_size=[32, 4, 2, 2], window=5, seed=s)
        # ###multipass
        # main(source='QEMFI', name='EV', reg=1e-14, sigma=1600.0, global_tol=[1e-1], 
        #      local_tol=[1e-1, 1e-1, 1e-1, 1e-1], initial_size=[32,16,8,4], 
        #      batch_size=[32, 4, 2, 2], window=5, seed=s) #window 5
        # main(source='ANI',name='ANI', reg=2e-12, sigma=8e4, 
        #      global_tol=[2,2,2,2], local_tol=[1,1,1,1], 
        #      initial_size=[32,16,8,4],window=5, batch_size=[32,8,4,2], seed=s)