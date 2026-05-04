import os
import numpy as np

from MFMLManager import MFMLConfig, MFMLExperimentManager
from dataloaders import get_dataset

def generate_indexes(nfids=4,size=1000):
    
    indexes = np.zeros((nfids),dtype=object)
    
    ordered_ind = np.arange(0,size)
    patched_ind = np.vstack([ordered_ind,ordered_ind]).T
    for i in range(nfids):
        indexes[i] = np.copy(patched_ind)
    return indexes

def main(source='VIB5', name='CH3Cl', reg=1e-1, 
         sigma=170.0, global_tol=[1e-3], window=5, 
         local_tol=[1e-4, 1e-4, 1e-3, 1e-2], 
         initial_size=[64, 32, 16, 8], batch_size=[128,8,4,2], 
         seed:int = 42):
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
        seed=seed
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
        X_val=X_val, 
        y_val_target=y_val[:, -1], 
        X_test=X_test, 
        y_test_target=y_test[:, -1], 
        config=config
    )

    # Multi-Pass saturation
    m_glob, v_glob, n_glob, a_glob = manager.global_multipass_MFML_wrapper()
    
    np.save(f'outputs_avg/{seed}_{name}_globalpass_mae.npy', m_glob)
    np.save(f'outputs_avg/{seed}_{name}_globalpass_valmae.npy', v_glob)
    np.save(f'outputs_avg/{seed}_{name}_globalpass_trainsizes.npy', n_glob)
    np.save(f'outputs_avg/{seed}_{name}_globalpass_diffs.npy', a_glob)

    # # Cascading Saturation
    m_casc, v_casc, n_casc, a_casc = manager.see_saw_saturation_MFML()

    np.save(f'outputs_avg/{seed}_{name}_cascading_mae.npy', m_casc)
    np.save(f'outputs_avg/{seed}_{name}_cascading_valmae.npy', v_casc)
    np.save(f'outputs_avg/{seed}_{name}_cascading_trainsizes.npy', n_casc)
    np.save(f'outputs_avg/{seed}_{name}_cascading_diffs.npy', a_casc)
    

if __name__=='__main__':
    os.makedirs('outputs_avg', exist_ok=True)
    #uncomment as needed
    for s in range(10):
        print('\nSeed: ',s,'\n')
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
        main(source='ANI',name='ANI', reg=2e-12, sigma=8e4, 
             global_tol=[2,2,2,2], local_tol=[1,1,1,1], 
             initial_size=[32,16,8,4],window=5, batch_size=[32,8,4,2], seed=s)