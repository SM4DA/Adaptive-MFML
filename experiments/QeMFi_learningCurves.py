import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from sklearn.utils import shuffle
from tqdm import tqdm
from qml.math import cho_solve
from qml.kernels import matern_kernel
from saturatedmfml.MFML_Model import ModelMFML

from saturatedmfml.dataloaders import generate_indexes, partition_QeMFi_data
from saturatedmfml.dataset_paths import RESULTS_SINGLE_PASS, results_dir
from saturatedmfml.experiment_params import get_hyperparams

def sf_LC(X_train, y_train, X_test, y_test, 
          sigma=100.0, reg=1e-10,
          navg=10):
    nmax = 12#np.int64(np.floor(np.log2(X_train.shape[0])))
    full_maes = np.zeros((nmax,navg),dtype=float)
    for n in tqdm(range(navg),desc='SF LC avg loop'):
        maes = []
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        #training size loop
        for i in tqdm(range(1,1+nmax),leave=False, desc='training size loop'):
            k_train = matern_kernel(X_train[:2**i],X_train[:2**i],sigma=20.0,order=1,metric='l2')
            k_train[np.diag_indices_from(k_train)] += 1e-12
            k_test = matern_kernel(X_train[:2**i],X_test,sigma=20.0,order=1,metric='l2')
            alphas = cho_solve(k_train,y_train[:2**i])
            preds_test = np.dot(alphas,k_test) 
            maes.append(np.mean(np.abs(preds_test-y_test)))
        maes = np.asarray(maes)
        full_maes[:,n] = np.copy(maes)
    return full_maes

def ordered_MFML(X_train, y_trains, index,
                 X_test, y_test, start_size, seed=42,
                 reg=1e-10, sigma=160.0):
    np.random.seed(seed)
    nfids = y_trains.shape[0]
    
    maes = []
    selected_inds = []
    s = start_size[0]
    # print(s)
    random_select = np.random.choice(index,size=s,replace=False)
    selected_inds.extend(random_select)
    
    X_train = X_train[selected_inds]
    # print(X_train.shape)
    energies = np.zeros((nfids),dtype=object)
    for i in range(nfids):
        energies[i] = np.copy(y_trains[i,selected_inds])

    MFML_indexes =  generate_indexes(nfids=4,size=s)

    model = ModelMFML(reg=reg, kernel='matern', 
                      sigma=sigma, order=1, metric='l2', 
                      gammas=None, p_bar=False)
    model.train(X_train_parent = X_train, fidelities=None, 
                y_trains=energies, indexes=MFML_indexes, 
                shuffle=False, 
                n_trains=start_size, seed=42)
    _ = model.predict(X_test = X_test, y_test = y_test, 
                      X_val = None, y_val = None, 
                      optimiser='default')
    # _ = model.predict(X_test = X_test, y_test = y_test, 
    #                   X_val = X_val, y_val = y_val, 
    #                   optimiser='OLS')
    # omfml_mae[i-1]+= np.copy(model.mae)
    # _ = model.predict(X_test=X_test,y_test=y_test,
    #                   X_val=X_val, y_val=y_val, optimiser='KRR', 
    #                   sigma=250.0, reg=1e-10, kernel_type='gaussian', 
    #                   order=1,metric='l2')
    # krr_mae[i-1] += np.copy(model.mae)
    # _ = model.predict(X_test=X_test,y_test=y_test,
    #                   X_val=X_val, y_val=y_val, optimiser='CompKRR', 
    #                   sigma=250.0, reg=1e-10, kernel_type='gaussian', order=1,metric='l2')
    # compkrr_mae[i-1] +=np.copy(model.mae)
    return model.mae

def Basic_MFML(X_train, y_trains, index, X_test, y_test, 
               reg:float=1e-10, sigma:float=20.0, scale:int=2, 
               X_val=None, y_val=None, 
               navg:int=10, nmax:int=9):
    
    mfml_mae = np.zeros((navg,nmax),dtype=float)
    # nfids = energies.shape[0]
    
    for n in tqdm(range(navg)):
        for i in tqdm(range(1,1+nmax),leave=False):
            train_sizes = np.asarray([scale**3,scale**2,scale,1])*2**i
            
            m = ordered_MFML(X_train,y_trains,index,X_test,y_test,
                             start_size=train_sizes,seed=n,reg=reg,sigma=sigma)
            
            mfml_mae[n,i-1] = np.copy(m)
    return mfml_mae


def saturated_MFML(X_train, y_trains, index, X_test, y_test,
                   reg:float=1e-10, sigma:float=20.0, 
                   X_val=None, y_val=None, 
                   navg:int=10, nstart:int=5, scale:int=2, window:int=10,
                   tol:float=1e-4, maxiter:int=1000, batch_size:int=32):
    # mfml_mae = np.zeros((navg,nmax),dtype=float)
    all_fidelity_maes = np.zeros((4,navg),dtype=object)
    all_fidelity_valmaes = np.zeros((4,navg),dtype=object)
    all_fidelity_diffs = np.zeros((4,navg),dtype=object)
    all_fidelity_trainsize = np.zeros((4,navg),dtype=object)
    #start_size = np.asarray([scale**3,scale**2,scale,1])*2**nstart
    start_size = np.asarray([9744,  104,   52,   26])
    # print('Starting Multifidelity Training Set Size: ',start_size)
    for n in tqdm(range(navg),leave=True,desc='Average loop for saturating MFML'):
        #start_size = np.asarray([scale**3,scale**2,scale,1])*2**nstart
        start_size = np.asarray([9744,  104,   52,   26])
        for f in tqdm(range(4),desc='Saturated MFML'):
            maes = []
            val_maes = []
            training_sizes = []
            if f>0:
                #select training set size determined from previous fidelity saturation
                start_size=np.copy(all_fidelity_trainsize[f-1,n][-1])
            else:
                pass
            training_sizes.append(np.copy(start_size))
            temp_mae = ordered_MFML(X_train, y_trains, index, X_test, y_test, 
                                    start_size, seed=n,reg=reg,sigma=sigma)
            v_mae = ordered_MFML(X_train, y_trains, index, X_val, y_val, 
                                    start_size, seed=n,reg=reg,sigma=sigma)
            
            maes.append(np.copy(temp_mae))
            val_maes.append(np.copy(v_mae))
            mae_diff = []
            mae_diff.append(np.copy(v_mae))
            addition = np.zeros((4),dtype=int)
            
            #start by adding training samples at the fidelity f
            for i in tqdm(range(maxiter),desc=f'Saturating at fidelity {f}',leave=True):
                addition[f] = batch_size[f]
                start_size+=addition
                
                ####break conditions for size consistency of lower fidelities
                if f>0 and start_size[f]>=start_size[f-1]*scale:
                    #if N_f approaches gamma*N_{f-1} then we are back to MFML regular.
                    #this condition will avoid it
                    print('terminated')
                    break
                # print(start_size)
                training_sizes.append(np.copy(start_size))
                # print(training_sizes)
                v_mae = ordered_MFML(X_train,y_trains,index,X_val,y_val,start_size,seed=n,reg=reg,sigma=sigma)
                temp_mae = ordered_MFML(X_train,y_trains,index,X_test,y_test,start_size,seed=n,reg=reg,sigma=sigma)
                maes.append(np.copy(temp_mae))
                val_maes.append(np.copy(v_mae))
                
                if i>window:
                    moving_avg = np.mean(val_maes[-window:])
                    mae_diff.append(np.copy(moving_avg))
                    value = np.abs(mae_diff[i-window]-mae_diff[i-1-window])
                    if value<tol[f]:
                        break
            all_fidelity_maes[f,n] = np.copy(maes)
            all_fidelity_valmaes[f,n] = np.copy(val_maes)
            all_fidelity_diffs[f,n] = np.copy(mae_diff)
            all_fidelity_trainsize[f,n] = np.copy(training_sizes)
        print(f'Final Multifidelity Training Set Sizes: {start_size}')
    return all_fidelity_maes, all_fidelity_valmaes, all_fidelity_trainsize, all_fidelity_diffs    

def main(prop='EV'):
    out = results_dir(RESULTS_SINGLE_PASS)
    reg, sigma = get_hyperparams('QeMFi', prop)
    print(f'Running for {prop} of QeMFi')
    #screen out data
    X_train, X_val, X_test, y_train, y_val, y_test = partition_QeMFi_data(prop=prop, seed=42, center=True)
    # indexes = generate_indexes(y_train.shape[1],y_train.shape[0])
    # np.save('QeMFi_indexes.npy',indexes,allow_pickle=True)
    indexes = np.load('EV_indexes.npy',allow_pickle=True)
    indexes=np.copy(indexes[0][:,0])
    ########sf krr
    # sf_mae = sf_LC(X_train, y_train[:,-1], 
    #                 X_test, y_test[:,-1], 
    #                 navg=10, 
    #             reg=6e-10, sigma=170.0)#sigma=1630.0, reg=1e-14)
    # np.save(f'{out}/QeMFi_{prop}_sf_mae.npy',sf_mae)

    # ########basic mfml
    # ntops = [9,9,8,5]
    # gammas = [2,3,5,10]
    # for s in range(4): 
    #      mfml_mae = Basic_MFML(X_train, y_train.T, indexes, X_test, y_test[:,-1], 
    #                            reg=6e-10, sigma=170.0,#sigma=1630.0, reg=1e-14,  
    #                            scale=gammas[s], 
    #                            X_val=None, y_val=None, 
    #                            navg=10, nmax=ntops[s])
    #      np.save(f'{out}/QeMFi_{prop}_mfml_mae_gamma{gammas[s]}.npy',mfml_mae)

    # # ########saturated MFML
    m,v,n,a = saturated_MFML(X_train, y_train.T, indexes, X_test, y_test[:,-1],
                           reg=reg, sigma=sigma,
                           X_val=X_val, y_val=y_val[:,-1], 
                           navg=1, nstart=1, scale=2,
                           tol=[1e-2,1e-3,1e-3,1e-3], 
                           maxiter=50, window=1, batch_size=[256,8,4,2])
    # np.save(f'{out}/4{prop}QeMFi_satMFML_mae.npy',m)
    # np.save(f'{out}/4{prop}QeMFi_satMFML_valmae.npy',v)
    # np.save(f'{out}/4{prop}QeMFi_satMFML_trainsizes.npy',n)
    # np.save(f'{out}/4{prop}QeMFi_satMFML_movingavg.npy',a)

if __name__=='__main__':
    #[1e-14, 1632.1941439039126, 'matern']
    #reg=6e-10, sigma170.0 - SCF
    main(prop='EV')
