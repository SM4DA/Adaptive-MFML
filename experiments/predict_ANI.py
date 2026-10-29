import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from qml.kernels import matern_kernel
from qml.math import cho_solve
from saturatedmfml.MFML_Model import ModelMFML
from saturatedmfml.dataloaders import get_dataset
from saturatedmfml.dataset_paths import RESULTS_AVERAGED, RESULTS_SINGLE_PASS, results_dir
from saturatedmfml.experiment_params import get_hyperparams
from sklearn.utils import shuffle

def KRR_SF(nmax = 11):
    X_train, _, X_test, y_train, _, y_test = get_dataset('ANI', 'ANI', seed=42, center=True)
    X_train,y_train = shuffle(X_train,y_train,random_state=42)
    K_train = matern_kernel(X_train[:2**nmax],X_train[:2**nmax],sigma=8e4,order=1,metric='l2')
    K_train[np.diag_indices_from(K_train)] += 2e-12
    alphas = cho_solve(K_train, y_train[:2**nmax,-1])
    del K_train
    K_test = matern_kernel(X_train[:2**nmax],X_test,sigma=8e4,order=1,metric='l2')
    del X_test
    predictions = np.dot(alphas,K_test)
    
    mae = np.mean(np.abs((predictions-y_test[:,-1])/y_test[:,-1]))
    print(f"FINAL TEST RMAE: {mae:.6f}")
    
    try:
        pred_dir = results_dir(RESULTS_SINGLE_PASS / 'final_predictions')
        np.save(f'{pred_dir}/ANI_SF_predictions.npy', predictions)
    except AttributeError:
        print("something went wrong.")

def get_final_sizes(trainsizes_file):
    """
    Helper function to extract the absolute final training sizes 
    reached at the end of the adaptive sampling run.
    """
    try:
        # Load the arrays for the first average run [:, 0]
        n_data = np.load(trainsizes_file)[50]#50 is where time is about 3hrs
    except FileNotFoundError:
        print(f"Could not find {trainsizes_file}; falling back to the default training sizes.")
        n_data = np.asarray([32,16,8,4])
            
    return n_data

def generate_mfml_indexes(nfids, size):
    """Generates the patched indices required by ModelMFML."""
    indexes = np.zeros((nfids), dtype=object)
    ordered_ind = np.arange(0, size)
    patched_ind = np.vstack([ordered_ind, ordered_ind]).T
    for i in range(nfids):
        indexes[i] = np.copy(patched_ind)
    return indexes

def train_and_predict_final_ani(strategy='cascading', prop='energy', seed=42):
    reg, sigma = get_hyperparams('ANI', 'ANI')
    
    file_path = f'{RESULTS_AVERAGED}/0_ANI_{strategy}_trainsizes.npy' 
    final_sizes = get_final_sizes(file_path)
    print(final_sizes)
    
    X_train, X_val, X_test, y_train, y_val, y_test = get_dataset('ANI', prop, seed=seed, center=True)
    
    y_trains = y_train.T 
    nfids = y_trains.shape[0]
    
    try:
        indexes = np.load(f'ANI_indexes.npy', allow_pickle=True)
        indexes = np.copy(indexes[0][:, 0])
    except FileNotFoundError:
        print(f"Indexes file not found! Generating simple sequential index array.")
        indexes = np.arange(X_train.shape[0])

    np.random.seed(seed)
    
    s = final_sizes[0]
    random_select = np.random.choice(indexes, size=s, replace=False)
    
    X_train_sub = X_train[random_select]
    energies = np.zeros((nfids), dtype=object)
    for i in range(nfids):
        energies[i] = np.copy(y_trains[i, random_select])

    mfml_indexes = generate_mfml_indexes(nfids, s)

    model = ModelMFML(reg=reg, kernel='matern',
                      sigma=sigma, order=1, metric='l2', gammas=None,
                      p_bar=True, relative=True)
                      
    model.train(X_train_parent=X_train_sub, fidelities=None, 
                y_trains=energies, indexes=mfml_indexes, 
                shuffle=True, n_trains=final_sizes, seed=0)
    
    y_test_target = y_test[:, -1] 
    
    predictions=model.predict(X_test=X_test, y_test=y_test_target, optimiser='default')
    
    print(f"FINAL TEST RMAE: {model.mae:.6f}")
    
    try:
        pred_dir = results_dir(RESULTS_SINGLE_PASS / 'final_predictions')
        np.save(f'{pred_dir}/ANI_{strategy}_predictions.npy', predictions)
        
        
    except AttributeError:
        print("Note: Could not find 'y_pred' attribute in ModelMFML to save raw predictions. Please check your class definition if you need the raw array!")
        

if __name__ == '__main__':
    train_and_predict_final_ani(strategy='globalpass', prop='energy')
    # KRR_SF(nmax=6)