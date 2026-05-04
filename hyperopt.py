import numpy as np
from sklearn.model_selection import KFold
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from scipy.interpolate import griddata
from skopt.utils import use_named_args
from sklearn.utils import shuffle
import qml.kernels as kernels
from qml.math import cho_solve
import matplotlib.pyplot as plt
from tqdm import tqdm
from skopt.callbacks import DeltaXStopper, DeltaYStopper
import pandas as pd
from sklearn.model_selection import train_test_split
import csv

def partition_QeMFi_data(prop='EV',seed=42,center=True):
    molnames = ['urea','acrolein','alanine','sma','nitrophenol',
                'urocanic','dmabn','thymine','o-hbdi']
    idx = np.arange(0,15000)
    idx = shuffle(idx,random_state=seed)
    X = np.load('/home/vvinod/2024/QeMFi/MFML/Reps/o-hbdi_CM.npy')
    #largest SLATM is for o-hbdi with 6438 features.
    X=np.zeros((135000,X.shape[1]),dtype=float)  
    #Rest will be padded to this size.
    energies = np.zeros((135000,5),dtype=float)
    
    start=0
    end=15000
    idx_names = np.zeros((135000),dtype=float)
    for i,m in enumerate(molnames):
        names = np.full(15000,i)
        idx_names[start:end] = np.copy(names)
        temp_X = np.load(f'/home/vvinod/2024/QeMFi/MFML/Reps//{m}_CM.npy')
        X[start:end,:temp_X.shape[-1]] = temp_X[idx,:]
        if prop=='EV':
            temp_data=np.load(f'/home/vvinod/2024/QeMFi/dataset/QeMFi_{m}.npz')['EV'][:,:,0]*0.0029
        elif prop=='SCF':
            temp_data=np.load(f'/home/vvinod/2024/QeMFi/dataset/QeMFi_{m}.npz')['SCF']*630
        energies[start:end,:] = temp_data[idx,:]
        
        #increment for next molecule
        start+= 15000
        end += 15000
    
    ####
    X_train, X_test, y_train, y_test = train_test_split(X, energies, 
                                                        train_size=0.9, random_state=seed)
    X_train,X_val,y_train,y_val = train_test_split(X_train, y_train,
                                                   train_size=0.85/0.9, random_state=seed)

    del energies, X
    if center==True:
        for i in range(5):
            hello_mean_here = np.mean(y_train[i])
            y_train[:,i] = y_train[:,i] - hello_mean_here
            y_test[:,i] = y_test[:,i] - hello_mean_here
            y_val[:,i] = y_val[:,i] - hello_mean_here
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    return X_train,X_val,X_test,y_train,y_val,y_test

def partition_VIB5_data(molname='CH3Cl',seed=42,center=True):
    X = np.load(f'/home/vvinod/2025/BigDatasets/VIB5/{molname}_CM.npy')
    energies = np.zeros((X.shape[0],4),dtype=float)
    fids = ['CCSD-T','HF-QZ','HF-TZ','MP2']
    for i in range(4):
        energies[:,i] = np.loadtxt(f'/home/vvinod/2025/BigDatasets/VIB5/RAWDATA/{molname}_{fids[i]}.dat')

    X_train, X_test, y_train, y_test = train_test_split(X,energies,train_size=0.9,random_state=seed)
    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train,train_size=0.85/0.9,random_state=seed)
    # print(y_train.shape,y_test.shape,y_val.shape)

    del energies, X
    if center==True:
        for i in range(4):
            hello_mean_here = np.mean(y_train[i])
            y_train[:,i] = y_train[:,i] - hello_mean_here
            y_test[:,i] = y_test[:,i] - hello_mean_here
            y_val[:,i] = y_val[:,i] - hello_mean_here
    ####
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"X_test:  {X_test.shape}, y_test:  {y_test.shape}")
    return X_train,X_val,X_test,y_train,y_val,y_test



###define global vars here
k_fold = 5 #number of cross vals to perform
seed=42 #random seed for reproducibility

#### here we define the hyper-parameter space to be sampled from
space = [
    Real(1e-14, 1e-6, "log-uniform", name='lamba'),
    Real(1, 1e5, "log-uniform", name='sigma'),
    Categorical(['matern','gaussian','laplacian'],transform='string',name='kernel')
    
]

total_calls=100 #number of runs for optimizer
n_initial=5 #initial random guesses for Bayesian optimizer

@use_named_args(space)
def objective(lamba, sigma,kernel):
    # To optimize kernel type, 'kernel' would be added to the 'space' and handled in this function.
    model = KRR(reg=lamba, kernel=kernel, sigma=sigma,
                order=1, metric='l2', gammas=None)

    # Perform K-Fold Cross-Validation
    kf = KFold(n_splits=k_fold, shuffle=True, random_state=seed)
    maes = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        m = model.train_predict(X_train, y_train, X_test, y_test)
        maes.append(m)
    maes = np.mean(maes)
    #update tqdm bar by 1 after each optimizer run
    if global_tqdm_bar:
        global_tqdm_bar.update(1)
    
    return maes

class KRR():
    def __init__(self,reg:float=1e-9,kernel:str='matern',sigma:float=700.0,order:int=1, metric:str='l2', gammas:np.ndarray=None):
        #init params
        self.reg = reg
        self.kernel = kernel
        self.sigma = sigma
        self.order = order
        self.metric = metric
        self.gammas = gammas
        #train params
        self.X_train = None
        self.y_train = None
        #model params
        self.mae = 0.0
        self.rmse = 0.0
    
    def kernel_generators(self, X1:np.ndarray, X2:np.ndarray = None):
        #Case for training kernel
        if isinstance(X2,type(None)):
            X2=np.copy(X1) #make X2 a copy of X1 if X2 is not specified
        #generating kernels
        if self.kernel=='sargan':
            K = kernels.sargan_kernel(X1, X2, self.sigma, self.gammas)
        elif self.kernel=='gaussian':
            K = kernels.gaussian_kernel(X1, X2, self.sigma)
        elif self.kernel=='laplacian':
            K = kernels.laplacian_kernel(X1, X2, self.sigma)
        elif self.kernel=='linear':
            K = kernels.linear_kernel(X1, X2)
        elif self.kernel=='matern':
            K = kernels.matern_kernel(X1, X2, sigma=self.sigma, 
                                      order=self.order, 
                                      metric=self.metric)
        else:
            K = None
        return K
        
    def train_predict(self,X_train,y_train,X_test,y_test):
        K_train = self.kernel_generators(X_train,X_train)
        K_train[np.diag_indices_from(K_train)] += self.reg
        alpha = cho_solve(K_train,y_train)

        del K_train

        k_test = self.kernel_generators(X_train,X_test)
        preds = np.dot(alpha,k_test)
        mae = np.mean(np.abs(preds-y_test))
        return mae

    def train_predict_vector(self,X_train,y_train,X_test,y_test):
        K_train = self.kernel_generators(X_train,X_train)
        K_train[np.diag_indices_from(K_train)] += self.reg
        alpha = np.linalg.solve(K_train,y_train)

        del K_train
        k_test = self.kernel_generators(X_test,X_train)
        preds = np.dot(k_test,alpha)
        mae = np.mean(np.abs(preds-y_test))
        return mae
        



results = {}


_, X, _, _, y, _ = partition_QeMFi_data(prop='SCF', seed=42, center=True)
y = np.copy(y[:,0])
global_tqdm_bar = tqdm(total=total_calls, desc="Optimizing Hyperparameters")
res_gp = gp_minimize(objective,                  # the objective function to minimize
                     space,                      # the search space
                     n_calls=total_calls,                 # number of iterations
                     n_initial_points=n_initial,         # number of random samples to start with
                     callback = [DeltaYStopper(1e-3),DeltaXStopper(1e-3)], #early stop
                     random_state=seed)             # random state for reproducibility

global_tqdm_bar.close()
results = {'hyperparameters': res_gp.x,
                   'mae': res_gp.fun}

np.save('QeMFi_SCF_opt.npy',results)





#function to plot the results
def plot_hyperparam_contour(output,num_grid_points=100,path='hyperopt.png'):
    param_names = [dim.name for dim in output.space.dimensions]
    lambda_name, sigma_name = param_names[0], param_names[1]
    # Extract evaluated points and their corresponding function values
    lambda_iters = [p[0] for p in output.x_iters]
    sigma_iters = [p[1] for p in output.x_iters]
    func_vals = output.func_vals
    # Transform actual evaluated points to log10 space for interpolation
    log_lambda_iters = np.log10(lambda_iters)
    log_sigma_iters = np.log10(sigma_iters)
    
    # Determine the range for each parameter from the space definition
    # Get log10 of the 50% bounds
    log_lambda_min = np.log10(0.5*output.space.dimensions[0].low)
    log_lambda_max = np.log10(1.5*output.space.dimensions[0].high)
    log_sigma_min = np.log10(0.5*output.space.dimensions[1].low)
    log_sigma_max = np.log10(1.5*output.space.dimensions[1].high)
    
    grid_log_lambda = np.linspace(log_lambda_min, log_lambda_max, num_grid_points)
    grid_log_sigma = np.linspace(log_sigma_min, log_sigma_max, num_grid_points)
    
    # Create the meshgrid for the contour plot using log-transformed coordinates
    LOG_LAMBDA_GRID, LOG_SIGMA_GRID = np.meshgrid(grid_log_lambda, grid_log_sigma)
    
    # Use griddata to interpolate the scattered log-transformed points onto the new log-grid
    Z_interpolated = griddata(
        points=(log_lambda_iters, log_sigma_iters), # Interpolate using log-transformed points
        values=func_vals,
        xi=(LOG_LAMBDA_GRID, LOG_SIGMA_GRID), # Interpolate onto the log-transformed grid
        method='cubic' 
        #rescale=False, fill_value=np.max(func_vals)
    )
    
    fig = plt.figure(figsize=(6, 5))
    contour = plt.contourf(10**LOG_LAMBDA_GRID, 10**LOG_SIGMA_GRID, Z_interpolated,
                           cmap='viridis_r', 
                           extend='both')
    # Contour lines - makes the plot look nicer imo...
    plt.contour(10**LOG_LAMBDA_GRID, 10**LOG_SIGMA_GRID, Z_interpolated,
                levels=contour.levels, colors='black', linewidths=0.5)
    
    #Color bar for MAEs
    cbar = plt.colorbar(contour)
    cbar.set_label('MAE')
    
    #evaluated points
    plt.scatter(lambda_iters, sigma_iters, c='white', s=20, edgecolors='black', alpha=0.8, label='Evaluated Points')
    
    #best point
    best_lambda, best_sigma = output.x[0], output.x[1]
    plt.scatter(best_lambda, best_sigma, c='red', marker='X', s=100, edgecolors='black', 
                label=f'Best ({best_lambda:.1E},{best_sigma:.2f},{output.fun:.4f})', zorder=5)
    
    plt.xscale('log')
    plt.yscale('log')
    # Add labels and title
    plt.xlabel('$\lambda$')
    plt.ylabel('$\sigma$')
    fig.legend()
    plt.tight_layout()
    plt.xscale('log')
    plt.yscale('log')

    plt.savefig(path,format='png',dpi=100,bbox_inches='tight')



plot_hyperparam_contour(res_gp,num_grid_points=300,
                            path='QeMFi_SCF_opt_find.png')
