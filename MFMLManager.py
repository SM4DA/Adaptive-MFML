import numpy as np
from tqdm.auto import tqdm
from sklearn.utils import shuffle
from dataclasses import dataclass, field
from typing import List

from MFML_Model import ModelMFML
from qml.math import cho_solve
from qml.kernels import matern_kernel

@dataclass
class MFMLConfig:
    reg: float = 6e-10
    sigma: float = 170.0
    scale: int = 2
    navg: int = 1
    
    window: int = 1
    global_tol: float = 1e-5
    maxiter: int = 50
    max_passes: int = 5
    
    local_tol: List[float] = field(default_factory=lambda: [1e-5, 1e-10, 1e-10, 1e-10])
    batch_size: List[int] = field(default_factory=lambda: [128, 8, 4, 2])
    initial_size: List[int] = field(default_factory=lambda: [16, 8, 4, 2])
    
    # Specific to Basic MFML and SF
    nmax_basic: int = 9
    nmax_sf: int = 12,
    seed:int=42


class MFMLExperimentManager:
    def __init__(self, X_train, y_trains, index, X_val, y_val_target, X_test, y_test_target, config: MFMLConfig):
        self.X_train = X_train
        self.y_trains = y_trains
        self.index = index
        self.X_val = X_val
        self.y_val_target = y_val_target
        self.X_test = X_test
        self.y_test_target = y_test_target
        self.config = config
        
        
        self.nfids = self.y_trains.shape[0]
        self.model = None
        
        assert len(self.config.initial_size) == self.nfids, "Config initial_size must match nfids."
        assert len(self.config.batch_size) == self.nfids, "Config batch_size must match nfids."
        assert len(self.config.local_tol) == self.nfids, "Config local_tol must match nfids."

    def _generate_indexes(self, size):
        """
        Generates the patched indices required by ModelMFML.
        
        Parameters:
        -----------
        size : array-like
            An array containing the number of training samples for each fidelity level.
            Example: [800, 400, 200, 100]
        """
        size = np.asarray(size, dtype=int)
        indexes = np.zeros((self.nfids), dtype=object)
        
        for i in range(self.nfids):
            ordered_ind = np.arange(0, size[i])
            patched_ind = np.vstack([ordered_ind, ordered_ind]).T
            indexes[i] = np.copy(patched_ind)
            
        return indexes

    def _train(self, start_size, seed):
        np.random.seed(seed)
        
        s = start_size[0]
        random_select = np.random.choice(self.index, size=s, replace=False)
        
        X_train_sub = self.X_train[random_select]
        energies = np.zeros((self.nfids), dtype=object)
        for i in range(self.nfids):
            energies[i] = np.copy(self.y_trains[i, random_select[:start_size[i]]])
        
        mfml_indexes = self._generate_indexes(size=start_size)

        # Build and train MFML model
        self.model = ModelMFML(reg=self.config.reg, kernel='matern', 
                          sigma=self.config.sigma, order=1, metric='l2', 
                          gammas=None, p_bar=False)
                          
        self.model.train(X_train_parent=X_train_sub, fidelities=None, 
                    y_trains=energies, indexes=mfml_indexes, 
                    shuffle=False, n_trains=start_size, seed=seed)
        
        
    def _evaluate(self,mode='test'):
        sss = np.asarray([self.model.y_trains[i].shape[0] for i in range(self.model.y_trains.shape[0])])[:self.nfids]
        
        if mode == 'val':
            self.model.predict(X_test=self.X_val, y_test=self.y_val_target, optimiser='default')
        elif mode == 'test':
            self.model.predict(X_test=self.X_test, y_test=self.y_test_target, optimiser='default')
        else:
            raise ValueError("mode must be 'val' or 'test'")
            
        return self.model.mae, sss
    
    def sf_LC(self):
        target_fid = self.nfids - 1
        y_train_target = self.y_trains[target_fid]
        
        full_maes = np.zeros((self.config.nmax_sf, self.config.navg), dtype=float)
        
        for n in tqdm(range(self.config.navg), desc='SF LC Avg Loop'):
            maes = []
            X_shuffled, y_shuffled = shuffle(self.X_train, y_train_target, random_state=n)
            
            for i in tqdm(range(1, 1 + self.config.nmax_sf), leave=False, desc='SF Train Size Loop'):
                size = 2**i
                k_train = matern_kernel(X_shuffled[:size], X_shuffled[:size], sigma=self.config.sigma, order=1, metric='l2')
                k_train[np.diag_indices_from(k_train)] += self.config.reg
                
                k_test = matern_kernel(X_shuffled[:size], self.X_test, sigma=self.config.sigma, order=1, metric='l2')
                alphas = cho_solve(k_train, y_shuffled[:size])
                
                preds_test = np.dot(alphas, k_test) 
                maes.append(np.mean(np.abs(preds_test - self.y_test_target)))
                
            full_maes[:, n] = np.array(maes)
            
        return full_maes

    def basic_MFML(self):
        mfml_mae = np.zeros((self.config.navg, self.config.nmax_basic), dtype=float)
        ns = np.zeros((self.config.nmax_basic,4), dtype=float)
        scale_multipliers = [self.config.scale**(self.nfids - 1 - i) for i in range(self.nfids)]
        iterator1 = tqdm(range(self.config.navg), desc='Basic MFML Avg Loop')
        for n in iterator1:
            iterator2 = tqdm(range(1, 1 + self.config.nmax_basic), leave=False, desc='Basic MFML')
            for i in iterator2:
                # E.g., [8, 4, 2, 1] * 2^i
                train_sizes = np.asarray(scale_multipliers) * (2**i)
                self._train(start_size=train_sizes, seed=n)
                m,s = self._evaluate(mode='test')
                mfml_mae[n, i-1] = m
                ns[i-1,:] = s
                iterator2.set_postfix({'Training Sizes':f'{s}','Test MAE':f'{m:.4f}'})
        np.save('mfmlsize.npy',ns)
        return mfml_mae

    def saturated_MFML(self,start_size=None,f_start:int=0,f_end:int=None, seed:int=0):
        nfids = self.nfids
        all_maes = []
        all_valmaes = []
        all_diffs = []
        all_trainsize = []
        
        if type(f_end)==type(None):
            f_end = np.copy(nfids)
        if f_end<=f_start:
            assert f_end>f_start, f'f_end({f_end}) must be strictly larger than f_start({f_start})' 
        
        if f_start>0:
            assert type(start_size)!=type(None), 'start_size must be provided for f_start>0'
        if type(start_size)==type(None):
                start_size = np.copy(self.config.initial_size)
        iterator1 = tqdm(range(f_start,f_end),desc='Single-Pass Adaptive MFML',leave=False)
        for f in iterator1:
            if f_start==0 and f > 0:
                start_size = np.copy(all_trainsize[-1])
            all_trainsize.append(np.copy(start_size))
            
            # Baseline evaluations
            self._train(start_size=start_size, seed=seed)
            t_mae,_ = self._evaluate(mode='test')
            v_mae,_ = self._evaluate(mode='val')
            
            all_maes.append(t_mae)
            all_valmaes.append(v_mae)
            
            all_diffs.append(v_mae)
            
            iterator2 = tqdm(range(self.config.maxiter), desc=f'Saturating at fidelity {f}', leave=False)
            for i in iterator2:
                if f > 0:
                    if (start_size[f] + self.config.batch_size[f]) >= start_size[f-1]//self.config.scale:
                    # Break condition for size consistency of lower fidelities
                        break
                    if (start_size[f] + self.config.batch_size[f])==start_size[f-1]:
                        break
                start_size[f] += self.config.batch_size[f]
                all_trainsize.append(np.copy(start_size))
                
                self._train(start_size=start_size, seed=seed)
                v_mae,s = self._evaluate(mode='val')
                t_mae,s = self._evaluate( mode='test')
                
                all_maes.append(t_mae)
                all_valmaes.append(v_mae)
                
                iterator2.set_postfix({'Training Sizes':f'{s}','Val MAE': f'{v_mae:.4f}', 'Test MAE':f'{t_mae:.4f}'})
                # moving average check
                if i >= self.config.window: 
                    moving_avg = np.mean(all_valmaes[-self.config.window:])
                    all_diffs.append(moving_avg)
                    
                    # Compare the last two moving averages
                    if len(all_diffs) > 1:
                        value = np.abs(all_diffs[-1] - v_mae)
                        if value < self.config.local_tol[f]:
                            message = 'deltaError<tol'
                            break
                        else:
                            message = 'MaxIter'
            iterator1.set_postfix({'Sampling at Fidelity':f'{f}','Message':f'{message}'})
                            
        
        return all_maes, all_valmaes, all_trainsize, all_diffs
        
    
    def global_multipass_MFML_wrapper(self):
        navg, nfids = self.config.navg, self.nfids
        all_maes = []
        all_valmaes = []
        all_diffs = []
        all_trainsize = []

        start_size=np.copy(self.config.initial_size)
        
        #wrap the satMFML through a global loop
        iterator1 = tqdm(range(self.config.max_passes),desc='Adaptive MFML',leave=True, position=0)
        last_mae = 1e5
        for i in iterator1:
            m,v,n,d = self.saturated_MFML(start_size=start_size,
                                                  f_start=0, f_end=None, 
                                                  seed=self.config.seed)
            # print(n[-5:])
            all_maes.extend(m)
            all_valmaes.extend(v)
            all_trainsize.extend(n)
            all_diffs.extend(d)
            start_size = np.copy(n[-1])
            improvement = np.copy(last_mae-v[-1])
            iterator1.set_postfix({'Training Sizes':f'{start_size}','Global Improvement':f'{improvement:.4f}'})
            
            if improvement<self.config.global_tol[0]:
                break
            last_mae = np.copy(v[-1])
        return all_maes, all_valmaes, all_trainsize, all_diffs
    
    def see_saw_saturation_MFML(self):
        all_maes = []
        all_valmaes = []
        all_trainsize = []
        all_diffs = []
        
        # Start by saturating F0 and F1
        f_end = 2
        current_start_size = np.copy(self.config.initial_size)
        self._train(start_size=current_start_size,seed=self.config.seed) 
        last_global_mae,_ = self._evaluate(mode='val')

        count = 0
        max_run = 0
        with tqdm(desc='Ceiling-Adaptive MFML', leave=True) as pbar:
            while f_end <= self.nfids:
                #saturate from f_start to f_end
                m,v,t,d = self.saturated_MFML(start_size=current_start_size,
                                                      f_start=0, f_end=f_end, 
                                                      seed=self.config.seed)
                
                all_maes.extend(m)
                all_valmaes.extend(v)
                all_trainsize.extend(t)
                all_diffs.extend(d)
                
                current_mae = np.copy(v[-1])
                current_start_size = np.copy(t[-1])
                improvement = last_global_mae - current_mae
    
                #update progress bar
                pbar.set_postfix({'sampling up':f'{f_end}',
                                  'Sizes': f'{current_start_size}', 'Imprmnt': f'{improvement:.4f}'})
                pbar.update(1)
                
                if improvement < self.config.global_tol[f_end-1]:
                    f_end += 1 
                
                last_global_mae = np.copy(current_mae)
                # previous_imp = np.copy(improvement)
        return np.array(all_maes), np.array(all_valmaes), np.array(all_trainsize), np.array(all_diffs)



    #UNUSED
    def see_saw_recurring_MFML(self):
        all_maes = []
        all_valmaes = []
        all_trainsize = []
        all_diffs = []
        
        # Start by saturating F0 and F1
        f_end = 2
        current_start_size = np.copy(self.config.initial_size)
        self._train(start_size=current_start_size,seed=self.config.seed) 
        last_global_mae,_ = self._evaluate(mode='val')

        count = 0
        max_run = 0
        
        count = 0
        with tqdm(desc='Ceiling-Adaptive MFML', leave=True) as pbar:
            while True:
                m,v,t,d = self.saturated_MFML(start_size=current_start_size,
                                                      f_start=0, f_end=f_end, 
                                                      seed=self.config.seed)
                all_maes.extend(m)
                all_valmaes.extend(v)
                all_trainsize.extend(t)
                all_diffs.extend(d)
                
                current_mae = np.copy(v[-1])
                current_start_size = np.copy(t[-1])
                improvement = last_global_mae - current_mae
                
                if improvement < self.config.global_tol[f_end-1] or f_end==self.nfids:
                    f_end += 1 
                #rest f_end if we crpss nfids sp we can repeat this entire process.
                if f_end >self.nfids:
                    f_end = 2
                    count += 1
                #update progress bar
                pbar.set_postfix({'Repeats':f'{count}','sampling up':f'{f_end}',
                                  'Sizes': f'{current_start_size}', 'Imprmnt': f'{improvement:.4f}'})
                pbar.update(1)
                if count >=3:
                    break
                
                
                last_global_mae = np.copy(current_mae)
        return np.array(all_maes), np.array(all_valmaes), np.array(all_trainsize), np.array(all_diffs)

