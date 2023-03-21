import numpy as np
data = np.load('qm7_test.npz',allow_pickle=True)
charges, coords, energies = data['charges'], data['coords'], data['Y']

from qml.representations import generate_coulomb_matrix, generate_fchl_acsf
cm = np.array([generate_coulomb_matrix(q, r, size = 23) for q,r in zip(charges, coords)])
fchl = np.array([generate_fchl_acsf(q, r, elements = np.unique(np.concatenate(charges)), pad=23) for q,r in zip(charges,coords)])

train_idx, test_idx = range(1000), range(1000, 2000)

param_grid={'lambda':[1e-3,1e-6,1e-9,1e-10],
            'length':[10**i for i in range(-2,4)]} #hyper-parameter search grid

''' 
Global Kernel ridge regression
'''
from KRR import GridSearchCV, KRR_global

#Hyper-parameter optimization using cross-validated grid searcch
best_params = GridSearchCV(cm[train_idx],energies[train_idx], param_grid, 
                           kernel = 'laplacian', norm = 1, cv = 4) #4-fold cross-validated grid search, use kernel = 'gaussian', norm = 2 for the gaussian kernel

#performing kernel ridge regression using the best hyper-parameters found
preds = KRR_global(cm[train_idx], energies[train_idx], cm[test_idx], 
            best_params, kernel='laplacian', norm=1) #use kernel = 'gaussian', norm = 2 for the gaussian kernel


''' 
Local Kernel ridge regression
'''
from KRR import GridSearchCV_local, KRR_local

#Hyper-parameter optimization using cross-validated grid searcch
best_params = GridSearchCV_local(fchl[train_idx], charges[train_idx], energies[train_idx], 
                                 param_grid, cv = 4) #4-fold cross-validated grid search

#performing kernel ridge regression using the best hyper-parameters found
preds = KRR_local(fchl[train_idx], charges[train_idx], energies[train_idx], fchl[test_idx], charges[test_idx], 
            best_params)


''' 
Kernel ridge regression with a pre-computed kernel matrix
'''
from qml.kernels import get_local_symmetric_kernel
K = get_local_symmetric_kernel(fchl, charges, SIGMA= 1) #calculating the kernel for the entire dataset

from KRR import KRR_indexing
preds = KRR_indexing(K, energies[train_idx], train_idx, test_idx, lam = 1e-6)


''' 
Kernel ridge regression with a list of pre-computed kernel matrices with different sigma
'''
from qml.kernels import get_local_symmetric_kernels
K_list = get_local_symmetric_kernels(fchl, charges,SIGMAS=[10**i for i in range(-4,5)]) #calculating the kernel for the entire dataset

from KRR import KRR_indexing
y, maes = [], []

for K in K_list:
    preds = KRR_indexing(K, energies[train_idx], train_idx, test_idx, lam = 1e-6)
    error = np.mean(np.abs(preds - energies[test_idx]))
    y.append(preds)
    maes.append(error)

best_pred = y[np.argmin(maes)] #pick the prediction with the lowest MAE, i.e. the kernel with the best sigma
