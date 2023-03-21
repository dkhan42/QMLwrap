import numpy as np
from qml import kernels
from scipy.linalg import cho_solve
from sklearn.model_selection import KFold
from itertools import product
from scipy.spatial.distance import euclidean, cityblock
from scipy.stats import wasserstein_distance

def p_distance_scipy(X,Y,p=2):
    if p==1:
        return cityblock(X,Y)
    elif p==2:
        return euclidean(X,Y)
    elif p=='dot':
        return kernels.linear_kernel(X,Y)
    elif p=='wass':
        return wasserstein_distance(X,Y)

def p_distance(X,Y,p=2):
    if p==1:
        return kernels.distance.manhattan_distance(X,Y)
    elif p==2:
        return kernels.distance.l2_distance(X,Y)
    elif p=='dot':
        return kernels.linear_kernel(X,Y)


def KRR_global(X_train,Y_train,X_test,best_params,kernel='rbf',norm=2,dist1='na',dist2='na'):
    '''
    Returns the Kernel Ridge Regression based predictions for global representations for a variety of kernels. Available options are Linear, Polynomial, Gaussian, Laplacian,
    Rational Quadratic, Matern 3/2 and Matern 5/2 kernels. The L1 and L2 norms can be used with all of the kernels. The norms are calculated using the QML-code library.
    '''
    lam = best_params['lambda']
    params = best_params
    if kernel in ['linear','polynomial']:
        if type(dist1)==str:
            dist1=kernels.linear_kernel(X_train,X_train)
        K=covariance(dist1,kernel,params)
        K+=(np.eye(K.shape[0])*lam)
        try:
            L=np.linalg.cholesky(K)
        except:
            return 'Gram matrix is not PSD'
        else:
            try:
                alpha=cho_solve((L,True),Y_train)
            except:
                return 'Cholesky decomposition failed, check distance matrices'
            else:                    
                if type(dist2)==str:
                    dist2=kernels.linear_kernel(X_train,X_test)
                k=covariance(dist2,kernel,params)
                return np.dot(k.T,alpha)
    else:
        if type(dist1)==str:
            dist1=p_distance(X_train,X_train,p=norm)
        K=covariance(dist1,kernel,params)
        K+=(np.eye(K.shape[0])*lam)
        try:
            L=np.linalg.cholesky(K)
        except:
            return 'Gram matrix is not PSD'
        else:
            try:
                alpha=cho_solve((L,True),Y_train)
            except:
                return 'Cholesky decomposition failed, check distance matrices'
            else:
                if type(dist2)==str:
                    dist2=p_distance(X_train,X_test,p=norm)
                k=covariance(dist2,kernel,params)
                return np.dot(k.T,alpha)

def covariance(dist,kernel,params):
    if kernel=='linear':
        K=(params['sigma0']**2)+((params['sigma1']**2)*dist)
        return K
    elif kernel=='polynomial':
        K=((params['sigma0']**2)+((params['sigma1']**2)*dist))**params['order']
        return K
    elif kernel=='rbf':
        dist=dist/(params['length'])
        return np.exp(-(dist**2)/2)
    elif kernel=='laplacian':
        dist=dist/(params['length'])
        return np.exp(-dist)
    elif kernel=='matern1':
        dist=(3**0.5)*dist/(params['length'])
        return (1+dist)*np.exp(-dist)
    elif kernel=='matern2':
        dist1=(5**0.5)*dist/(params['length'])
        dist2=5*(dist**2)/(3*(params['length']**2))
        return (1+dist1+dist2)*np.exp(-dist)
    elif kernel=='rq':
        dist=(dist**2)/(2*params['alpha']*(params['length']**2))
        return (1+dist)**(-params['alpha'])

def GridSearchCV(X,Y,params,cv=4,kernel='rbf',norm=2,FCHL=False,local=False,q=None):
    """
    Performs a cross-validated grid search for hyper-parameter optimization of KRR models using global representations. The best hyperparameters
    and their cross-validated mean absolute error score is returned as a dictionary. These include the kernel hyper-parameters and the regularizer value.
    """
    kf=KFold(n_splits=cv)
    if FCHL==False:
        X_train,X_test=[],[]
        Y_train,Y_test=[],[]
        for train,test in kf.split(X):
            X_train.append(X[train])
            Y_train.append(Y[train])
            X_test.append(X[test])
            Y_test.append(Y[test])
    else:
        X_train,X_test=[],[]
        Q_train,Q_test=[],[]
        Y_train,Y_test=[],[]
        for train,test in kf.split(X):
            X_train.append(X[train])
            Y_train.append(Y[train])
            Q_train.append(q[train])
            X_test.append(X[test])
            Y_test.append(Y[test])
            Q_test.append(q[test])
    if FCHL==False:
        if kernel in ['linear','polynomial']:
            dist=[(kernels.linear_kernel(X_train[i],X_train[i]),
            kernels.linear_kernel(X_train[i],X_test[i])) for i in range(cv)]
        else:
            dist=[(p_distance(X_train[i],X_train[i],p=norm),
            p_distance(X_train[i],X_test[i],p=norm)) for i in range(cv)]
    if kernel in ['rbf','laplacian','matern1','matern2']:
        mae=np.inf
        for i,j in product(params['lambda'],params['length']):
            mae_new=[]
            for k in range(cv):
                if FCHL==False:
                    y=KRR_global(X_train[k],Y_train[k],X_test[k],{'length':j,'lambda':i},
                    kernel=kernel,norm=norm,dist1=dist[k][0],dist2=dist[k][1])
                else:
                    y=KRR_global(X_train[k],Y_train[k],X_test[k],{'length':j},i,FCHL=True,local=local,q1=Q_train[k],q2=Q_test[k])
                if type(y)==str:
                    score=np.inf
                else:
                    score=np.mean(np.abs(Y_test[k]-np.array(y)))
                mae_new.append(score)
            val=np.mean(mae_new)
            if val<mae:
                mae=val
                best_lambda=i
                best_length=j
        try:
            best={'mae':mae,'lambda':best_lambda,'length':best_length}
        except:
            best={'mae':mae,'lambda':'none','length':'none'}
    elif kernel=='rq':
        mae=np.inf
        for i,j,a in product(params['lambda'],params['length'],params['alpha']):
            mae_new=[]
            for k in range(cv):
                y=KRR_global(X_train[k],Y_train[k],X_test[k],{'length':j,'alpha':a},i,
                kernel=kernel,norm=norm,dist1=dist[k][0],dist2=dist[k][1])
                if type(y)==str:
                    score=np.inf
                else:
                    score=np.mean(np.abs(Y_test[k]-np.array(y)))
                mae_new.append(score)
            val=np.mean(mae_new)
            if val<mae:
                mae=val
                best_lambda=i
                best_length=j
                best_alpha=a
        best={'mae':mae,'lambda':best_lambda,'length':best_length,'alpha':best_alpha}
    elif kernel=='linear':
        mae=np.inf
        for i,j,a in product(params['lambda'],params['sigma0'],params['sigma1']):
            mae_new=[]
            for k in range(cv):
                y=KRR_global(X_train[k],Y_train[k],X_test[k],{'sigma0':j,'sigma1':a},i,
                kernel=kernel,norm=norm,dist1=dist[k][0],dist2=dist[k][1])
                if type(y)==str:
                    score=np.inf
                else:
                    score=np.mean(np.abs(Y_test[k]-np.array(y)))
                mae_new.append(score)
            val=np.mean(mae_new)
            if val<mae:
                mae=val
                best_lambda=i
                best_length=j
                best_alpha=a
        best={'mae':mae,'lambda':best_lambda,'sigma0':best_length,'sigma1':best_alpha}
    elif kernel=='polynomial':
        mae=np.inf
        for i,p,j,a in product(params['lambda'],params['order'],params['sigma0'],params['sigma1']):
            mae_new=[]
            for k in range(cv):
                y=KRR_global(X_train[k],Y_train[k],X_test[k],{'sigma0':j,'sigma1':a,'order':p},i,
                kernel=kernel,norm=norm,dist1=dist[k][0],dist2=dist[k][1])
                if type(y)==str:
                    score=np.inf
                else:
                    score=np.mean(np.abs(Y_test[k]-np.array(y)))
                mae_new.append(score)
            val=np.mean(mae_new)
            if val<mae:
                mae=val
                best_lambda=i
                best_length=j
                best_alpha=a
                best_order=p
        best={'mae':mae,'lambda':best_lambda,'sigma0':best_length,'sigma1':best_alpha,'order':best_order}
    return best

def KRR_indexing(K1,Y_train,index_train,index_test,lam):
    """
    Returns the KRR predictions when a precomputed kernel matrix for test+train set is applied. Requires the indices for the 
    training and test sets and the value of the regularizer.
    """
    K=K1[index_train][:,index_train]
    K+=(np.eye(K.shape[0])*lam)
    try:
        L=np.linalg.cholesky(K)
    except:
        return 'Gram matrix is not PSD'
    else:
        try:
            alpha=cho_solve((L,True),Y_train)
        except:
            return 'Cholesky decomposition failed, check distance matrices'
        else:
            k=K1[index_train][:,index_test]
            return np.dot(k.T,alpha)

def KRR_local(X_train,Q_train,Y_train,X_test,Q_test,best_params):
    """
    Returns the KRR predictions for local representations. Available options for the kernels are the local Gaussian and Laplacian kernels
     as implemented in the QML-code library.
    """
    sigma,lam = best_params['length'], best_params['lambda']
    K=kernels.get_local_symmetric_kernel(X_train,Q_train,[sigma])
    K+=(np.eye(K.shape[0])*lam)
    try:
            L=np.linalg.cholesky(K)
    except:
        return 'Gram matrix is not PSD'
    else:
        try:
            alpha=cho_solve((L,True),Y_train)
        except:
            return 'Cholesky decomposition failed, check distance matrices'
        else:
            k=kernels.get_local_kernels(X_train,X_test,Q_train,Q_test,[sigma]).T
            return np.dot(k.T,alpha)

def GridSearchCV_local(X,Q,Y,params,kernel='Gaussian',cv=4):
    """
    Performs a cross-validated grid search for hyper-parameter optimization of KRR models using local representations. The best hyperparameters
    and their cross-validated mean absolute error score is returned as a dictionary. These include the kernel hyper-parameters and the regularizer value.
    """
    kf=KFold(n_splits=cv)
    X_train,X_test=[],[]
    Q_train,Q_test=[],[]
    Y_train,Y_test=[],[]
    for train,test in kf.split(X):
        X_train.append(X[train])
        Y_train.append(Y[train])
        Q_train.append(Q[train])
        X_test.append(X[test])
        Y_test.append(Y[test])
        Q_test.append(Q[test])
    mae=np.inf
    #from tqdm import tqdm
    for i,j in list(product(params['lambda'],params['length'])):
        mae_new=[]
        for k in range(cv):
            y=KRR_local(X_train[k],Q_train[k],Y_train[k],X_test[k],Q_test[k],kernel,{'length':j,'lambda':i})
            if type(y)==str:
                score=np.inf
            else:
                score=np.mean(np.abs(Y_test[k]-np.array(y)))
            mae_new.append(score)
        val=np.mean(mae_new)
        if val<mae:
            mae=val
            best_lambda=i
            best_length=j
        try:
            best={'mae':mae,'lambda':best_lambda,'length':best_length}
        except:
            best={'mae':mae,'lambda':'none','length':'none'}
    return best

def KRR_soap(X_train,Y_train,X_test,kernel='average',metric='rbf',gamma=1,lam=1e-6,normalized=False):
    if kernel=='average':
        from dscribe.kernels import AverageKernel
        if metric=='linear':
            ker=AverageKernel(metric='linear')
        else:
            ker=AverageKernel(metric=metric,gamma=gamma)
    else:
        from dscribe.kernels import REMatchKernel
        if metric=='linear':
            ker=REMatchKernel(metric='linear')
        else:
            ker=REMatchKernel(metric=metric,gamma=gamma)
        if normalized==False:
            from sklearn.preprocessing import normalize
            X_train=np.array([normalize(arr) for arr in X_train])
            X_test=np.array([normalize(arr) for arr in X_test])
    K=ker.create(X_train)
    K+=(np.eye(K.shape[0])*lam)
    try:
            L=np.linalg.cholesky(K)
    except:
        return 'Gram matrix is not PSD'
    else:
        try:
            alpha=cho_solve((L,True),Y_train)
        except:
            return 'Cholesky decomposition failed, check distance matrices'
        else:
            k=ker.create(X_test,X_train)
            return np.dot(k,alpha)

def GridSearchCV_soap(X,Y,params,cv=4,kernel='average',metric='rbf',normalized=False):
    kf=KFold(n_splits=cv)
    X_train,X_test=[],[]
    Y_train,Y_test=[],[]
    for train,test in kf.split(X):
        X_train.append(X[train])
        Y_train.append(Y[train])
        X_test.append(X[test])
        Y_test.append(Y[test])
    mae=np.inf
    for i,j in product(params['lambda'],params['length']):
        mae_new=[]
        for k in range(cv):
            y=KRR_soap(X_train[k],Y_train[k],X_test[k],kernel=kernel,metric=metric,gamma=j,lam=i,normalized=normalized)
            if type(y)==str:
                score=np.inf
            else:
                score=np.mean(np.abs(Y_test[k]-np.array(y)))
            mae_new.append(score)
        val=np.mean(mae_new)
        if val<mae:
            mae=val
            best_lambda=i
            best_length=j
        try:
            best={'mae':mae,'lambda':best_lambda,'length':best_length}
        except:
            best={'mae':mae,'lambda':'none','length':'none'}
    return best

from sklearn import svm
def SVC_precomputed(k_train, y_train, k_test, lam=1e-10):
    k_train += np.eye(k_train.shape[0])*lam
    clf=svm.SVC(kernel = 'precomputed')
    clf.fit(k_train, y_train)
    return clf.predict(k_test.T)

def SVC_indexing(k, y_train, index_train, index_test, lam=1e-10):
    k_train = k[index_train][:, index_train]
    k_train += np.eye(k_train.shape[0])*lam
    clf = svm.SVC(kernel = 'precomputed')
    clf.fit(k_train, y_train)
    k_test = k[index_train][:, index_test]
    return clf.predict(k_test.T)

def clf_score(y_pred, y_true):
    #same_labels = np.sum((y_pred - y_true)==0)
    same_labels = np.sum(y_pred==y_true)
    return same_labels/len(y_true)

def GridSearch_SVC(k_list, lam_list, Y_train, index_train, index_test, Y_test, progress=False):
    score = 0
    from itertools import product
    if progress == True:
        from tqdm import tqdm
        for i, lam in tqdm(list(product(range(len(k_list)), lam_list))):
            y_pred = SVC_indexing(k_list[i], Y_train, index_train, index_test, lam=lam)
            jac = clf_score(y_pred, Y_test)
            #print(score)
            if jac > score:
                best = i, lam
                score = jac
    else:
        for i, lam in list(product(range(len(k_list)), lam_list)):
            y_pred = SVC_indexing(k_list[i], Y_train, index_train, index_test, lam=lam)
            jac = clf_score(y_pred, Y_test)
            if jac > score:
                best = i, lam
                score = jac
    return {'index': best[0], 'lambda': best[1]}

def GridSearch_KRC(k_list, lam_list, Y_train, index_train, index_test, Y_test, progress=False):
    score = -1
    from itertools import product
    if progress == True:
        from tqdm import tqdm
        for i, lam in tqdm(list(product(range(len(k_list)), lam_list))):
            y_pred = KRR_indexing(k_list[i], Y_train, index_train, index_test, lam=lam)
            if type(y_pred)==str:
                jac = 0
            else:
                jac = clf_score(np.sign(y_pred), Y_test)
            if jac > score:
                best = i, lam
                score = jac
    else:
        for i, lam in list(product(range(len(k_list)), lam_list)):
            y_pred = KRR_indexing(k_list[i], Y_train, index_train, index_test, lam=lam)
            if type(y_pred)==str:
                jac = 0
            else:
                jac = clf_score(np.sign(y_pred), Y_test)
            if jac > score:
                best = i, lam
                score = jac
    return {'index': best[0], 'lambda': best[1]}
