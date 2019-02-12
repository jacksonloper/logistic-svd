import numpy as np

import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import sys

from . import simpletqdm

def pge_safe(x):
    x=np.abs(x)
    good=(x>.000001)
    rez=np.zeros(x.shape)

    # the bad stuff we use quadratic approx for
    rez[~good]=.25-0.020833333333333332*(x[~good]**2)

    # the other stuff we can do with tanh directly
    xgood=x[good]
    rez[good]=np.tanh(xgood/2)/(2*xgood)
    
    return rez

def pge_safe_approx(x):
    x=np.abs(x)
    good=(x>.000001)
    rez=np.zeros(x.shape)

    # the bad stuff we use quadratic approx for
    rez[~good]=.25-0.020833333333333332*(x[~good]**2)

    # the other stuff we can do with tanh directly
    xgood=x[good]
    rez[good]=(xgood+.3)/(4+2*xgood**2)
    
    return rez

def train(rank,n_iterations,binary_matrix=None,dmhalf=None,verbose=True,compute_all_likelihoods=True,
                        penalty=1.0):
    if dmhalf is None:
        dmhalf =binary_matrix-.5
    dmhalf=np.require(dmhalf,dtype=np.float64)

    # initialize with svd
    simpletqdm.pnn('computing svd...',verbose=verbose)
    U,e,V = sp.sparse.linalg.svds(4*dmhalf,rank)
    z=U@np.diag(e)
    alpha=V.T
    simpletqdm.pnn('computing done.',verbose=verbose)

    # keep track of how well we're doing
    likelihoods=[]
    if not compute_all_likelihoods:
        likelihoods.append(np.mean(likelihood(dmhalf,z,alpha,penalty)))

    # iterate training steps
    for i in simpletqdm.tqdm_dispatcher(n_iterations,verbose=verbose):
        if compute_all_likelihoods:
            likelihoods.append(np.mean(likelihood(dmhalf,z,alpha,penalty)))
        alpha=update_alpha(dmhalf,z,alpha,penalty)
        z=update_z(dmhalf,z,alpha,penalty)

    # calculate final likelihood
    likelihoods.append(np.mean(likelihood(dmhalf,z,alpha,penalty)))

    # likelihoods is an array not a list
    likelihoods=np.array(likelihoods)

    # return our conclusions
    return z,alpha,likelihoods

def likelihood(dmhalf,z,alpha,penalty,verbose=False):
    l=0.0
    Nc,Nk=dmhalf.shape

    for c in simpletqdm.tqdm_dispatcher(Nc,verbose=verbose):
        logits = alpha@z[c]
        l+=np.sum(logits*dmhalf[c]) - np.sum(np.log(2*np.cosh(logits/2)))

    l+=-.5*penalty*(np.sum(z**2)+np.sum(alpha**2))

    return l/(Nc*Nk)

def update_z(dmhalf,z,alpha,penalty,verbose=False,approx=True):
    return update_alpha(dmhalf.T,alpha,z,penalty,verbose=verbose,approx=approx)

def update_alpha_onestep(dmhalfg,z,alphag,penalty,prepzs,approx=True):
    Nk=z.shape[1]

    # compute M for this gene for each cells
    logits = z@alphag  # Nc
    if approx:
        Mg = pge_safe_approx(logits) # Nc
    else:
        Mg = pge_safe(logits) # Nc

    # solve the equation
    ztx = z.T @ dmhalfg
    mtx = (Mg @ prepzs).reshape((Nk,Nk))+np.eye(Nk)*penalty
    return np.linalg.solve(mtx,ztx)

def update_alpha(dmhalf,z,alpha,penalty,verbose=False,approx=True):
    '''
    returns an updated alpha
    '''

    Nc,Ng=dmhalf.shape
    Nk=z.shape[1]

    newalpha=np.zeros((Ng,Nk))

    # we will need to compute z[c]z[c]^T over and over again for each gene...
    prepzs=np.zeros((Nc,Nk*Nk))
    for c in range(Nc):
        prepzs[c]=np.outer(z[c],z[c]).ravel()

    for g in simpletqdm.tqdm_dispatcher(Ng,verbose=verbose):
        newalpha[g] = update_alpha_onestep(dmhalf[:,g],z,alpha[g],penalty,prepzs,approx=approx)

    return newalpha 

def calc_minorizer(logits):
    minorizer_M = pge_safe(logits)
    minorizer_k = .5*minorizer_M*logits**2 - np.log(2*np.cosh(logits/2))  

    return minorizer_M,minorizer_k
