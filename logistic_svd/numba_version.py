import numpy as np

import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import sys

import numba
from numba import float64
from numba import int64
from numba import uint8
from numba.types import Tuple

from . import simpletqdm

'''
Todo.

1) Figure out what goes wrong with numba prange.  Possibly submit a bug report.

2) Debug the new variational code.

'''

@numba.vectorize([float64(float64)],nopython=True,fastmath=True)
def pge_safe(x):
    if np.abs(x)<.00001:
        return .25-0.020833333333333332*(x**2)
    else:
        return np.tanh(x/2)/(2*x)

@numba.vectorize([float64(float64)],nopython=True,fastmath=True)
def pge_safe_approx_II(x):
    x=np.abs(x)
    if x<1.25:
        return .25-0.020833333333333332*(x**2)
    else:
        return (x+.3)/(4+2*x**2)

def pge_safe_approx_np(x):
    x=np.abs(x)
    good=(x>.000001)
    rez=np.zeros(x.shape)

    # the bad stuff we use quadratic approx for
    rez[~good]=.25-0.020833333333333332*(x[~good]**2)

    # the other stuff we can do with tanh directly
    xgood=x[good]
    rez[good]=np.tanh(xgood/2)/(2*xgood)
    
    return rez


def train(rank,n_iterations,binary_matrix=None,dmhalf=None,verbose=True,approx=True,compute_all_likelihoods=True,
                        penalty=1.0):
    if dmhalf is None:
        dmhalf =binary_matrix-.5
    dmhalf=np.require(dmhalf,dtype=np.float64)

    # numba wants the approximation boolean as an integer
    approx=approx*1

    # initialize with svd
    simpletqdm.pnn('computing svd...',verbose=verbose)
    U,e,V = sp.sparse.linalg.svds(4*dmhalf,rank)
    z=U@np.diag(e)
    alpha=V.T
    logits=z@alpha.T
    simpletqdm.pnn('computing done.',verbose=verbose)

    # keep track of how well we're doing
    likelihoods=[]
    if not compute_all_likelihoods:
        likelihoods.append(np.mean(likelihood(dmhalf,z,alpha,penalty)))

    # iterate training steps
    for i in simpletqdm.tqdm_dispatcher(n_iterations,verbose=verbose):
        if compute_all_likelihoods:
            likelihoods.append(np.mean(likelihood(dmhalf,z,alpha,penalty)))
        alpha=update_alpha(dmhalf,z,alpha,approx,penalty)
        z=update_z(dmhalf,z,alpha,approx,penalty)

    # calculate final likelihood
    likelihoods.append(np.mean(likelihood(dmhalf,z,alpha,penalty)))

    # likelihoods is an array not a list
    likelihoods=np.array(likelihoods)

    # return our conclusions
    return z,alpha,likelihoods

@numba.njit(float64(float64[:,:], float64[:,:],float64[:,:],float64),fastmath=True)
def likelihood(dmhalf,z,alpha,penalty):
    l=0.0
    Nc,Nk=dmhalf.shape

    for c in numba.prange(Nc):
        logits = alpha@z[c]
        l+=np.sum(logits*dmhalf[c]) - np.sum(np.log(2*np.cosh(logits/2)))

    l+=-.5*penalty*(np.sum(z**2)+np.sum(alpha**2))

    return l/(Nc*Nk)


def update_z(dmhalf,z,alpha,approx,penalty):
    return update_alpha(dmhalf.T,alpha,z,approx,penalty)

@numba.njit(float64[:](float64[:], float64[:,:],float64[:],uint8,float64,float64[:,:]),fastmath=True)
def update_alpha_onestep(dmhalfg,z,alphag,approx,penalty,prepzs):
    Nk=z.shape[1]

    # compute M for this gene for each cells
    logits = z@alphag  # Nc
    if approx: 
        Mg = pge_safe_approx_II(logits) # Nc
    else:
        Mg = pge_safe(logits) # Nc

    # solve the equation
    ztx = z.T @ dmhalfg
    mtx = (Mg @ prepzs).reshape((Nk,Nk))+np.eye(Nk)*penalty
    return np.linalg.solve(mtx,ztx)

@numba.njit(float64[:,:](float64[:,:], float64[:,:],float64[:,:],uint8,float64),fastmath=True)
def update_alpha(dmhalf,z,alpha,approx,penalty):
    '''
    returns an updated alpha
    '''

    Nc,Ng=dmhalf.shape
    Nk=z.shape[1]

    newalpha=np.zeros((Ng,Nk))

    # we will need to compute z[c]z[c]^T over and over again for each gene...
    prepzs=np.zeros((Nc,Nk*Nk))
    for c in numba.prange(Nc):
        prepzs[c]=np.outer(z[c],z[c]).ravel()

    for g in numba.prange(Ng):
        newalpha[g] = update_alpha_onestep(dmhalf[:,g],z,alpha[g],approx,penalty,prepzs)

    return newalpha 


def calc_minorizer(logits):
    minorizer_M = pge_safe(logits)
    minorizer_k = .5*minorizer_M*logits**2 - np.log(2*np.cosh(logits/2))  

    return minorizer_M,minorizer_k
