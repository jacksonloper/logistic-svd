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

@numba.vectorize([float64(float64)],nopython=True,fastmath=True)
def pge_safe(x):
    if np.abs(x)<.00001:
        return .25
    else:
        return np.tanh(x/2)/(2*x)

@numba.vectorize([float64(float64)],nopython=True,fastmath=True)
def pge_safe_approx_II(x):
    x=np.abs(x)
    if x<1.25:
        return .25-0.020833333333333332*(x**2)
    else:
        return (x+.3)/(4+2*x**2)

def simpletqdm(n,verbose=True):
    if verbose:

        print("out of %d: "%n)
        batch=int(n/20)+1
        for i in range(n):
            if (i%batch)==0:
                sys.stdout.write("%d "%i)
                sys.stdout.flush()
            yield i
        print("...done")
    else:
        for i in range(n):
            yield i

def train(rank,n_iterations,binary_matrix=None,dmhalf=None,verbose=True,approx=True,compute_all_likelihoods=True,
                        penalty=1.0):
    if dmhalf is None:
        dmhalf =binary_matrix-.5
    dmhalf=np.require(dmhalf,dtype=np.float64)

    U,e,V = sp.sparse.linalg.svds(4*dmhalf,rank)

    approx=approx*1

    z=U@np.diag(e)
    alpha=V.T
    logits=z@alpha.T

    likelihoods=[]
    if not compute_all_likelihoods:
        likelihoods.append(np.mean(likelihood(dmhalf,z,alpha,penalty)))

    for i in simpletqdm(n_iterations,verbose=verbose):
        if compute_all_likelihoods:
            likelihoods.append(np.mean(likelihood(dmhalf,z,alpha,penalty)))
        alpha=update_alpha(dmhalf,z,alpha,approx,penalty)
        z=update_z(dmhalf,z,alpha,approx,penalty)

    likelihoods.append(np.mean(likelihood(dmhalf,z,alpha,penalty)))

    likelihoods=np.array(likelihoods)

    return z,alpha,likelihoods

@numba.njit(float64(float64[:,:], float64[:,:],float64[:,:],float64),parallel=True)    
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

@numba.njit(float64[:,:](float64[:,:], float64[:,:],float64[:,:],uint8,float64))
def update_alpha(dmhalf,z,alpha,approx,penalty):
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

    for g in range(Ng):
        # compute M for this gene for each cells
        logits = z@alpha[g]  # Nc
        if approx: 
            Mg = pge_safe_approx_II(logits) # Nc
        else:
            Mg = pge_safe(logits) # Nc

        # solve the equation
        ztx = z.T @ dmhalf[:,g]
        mtx = (Mg @ prepzs).reshape((Nk,Nk))+np.eye(Nk)*penalty
        newalpha[g] = np.linalg.solve(mtx,ztx)

    return newalpha 


def calc_minorizer(logits):
    minorizer_M = pge_safe(logits)
    minorizer_k = .5*minorizer_M*logits**2 - np.log(2*np.cosh(logits/2))  

    return minorizer_M,minorizer_k