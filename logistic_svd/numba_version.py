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

    for i in simpletqdm.tqdm_dispatcher(n_iterations,verbose=verbose):
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


# @numba.njit(Tuple(float64,float64,float64,float64)(float64[:,:], 
#     float64[:,:],float64[:,:,:],
#     float64[:,:],float64[:,:,:],
#     uint8),parallel=True)
# def elbo(dmhalf,mua,Siga,muz,Sigz,approx):
#     kl_gau_alpha=0.0
#     kl_gau_z=0.0
#     kl_pg=0.0
#     l_data=0.0
#     Nc,Ng=dmhalf.shape
#     Nk=muz.shape[1]

#     # we will need to compute E[alpha[c]alpha[c]^T] for each gene...
#     prepalphass=np.zeros((Ng,Nk*Nk))
#     for g in range(Ng):
#         prepalphass[g]=(np.outer(alpha[c],alpha[c]) + Siga[c]).ravel() 

#     for c in numba.prange(Nc):
#         kl_gau_z += .5*np.trace(Sigz[c]) + .5*np.sum(muz[c]*muz[c])
#         kl_gau_z += -.5*np.linalg.slogdet(Sigz[c])[1] - .5*Nk

#     for g in numba.prange(Ng):
#         kl_gau_alpha += .5*np.trace(Siga[g]) + .5*np.sum(mua[c]*mua[c])
#         kl_gau_alpha += -.5*np.linalg.slogdet(Siga[g])[1] - .5*Nk

#     for c in numba.prange(Nc):
#         psi = (np.outer(muz[c],muz[c])+Sigz[c]).ravel() # size Nk*Nk
#         Elogits = muz[c] @ mualpha.T
#         Elogitsq = np.sum(prepalphass*psi[None],axis=1) # size Ng
#         sqElogitsq = np.sqrt(Elogitsq)
#         if approx: 
#             Mg = pge_safe_approx_II(sqElogitsq) # Nc
#         else:
#             Mg = pge_safe(sqElogitsq) # Nc

#         cancel = np.sum(np.log(np.cosh(Elogits/2)))
#         l_data += np.sum(Elogits*dmhalf) - np.log(2)*Nk*Ng - cancel
#         kl_pg += .5*np.sum(Elogitsq*(Mg-1))
#         kl_pg += np.sum(np.log(np.cosh(sqElogitsq/2))) - cancel

#     return kl_gau_alpha,kl_gau_z,kl_pg,l_data

# def update_z_variational(dmhalf,mua,Siga,muz,Sigz,approx):
#     return update_alpha_variational(dmhalf.T,muz,Sigz,mua,Sigz,approx)

# @numba.njit(Tuple(float64[:,:],float64[:,:,:])(float64[:,:], 
#     float64[:,:],float64[:,:,:],
#     float64[:,:],float64[:,:,:],
#     uint8))
# def update_alpha_variational(dmhalf,mua,Siga,muz,Sigz,approx):
#     '''
#     returns an updated posteriordistribution for alpha
#     '''

#     Nc,Ng=dmhalf.shape
#     Nk=muz.shape[1]

#     newalpha=np.zeros((Ng,Nk))

#     # we will need to compute E[Z[c]Z[c]^T] for each gene...
#     prepzs=np.zeros((Nc,Nk*Nk))
#     for c in range(Nc):
#         prepzs[c]=(np.outer(z[c],z[c]) + Sigz[c]).ravel() 

#     for g in range(Ng):
#         # compute sqrt(E[logit**2])
#         psi = (np.outer(mua[g],mua[g])+Siga[g]).ravel() # size Nk*Nk

#         sqElogitsq = np.sqrt(np.sum(prepzs*psi[None],axis=1)) # size Nc
#         if approx: 
#             Mg = pge_safe_approx_II(sqElogitsq) # Nc
#         else:
#             Mg = pge_safe(sqElogitsq) # Nc

#         # solve the equation
#         ztx = z.T @ dmhalf[:,g]
#         mtx = (Mg @ prepzs).reshape((Nk,Nk))+np.eye(Nk)
#         newalpha[g] = np.linalg.solve(mtx,ztx)

#     return newalpha 