import numpy as np

import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
import sys
import numbers

from . import simpletqdm

def initialize(data,rank,verbose=False):
    '''
    Finds a good initialization for logistic svd.

    Inputs:
        data -- Nc x Ng
        rank -- scalar

    Output:
        z     -- Nc x rank
        alpha -- Ng x rank

    ======
    Let 

        Lambda[c,g] = sum_{k} z[c,k] alpha[g,k]

    Consider the objective 

        L = sum_{cg} (data[c,g]-.5) Lambda[c,g] - log(2(cosh(Lambda[c,g]/2)))

    We find a reasonable initial condition for (z,alpha) using svd.
    '''

    simpletqdm.pnn('computing svd...',verbose=verbose)
    U,e,V = sp.sparse.linalg.svds(4*(data-.5),rank)
    z=U@np.sqrt(np.diag(e))
    alpha=(np.sqrt(np.diag(e))@V).T
    simpletqdm.pnn('computing done.',verbose=verbose)

    return z,alpha

def quadratic(zt,reg_mtx=None,reg_vec=None):
    '''
    Computes quadratics on zt

    Input:
        zt      -- Nbatch x Nk
        reg_mtx -- Nbatch x Nk x Nk  [OR]  Nk x Nk   [OR]  scalar  [OR]  None
        reg_vec -- Nbatch x Nk       [OR]  None

    Output:
        R -- Nbatch

    ====

    Let 

        R[c] = -.5 sum_{ij} mtx[c,i,j] z[c,i] z[c,j]
               + sum_{k} vec[c,k] c[c,k]

    We return R

    Extra notes:
    - If reg_mtx is input as a scalar, we take mtx[c] = eye(Nk)*float(reg_mtx) for each c.
    - If reg_mtx is input as a matrix, we take mtx[c] = reg_mtx for each c.
    '''

    Nbatch,Nk=zt.shape

    # if there's no regularization, return zeros
    if (reg_mtx is None) and (reg_vec is None):
        return np.zeros(Nk,dtype=zt.dtype)

    # otherwise...
    R=0.0

    if reg_mtx is not None:
        if isinstance(reg_mtx,numbers.Number):
            R=R-.5*reg_mtx*np.sum(zt**2,axis=1)
        else:
            assert isinstance(reg_mtx,np.ndarray)
            shp=reg_mtx.shape
            if shp==(Nk,Nk):
                intermediate = zt @ reg_mtx
                R=-.5*np.sum(intermediate*zt,axis=1)
            elif shp==(Nbatch,Nk,Nk):
                R=-.5*np.einsum('cij,ci,cj->c',reg_mtx,zt,zt)
            else:
                raise Exception(f"reg_mtx must be number or have shape {Nk}x{Nk} or {Nbatch}x{Nk}x{Nk}, not {shp}")

    if reg_vec is not None:
        assert reg_vec.shape==(Nbatch,Nk)
        R=R+np.sum(reg_vec*zt,axis=1)

    return R

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

def logistic_likelihood(data,z,alpha,verbose=False):
    l=0.0
    Nc,Ng=data.shape

    for c in simpletqdm.tqdm_dispatcher(Nc,verbose=verbose):
        logits = alpha@z[c]
        l+=np.sum(logits*(data[c]-.5)) - np.sum(np.log(2*np.cosh(logits/2)))

    return l

def update_alpha_onestep(dmhalfg,z,alphag,mtx,vec,prepzs,protoz=None,protoalpha=None):
    Nk=z.shape[1]

    logits = z@alphag  # Nc

    if protoz is None:
        # compute M for this gene for each cells
        Mg = pge_safe(logits) # Nc

        # solve the equation
        ztx = z.T @ dmhalfg + vec
        mtx = (Mg @ prepzs).reshape((Nk,Nk))+mtx
        return np.linalg.solve(mtx,ztx)

    else:
        protologits=protoz@protoalpha # Nc
        Mg = pge_safe(logits+protologits) # Nc
        ztx = z.T @ (dmhalfg-Mg*protologits) + vec
        mtx = (Mg @ prepzs).reshape((Nk,Nk))+mtx
        return np.linalg.solve(mtx,ztx)

class ConstantList:
    def __init__(self,val):
        self.val=val

    def __getitem__(self,g):
        return self.val

def update_alpha(data,z,alpha,reg_mtx=None,reg_vec=None,verbose=False,protoz=None,protoalpha=None):
    '''
    Improve quadratically regularized logistic objective.

    Inputs:
        tdata   -- Nc x Ng, uint8
        z       -- Nc x Nk
        alpha0  -- Ng x Nk
        reg_mtx -- Ng x Nk x Nk   [OR]  scalar  [OR]  None
        reg_vec -- Ng x Nk        [OR]  None
        binsize -- scalar
        protoz     -- Nc x Nm        [OR]  None
        protoalpha -- Ng x Nm        [OR]  None

    Output:
        alpha1 -- Ng x Nk

    ========

    Consider the objective 

    L(alpha) = sum_{cg} (data[c,g]-.5) Lambda[c,g] - log(2(cosh(Lambda[c,g]/2)))
                        -.5 sum_{cij} mtx[g,i,j] alpha[g,i] alpha[g,j]
                        + sum_{ck} vec[g,k] alpha[g,k]
    Lambda[c,g] = sum_{k} z[c,k] alpha[g,k] + sum_{m} protoz[c,m] protoalpha[g,m]

    Using an initial condition alpha0, we find a new value alpha1 such that 
    L(alpha1) >= L(alpha0).  We compute the new values of alpha1 
    in batches of size `binsize.'  So, for example, we first compute 
    alpha1[:binsize], then alpha1[binsize:binsize*2], and so-on.

    Extra notes:
    - If reg_mtx is input as scalar, we take reg_mtx[g] = eye(Nk)*float(reg_mtx) for each g.

    '''

    Nc,Ng=data.shape
    Nk=z.shape[1]


    # process matrix regularizer
    if reg_mtx is None:
        reg_mtx=ConstantList(0.0)
    elif isinstance(reg_mtx,numbers.Number):
        reg_mtx = ConstantList(np.eye(Nk,dtype=z.dtype)*float(reg_mtx))
    else:
        assert isinstance(reg_mtx,np.ndarray)
        assert reg_mtx.shape==(Ng,Nk,Nk)

    # process vector regularizer
    if reg_vec is None:
        reg_vec=ConstantList(0.0)
    else:
        assert isinstance(reg_mtx,np.ndarray)
        assert reg_mtx.shape==(Ng,Nk)

    # process protoalpha
    if protoalpha is None:
        protoalpha=ConstantList(None)

    # we will need to compute z[c]z[c]^T over and over again for each gene...
    prepzs=np.zeros((Nc,Nk*Nk))
    for c in range(Nc):
        prepzs[c]=np.outer(z[c],z[c]).ravel()

    # comptue the updates
    newalpha=np.zeros((Ng,Nk))
    for g in simpletqdm.tqdm_dispatcher(Ng,verbose=verbose):
        newalpha[g] = update_alpha_onestep(
            data[:,g]-.5,
            z,alpha[g],
            reg_mtx[g],
            reg_vec[g],
            prepzs,
            protoz,
            protoalpha[g]
        )

    return newalpha 

