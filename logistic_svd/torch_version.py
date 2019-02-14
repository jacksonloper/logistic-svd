import torch
import numpy as np
from . import simpletqdm
import numbers

assert int(torch.__version__.split(".")[0])>=1,f'Torch version is {torch.__version__}, but we need at least 1.0'

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
        return torch.zeros(Nk,device=zt.device,dtype=zt.dtype)

    # otherwise...
    R=0.0

    if reg_mtx is not None:
        if isinstance(reg_mtx,numbers.Number):
            R=R-.5*reg_mtx*torch.sum(zt**2,1)
        else:
            assert isinstance(reg_mtx,torch.Tensor)
            shp=reg_mtx.shape
            if shp==(Nk,Nk):
                intermediate = zt @ reg_mtx
                R=-.5*torch.sum(intermediate*zt,1)
            elif shp==(Nbatch,Nk,Nk):
                R=-.5*torch.einsum('cij,ci,cj->c',reg_mtx,zt,zt)
            else:
                raise Exception(f"reg_mtx must be number or have shape {Nk}x{Nk} or {Nbatch}x{Nk}x{Nk}, not {shp}")

    if reg_vec is not None:
        assert reg_vec.shape==(Nbatch,Nk)
        R=R+torch.sum(reg_vec*zt,1)

    return R


def logistic_likelihood(tdata,zt,alphat,binsize=5000,verbose=False):
    '''
    Returns the logistic likelihood 

    Inputs:
        tdata   -- Nc x Ng, uint8
        z       -- Nc x Nk
        alpha   -- Ng x Nk
        binsize -- scalar
        verbose -- bool

    Output:
        L-- scalar

    =====================

    Let 

        Gamma[c,g] = sum_{k} z[c,k] alpha[g,k]
        L = sum_{cg} (tdata[c,g]-.5) Gamma[c,g] - log(2(cosh(Gamma[c,g]/2)))

    This function computes L.  It does so in batches of size binsize.  So
    for example, it first computes the sum over c=[0:binsize], then c[binsize:binsize*2].
    It accumulates as it goes.

    Extra notes:
    - If you are running out of ram on your gpu, try making the binsize smaller

    '''

    l=0.0
    Nc,Ng=tdata.shape

    bins=np.r_[0:Nc:binsize,Nc]

    for i in simpletqdm.tqdm_dispatcher(len(bins)-1,verbose):
        dmhalf=tdata[bins[i]:bins[i+1]].double()-.5
        logits = zt[bins[i]:bins[i+1]]@alphat.t()
        subl=torch.sum(logits*dmhalf) - torch.sum(torch.log(2*torch.cosh(logits/2)))
        l+=subl.item()

    return l

def pge_safe(x):
    switch=torch.abs(x)<.00001
    
    A=.25-0.020833333333333332*(x**2)
    B=torch.tanh(x/2)/(2*x)
    
    return torch.where(switch,A,B)

def prepzs(z):
    '''
    Input: 
      z - Nc x Nk 
    
    Output:
      prepzs: Nc x Nk x Nk
    
    Defined by:
      prepzs[c] = outer(z[c],z[c])
      
    That is, 
      prepzs[c,i,j] = z[c,i]*z[c,j] 
    '''
    
    return torch.einsum('ci,cj->cij',z,z)
    
def update_alpha_inner(dmhalf,z,alpha,prepzs,mtx,vec):
    '''
    Improve quadratically regularized logistic objective.

    Inputs:
        dmhalf -- Nc x Ng
        z      -- Nc x Nk
        alpha0 -- Ng x Nk
        prepzs -- Nc x Nk x Nk
        mtx    -- Ng x Nk x Nk
        vec    -- Ng x Nk 

    Output:
        alpha1 -- Ng x Nk

    ========
    
    Let 

        Gamma[c,g] = sum_{k} z[c,k] alpha[g,k]

    and consider the objective 

        L(alpha) = sum_{cg} dmhalf[c,g] Gamma[c,g] - log(2(cosh(Gamma[c,g]/2)))
                        -.5 sum_{cij} mtx[g,i,j] alpha[g,i] alpha[g,j]
                        + sum_{ck} vec[g,k] alpha[g,k]
        
    Using an initial condition alpha0, we find a new value alpha1 such that 
    L(alpha1) >= L(alpha0).  To do so we assume we have precomputed

        prepzs[c,i,j] = z[c,i]z[c,j]

    NOTE: The regularizers mtx and vec need only be broadcastable to the appropriate shapes.
    For example, mtx could be 1 x Nk x Nk, that would be fine.




    '''
    
    Nc,Ng=dmhalf.shape
    Nk=z.shape[1]

    logits = z@ alpha.t() # Nc x Ng
    Mg = pge_safe(logits) # Nc x Ng

    '''

    For each cell, we need to compute
    ztx[g,k] = vec[g,k] + sum_c z[c,k] dmhalf[c,g]
    mtx[g,k1,k2] = mtx[g,k1,k2] + sum_c Mg[c,g] prepzs[c,k1,k2]

    We will feed gesv matrices of the form
    - B (Nc, Nk,1)
    - A (Nc, Nk,Nk)

    '''

    ztx = (vec+dmhalf.t() @ z)[:,:,None] # <-- g x k x 1
    mtx = mtx + torch.einsum('cg,cij->gij',Mg,prepzs) # <-- g x k x k
    
    rez,LU=torch.gesv(ztx,mtx)
    
    return rez[:,:,0]

def update_alpha(tdata,z,alpha,reg_mtx=None,reg_vec=None,verbose=False,binsize=5000):
    '''
    Improve quadratically regularized logistic objective.

    Inputs:
        tdata   -- Nc x Ng, uint8
        z       -- Nc x Nk
        alpha0  -- Ng x Nk
        reg_mtx -- Ng x Nk x Nk   [OR]  scalar  [OR]  None
        reg_vec -- Ng x Nk        [OR]  None
        binsize -- scalar

    Output:
        alpha1 -- Ng x Nk

    ========

    Consider the objective 

    L(alpha) = sum_{cg} (tdata[c,g]-.5) Gamma[c,g] - log(2(cosh(Gamma[c,g]/2)))
                        -.5 sum_{cij} mtx[g,i,j] alpha[g,i] alpha[g,j]
                        + sum_{ck} vec[g,k] alpha[g,k]
    Gamma[c,g] = sum_{k} z[c,k] alpha[g,k]

    Using an initial condition alpha0, we find a new value alpha1 such that 
    L(alpha1) >= L(alpha0).  We compute the new values of alpha1 
    in batches of size `binsize.'  So, for example, we first compute 
    alpha1[:binsize], then alpha1[binsize:binsize*2], and so-on.

    Extra notes:
    - If reg_mtx is input as scalar, we take reg_mtx[g] = eye(Nk)*float(reg_mtx) for each g.
    - If you are running out of ram on your gpu, try making the binsize smaller

    '''
    

    Nc,Ng=tdata.shape
    Nc,Nk=z.shape

    bins=np.r_[0:Ng:binsize,Ng]

    # process matrix regularizer
    if reg_mtx is None:
        reg_mtxs=[0.0]*(len(bins)-1)
    elif isinstance(reg_mtx,numbers.Number):
        reg_mtx = torch.eye(Nk,device=z.device,dtype=z.dtype)*float(reg_mtx)
        reg_mtx=reg_mtx[None] # <-- 1 x Nk x Nk
        reg_mtxs=[reg_mtx]*(len(bins)-1)
    else:
        assert isinstance(reg_mtx,torch.Tensor)
        assert reg_mtx.device==z.device
        assert reg_mtx.dtype==z.dtype
        assert reg_mtx.shape==(Ng,Nk,Nk)
        reg_mtxs=[reg_mtx[bins[i]:bins[i+1]] for i in range(len(bins)-1)]

    # process vector regularizer
    if reg_vec is None:
        reg_vecs=[0.0]*(len(bins)-1)
    else:
        assert isinstance(reg_vec,torch.Tensor)
        assert reg_vec.shape==(Ng,Nk)
        reg_vecs=[reg_vec[bins[i]:bins[i+1]] for i in range(len(bins)-1)]
    
    # prep the outer products of the zs
    prep=prepzs(z)
    
    # do the updates (binned)
    alphas=[]
    for i in simpletqdm.tqdm_dispatcher(len(bins)-1,verbose):
        dmhalf=tdata[:,bins[i]:bins[i+1]].double()-.5
        alphas.append(update_alpha_inner(dmhalf,z,alpha[bins[i]:bins[i+1]],prep,reg_mtxs[i],reg_vecs[i]))
      
    return torch.cat(alphas)