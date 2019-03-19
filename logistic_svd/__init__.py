import numbers
import numpy as np


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


def calc_minorizer(logits):
    minorizer_M = pge_safe(logits)
    minorizer_k = .5*minorizer_M*logits**2 - np.log(2*np.cosh(logits/2))  

    return minorizer_M,minorizer_k
