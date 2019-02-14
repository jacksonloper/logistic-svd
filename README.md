# Objective

For any fixed matrix X with entries X_{cg} in {0,1} and any n>=1, let

    L = \sum_{c,g} ((X_{c,g}-.5)\left(\sum_k^n Z_{ck} \alpha_{gk}) - \log 2 \cosh .5\sum_k^n Z_{ck} \alpha_{gk})
    
We here consider the problem of maximizing L with respect to \alpha,z.  

We additionally consider the case that we would like to maximize a regularized objective.  Specifically, let 

    R^\alpha &= \sum_g-\frac{1}{2}\alpha_g ^T D^\alpha_g \alpha_g + \alpha_g^Td^\alpha_g
    R^z &= \sum_c-\frac{1}{2}z_c ^T D^z_c z_c + z_c^Td^z_c

where for each g we have D^\alpha_g is an n x n square matrix, d^\alpha is a n-vector, and likewise for D^z,d^z.  We can incorporate these regularizations by trying to maximize L+R^\alpha+R^z instead.

# What this code provides

1. z,\alpha <- logistic_svd.numpy_version.initialize(X).  Given X, uses SVD to give a reasonable initial estimate for $z,\alpha$.
1. \alpha' <- logistic_svd.numpy_version.update_alpha(X,z,\alpha,D^\alpha,d^\alpha).  Given X,D^\alpha,d^\alpha and an initial guess z,\alpha, this function calculates an improved estimate for \alpha', i.e. L(z,\alpha)+R^\alpha(\alpha) \leq L(z,\alpha')+R^{\alpha}(\alpha').  Note that, by the symmetry of this problem, this can be used to update z as well.  
1. \alpha' <- logistic_svd.torch_version.update_alpha(X,z,\alpha,D^\alpha,d^\alpha).  Same as above, but taking torch tensors as input instead of numpy arrays.
1. L <- logistic_svd.numpy_version.logistic_likelihood(X,z,\alpha).  Calculates the (unregularized) objective.
1. L <- logistic_svd.torch_version.logistic_likelihood(X,z,\alpha).  Same but for torch.
1. R <- logistic_svd.numpy_version.quadratic}(z,D^z,d^z).  Calculates a quadratic regularization.
1. R <- logistic_svd.torch_version.quadratic}(z,D^z,d^z).  Same but for torch.
