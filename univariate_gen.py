'''
Univariate problems generation

This version is a simple generative procedure that tries to find coordinates to place
each new centroids. It should be a little easier to understand than `gen_multivariate`. 
'''
from collections import defaultdict
import numpy as np

def gen_univariate(n, d, lim = 2, sigma = 0.3, max_try = 10, verbose=False):
    '''
    * n: max pair of points
    * d: dim
    * lim: \mathcal{X} domain limit
    * sigma: minimum acceptable distance between contradicting centroids
    * max_try: this method is not elegant at all, we sample coordinates at random to
      find free coordinates. For a nicer and more exact implementation, use
      `gen_multivariate` with d=1, but at the cost of controlling lim.

    Note: in [-lim, lim] there can be a maximum of 2*lim / sigma points,
    else we will have projection on one dim with points to close to one another.
    '''
    def find_point(Xi, sigma, lim, max_try):
        i = 0
        sol = np.random.uniform(-lim, lim)
        while (np.abs(Xi - sol) < sigma).any() and (i < max_try):
            i += 1
            sol = np.random.uniform(-lim, lim)
        if (i == max_try):
            raise ValueError()
        return sol

    X = np.zeros((2 * n, d))
    Y = np.ones((2 * n))
    S = np.ones((2 * n))  # ground-truth selection
    Y[::2] = 0  # alternate pair label

    for i in range(n):
        if verbose: print('Creating centroids', i)
        sel = np.random.randint(d)  # the selection is uniformly sampled
        S[2*i:2*i+2] = sel

        # compute a position common in all orthogonal dim for the pair
        for j in range(d):
            if j != sel:
                # merged common coordinates
                try:
                    orth_pos = find_point(X[:2*i,j], sigma, lim, max_try)
                    X[2*i:2*i+2,j] = orth_pos
                except:
                    S[2*i:2*i+2] = -1
                    if verbose: print('-> could not assign a position for centroid', j)
                    break
            else:
                # separate coordinates
                try:
                    sep_pos_1 = find_point(X[1:2*i:2,j], sigma, lim, max_try)
                    X[2*i, j] = sep_pos_1
                    sep_pos_2 = find_point(X[:2*i+1:2,j], sigma, lim, max_try)
                    X[2*i+1, j] = sep_pos_2
                except:
                    S[2*i:2*i+2] = -1
                    if verbose: print('-> could not assign a position for centroid', j)
                    break
    return X[S >= 0], Y[S >= 0], S[S >= 0]
