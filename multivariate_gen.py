'''
Generative process for tasks of any dimension.
'''
from collections import defaultdict
import numpy as np

from methods import bit_count

def gen_multivariate(n, d, max_sel_dim = 2, sigma = 0.25, prob_simplify = 0.2, mix_shape=True):
    '''
    Generates n shapes in d dimensions
    
    * max_sel_dim: maximum selection cardinality
    * sigma: min dist between points -> used to tune the Gaussian variances
    * prob_simplify: probability to delete some generated centroids to bring diversity
      in the shapes. For prob = 0., we generate hyper-cubes with binary labels (as an
      hypercube defines a bipartite graph, it's possible to do that.) 
    * mix_shape: a debugging parameter, use False to see all shapes nicely aligned
      along a \vec{(1, 1, ...)} line.
    '''
    assert max_sel_dim <= d
    X = []
    Y = []
    S = []
    n_points = 0
    shapes = []
    to_delete = []

    # 1 -- generate abstract problems
    used = [0 for i in range(d)]
    for i in range(n):
        # select a ground-truth solution for the shape
        sel_dim = np.random.randint(1, max_sel_dim + 1)
        sel = sorted(list(np.random.permutation(d)[:sel_dim]))

        # allocate some coordinates that do not collide with previous ones
        x_base = []
        for k in range(d):
            x_base.append(used[k])
            used[k] += 1
        x_separated = list(x_base)  # copy
        for k in sel:
            x_separated[k] = used[k]
            used[k] += 1

        # add a binary hypercube in sel_dim dimensions
        shapes.append([])
        for j in range(2 ** sel_dim):
            x = list(x_base)
            for i_k, k in enumerate(sel):
                if 1 << i_k & j > 0:
                    x[k] = x_separated[k]  # move point along dim k
            X.append(x)
            if bit_count(j) % 2 == 0:
                Y.append(1.)
            else:
                Y.append(0.)
            S.append(list(sel))  # don't forget to copy! else the removal fails
            shapes[-1].append(n_points)
            n_points += 1

        # delete hypercube points to create more complex selections
        for j in range(2 ** sel_dim):
            if np.random.random() < prob_simplify and bit_count(j) > 1:
                if shapes[-1][0] + j not in to_delete:
                    to_delete.append(shapes[-1][0] + j)
                for l in range(sel_dim):  # frees dependences for neighbors
                    neighbor = j ^ (1 << l)  # flip one dim to find neighbor
                    neighbor_id = shapes[-1][neighbor]
                    S[neighbor_id].remove(sel[l])
                    if len(S[neighbor_id]) == 0 and (neighbor_id not in to_delete):
                        to_delete.append(neighbor_id)  # don't keep isolated point

    # effectively delete points from all lists
    for point_to_del in sorted(to_delete)[::-1]:
        try:
            X.pop(point_to_del)
            Y.pop(point_to_del)
            S.pop(point_to_del)
        except:
            raise ValueError(point_to_del, sorted(to_delete))
    to_del_ind = 0
    for shape in shapes:  # remove from shapes
        while to_del_ind < len(to_delete) and to_delete[to_del_ind] in shape:
            shape.remove(to_delete[to_del_ind])
            to_del_ind += 1
        if to_del_ind >= len(to_delete):
            break
    acc_shape = 0
    for i, shape in enumerate(shapes):  # reorder points from 1 to n.
        l_shape = len(shape)
        shapes[i] = list(range(acc_shape, acc_shape + l_shape))
        acc_shape += l_shape

    # 2 -- translate into real points spaced by sigma
    real_coord = [np.linspace(0, sigma * (k_used - 1), k_used) for k_used in used]
    for k in range(d):
        real_coord[k] = real_coord[k] - np.mean(real_coord[k])
        if mix_shape:
            real_coord[k] = np.random.permutation(real_coord[k])
    X_real = np.zeros((len(X), d))
    for i, x in enumerate(X):
        for k in range(d):
            X_real[i,k] = real_coord[k][X[i][k]]
    Y_real = np.array(Y, dtype='float32')

    return X_real, Y_real, S, shapes
