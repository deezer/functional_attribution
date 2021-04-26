'''
Baseline univariate methods
'''
from collections import defaultdict
import numpy as np
from sklearn.linear_model import Ridge
from scipy.special import binom
import time


def binarise_subset(subset):
    ''' Set as list to set as binary number. '''
    bin_subset = 0
    for dim in subset:
        bin_subset += 1 << dim
    return bin_subset


def has_k_el(bin_subset, k):
    ''' Efficient implementation that checks if a subset has a maximum of
    k elements, with subset represented as a binary number.
    See "Hamming weight" for more details.

    Return -1 if false, else return the cardinality of the set. '''
    c = 0
    while bin_subset and c < k:
        bin_subset &= bin_subset - 1
        c += 1
    if bin_subset == 0:
        return c
    return -1

def bit_count(bin_subset):
    ''' Cardinality of the set corresponding to bin_subset. '''
    c = 0
    while bin_subset:
        bin_subset &= bin_subset - 1
        c += 1
    return c

def IOU(binset_a, binset_b):
    ''' Intersect-over-union of two sets. '''
    I = binset_a & binset_b
    U = binset_a | binset_b
    return bit_count(I) / bit_count(U)

def truncated_IOU(binset_truth, binset_predicted):
    ''' Returns interset-over-union only if we return too many features. '''
    if binset_predicted & binset_truth != binset_truth:
        return 0.
    I = binset_truth & binset_predicted
    U = binset_truth | binset_predicted
    return bit_count(I) / bit_count(U)


class Distribution:
    '''
    Utility that computes the variance and categories frequency.
    '''
    def __init__(self, data, continuous = False):
        assert len(data.shape) == 2
        d = data.shape[1]
        self.mean = np.mean(data, axis=0)
        self.var = np.var(data, axis=0)
        self.std = np.sqrt(self.var)
        self.centroids = data   # background_data, used by expected gradient
        self.hist = [{'n': 0, 'val': defaultdict(int)} for i in range(d)]  # d columns
        if not continuous:
            # can be slow to compute if too many points
            for x in data:
                for i in range(d):
                    self.hist[i]['n'] += 1
                    self.hist[i]['val'][x[i]] += 1
            for i in range(d):
                for val in self.hist[i]['val']:
                    self.hist[i]['val'][val] /= self.hist[i]['n']



class Explainer:
    def __init__(self, name):
        self.name = name

    @staticmethod
    def get_selection_from_feature(weights, level = 0.7, min_val = 1e-2, eps = 1e-16):
        '''
        Get attribution subsets from feature attribution as the sum of ratio
        explained by the biggest features relevance. Lower level selects
        sparser explanations. Minimum value: 1/n, with n the number of features.

            eg. for two features, we select only one if its relevance is higher
            than the ratio level, else we select the two.

            [-0.25412457, -0.06214613] -> {X1}              (level 75%)
            [ 0.19166127,  0.06710942] -> {X1,X2}           (level 75%)

        * min_val: if the weights are nowhere significant (ie. the attribution
          values imply a non-selection with threshold min_val), return the
          empty set. In real-data tasks, this avoids computing relevance in
          areas where the labels are purely stochastic.
          Beware to check that the used magnitude for min_val is relevant for
          non-selection (eg. 1e-2 may be usual for saliency methods computed
          on very smooth joint distributions).
        '''
        weights = np.abs(weights)
        if np.max(weights) < min_val:
            return []

        weights = weights / (np.sum(weights) + eps)
        biggest_weights = np.argsort(-weights)
        explained_var = 0.
        k = 0
        subset = []
        while explained_var <= level:
            explained_var += weights[biggest_weights[k]]
            subset.append(biggest_weights[k])
            k += 1

        return sorted(subset)


# ---------- UNIVARIATE METHODS ----------

class LIME(Explainer):
    '''
    LIME but shortened for our problems where all features have the same nature.
    And extended to categorical features without requiring integers values. We
    have the corespondency with the official method:

        explainer = l.lime_tabular.LimeTabularExplainer(X, feature_selection='none',
                discretize_continuous=False, sample_around_instance=True)
    <=>
        explainer = LIME(continuous = True)

    and,

        explainer = l.lime_tabular.LimeTabularExplainer(X, feature_selection='none',
                categorical_features=list(range(X.shape[1])), sample_around_instance=True)
    <=>
        explainer = LIME(continuous = False)
    '''
    def __init__(self, samples = 5000, continuous = False):
        super().__init__('LIME')
        self.samples = samples
        self.continuous = continuous

    def sample(self, x, dist):
        '''
        Consider points as discrete and categorical or continuous.

        * continuous treatment : normal perturbations, weighted with quad dist
          on normalised data ;
        * categorical treatment : {0, 1} if colum take the same value as x.
        '''
        if len(x.shape) != 1:
            x = x.reshape(-1)  # x is flat
        d = x.shape[0]

        if self.continuous:
            # -- continuous, normal sampling and proximity in space
            abstract_data = np.random.normal(0, 1, (self.samples + 1, d))
            abstract_data[0] = 0.
            data = x + abstract_data * dist.std.reshape((1, -1))
        else:
            # -- categorical, if value is different, have a 0 in the abstract space
            data = np.zeros((self.samples + 1, d))  # real data
            abstract_data = np.ones((self.samples + 1, d))  # interpretable space
            data[0,:] = x
            for i in range(d):
                values = list(dist.hist[i]['val'].keys())
                freqs = [dist.hist[i]['val'][val] for val in values]  # same order
                data[1:,i] = np.random.choice(values, size=self.samples, replace=True, p=freqs)
                abstract_data[1:,i] = (data[1:,i] == data[0,i]).astype(int)

        return data, abstract_data

    def explain(self, x0, f, dist):
        data, abstract_data = self.sample(x0, dist)
        Y = f(data) # uses the true py|x
        return self.regress(abstract_data, Y)

    def regress(self, X, Y):
        '''
        By construction, X is already normalised. Then perform ridge regression as LIME.

        I have no clue what they do with the intercept. In SHAP, it is
        constrained: g((0,..)) = E_X(f) and g((1,..)) = f(x).
        '''
        dist = np.sum(np.square(X - X[0:1,:]), axis=-1)
        weight = np.exp(-dist / 2)  # ignore reweighting by normal coefficient
        regressor = Ridge(alpha=1, fit_intercept=True)
        regressor.fit(X, Y, sample_weight = weight)
        return regressor.coef_ # , regressor.intercept_



class Shapley(Explainer):
    '''
    Implements exact shapley values computation.

    Different baseline mode :
    * 'function_mean': take mean point then apply f, as in KernelSHAP
    * 'mean_function': compute the true conditional mean, as classic shapley

    We assume a dimension sufficiently small to brute-force the shapley value.
    '''
    def __init__(self, mode='function_mean'):
        super().__init__('Shapley')
        self.mode = mode

    def explain(self, x0, f, dist):
        if len(x0.shape) != 1:
            x0 = x0.reshape(-1)  # x is flat
        d = len(x0)
        assert d < 20  # else the complexity explodes
        phis = np.zeros(d)

        if self.mode == 'function_mean':
            cached_weight = np.zeros((2 ** d))
            cached_eval = np.zeros((2 ** d))
            X_cache = np.zeros((2 ** d, d))
            X_cache[:] = dist.mean
            for subset in range(1, 2 ** d - 1):
                subset_ind = [k for k in range(d) if (((1 << k) & subset) > 0)]
                cached_weight[subset] = 1 / (d * binom(d-1, len(subset_ind)))
                X_cache[subset, subset_ind] = x0[subset_ind]
            X_cache[-1] = x0
            cached_eval[:] = f(X_cache)

            for i in range(d):
                phi = 0.
                for subset in range(2 ** d):
                    if ((1 << i) & subset) == 0:  # for all subsets without {i}
                        weight = cached_weight[subset]
                        Y_no_i = cached_eval[subset]
                        phi += -Y_no_i * weight
                    else:  # all corresponding subset with {i}
                        weight = cached_weight[subset - (1 << i)]
                        Y_with_i = cached_eval[subset]
                        phi += Y_with_i * weight
                phis[i] = phi

            del cached_weight, cached_eval

        elif self.mode == 'mean_function':
            # assumes the provided f is (x, S) \mapsto p(y=1|x_S)
            x0 = np.expand_dims(x0, 0)

            cached_weight = np.zeros((2 ** d))
            cached_marginals = np.zeros((2 ** d))
            for subset in range(2 ** d - 1):
                subset_ind = [k for k in range(d) if (((1<<k) & subset) > 0)]
                cached_weight[subset] = 1 / (d * binom(d-1, len(subset_ind)))
                cached_marginals[subset] = f(x0, subset_ind)
            cached_marginals[-1] = f(x0, list(range(d)))

            for i in range(d):
                phi = 0.
                for subset in range(2 ** d):
                    if ((1 << i) & subset) == 0:  # for all subsets without {i}
                        weight = cached_weight[subset]
                        Y_no_i = cached_marginals[subset]
                        phi += -Y_no_i * weight
                    else:  # all corresponding subset with {i}
                        weight = cached_weight[subset - (1 << i)]
                        Y_with_i = cached_marginals[subset]
                        phi += Y_with_i * weight
                phis[i] = phi

            del cached_weight, cached_marginals

        return phis



class Shapley_Sample(Explainer):
    ''' Sampled approximate version of Shapley. '''
    def __init__(self, mode='mean_function', samples = 100, use_exact = True):
        super().__init__('Shapley Sampled')
        self.mode = mode
        self.samples = samples
        self.use_exact = use_exact
        if use_exact:
            self.exact_shapley = Shapley(mode)

    def explain(self, x0, f, dist):
        '''
        f is f_marginal here, so our sampler is way more efficient than 2014Strumbelj
        '''
        if len(x0.shape) != 2:
            x0 = x0.reshape((1, -1))  # x is flat
        d = x0.shape[1]

        # if budget allows, use the exact computation.
        if self.use_exact and self.samples >= 2 ** d:
            return self.exact_shapley.explain(x0, f, dist)

        cached_marginals = {}
        phis = np.zeros(d)
        for k in range(self.samples):
            sigma = list(np.random.permutation(d))
            subset = 0
            for i in range(d):
                if subset in cached_marginals:
                    Y_no_i = cached_marginals[subset]
                else:
                    if self.mode == 'function_mean':
                        x_masked = np.expand_dims(np.copy(dist.mean), 0)
                        x_masked[:, sigma[:i]] = x0[:, sigma[:i]]
                        Y_no_i = f(x_masked)
                    else:  # mean_function
                        Y_no_i = f(x0, sigma[:i])

                    cached_marginals[subset] = Y_no_i
                subset += 1 << sigma[i]
                if subset in cached_marginals:
                    Y_with_i = cached_marginals[subset]
                else:
                    if self.mode == 'function_mean':
                        x_masked = np.expand_dims(np.copy(dist.mean), 0)
                        x_masked[:, sigma[:i]] = x0[:, sigma[:i]]
                        Y_with_i = f(x_masked)
                    else:
                        Y_with_i = f(x0, sigma[:i+1])
                    cached_marginals[subset] = Y_with_i
                phis[sigma[i]] += Y_with_i - Y_no_i
        phis = phis / self.samples
        del cached_marginals
        return phis



class GAM(Explainer):
    '''
    Uses the exact marginals.
    '''
    def __init__(self):
        super().__init__('GAM')

    def explain(self, x0, f_marginal, dist):
        if len(x0.shape) != 2:
            x0 = x0.reshape((1, -1))  # x is flat
        d = x0.shape[1]
        phis = np.zeros(d)
        f0 = f_marginal(x0, [])
        for i in range(d):
            phis[i] = f_marginal(x0, [i]) - f0
        return phis



class Grad(Explainer):
    def __init__(self):
        super().__init__('Gradient')

    def explain(self, x0, grad_f, dist):
        return grad_f(x0).reshape(-1)



class GradXInput(Explainer):
    def __init__(self):
        super().__init__('Gradient x Input')

    def explain(self, x0, grad_f, dist):
        return grad_f(x0).reshape(-1) * x0.reshape(-1)



class IntGrad(Explainer):
    '''
    Simple implementation of Integrated Gradient, we may be challenged on the
    choice of the baseline point that is set to the mean. But as we are dealing
    with synthetic data, it is hard to choose otherwise. Anyway, this paper is
    not about showing the flaws of this vanilla version of IntGrad, namely that it is
    very sensitive to the choice of baseline, this has been shown in previous papers.

    * m_steps: integral approximation points
    '''
    def __init__(self, m_steps = 20):
        super().__init__('Integrated Gradient')
        self.m_steps = m_steps

    @staticmethod
    def integral_approximation(gradients):
        ''' Affine bins approx '''
        grads = (gradients[:-1,:] + gradients[1:,:]) / 2.
        return np.mean(grads, axis=0)

    def integrated_gradients(self, baseline, point, grad_func):
        ''' We adapt the code from
        https://www.tensorflow.org/tutorials/interpretability/integrated_gradients
        '''
        alphas = np.linspace(start=0.0, stop=1.0, num=self.m_steps+1)
        path = np.expand_dims(baseline, 0) + np.expand_dims(point - baseline, 0) * np.expand_dims(alphas, 1)
        grad_path = grad_func(path)
        int_grad = self.integral_approximation(grad_path)
        integrated_gradients = (point - baseline) * int_grad

        return integrated_gradients

    def explain(self, x0, grad_func, dist):
        return self.integrated_gradients(
                dist.mean.reshape(-1),  # baseline point as mean value
                x0.reshape(-1),
                grad_func,
            )



class ExpectedGrad(IntGrad):
    '''
    Probably more fair to include this improvement on IntGrad. Also a
    reimplementation.
    '''
    def __init__(self, samples = 1000):
        super().__init__()
        self.name = 'Expected Gradient'
        self.samples = samples

    def explain(self, x0, grad_func, dist, std = 0.125):
        ''' E_{x',\alpha}[(x' - x)*dF/dx(x + \alpha(x' - x))] '''
        if len(x0.shape) != 2:
            x0 = x0.reshape((1, -1))
        n_data = self.samples // int(dist.centroids.shape[0]) + 1
        offset = np.random.randint(n_data * int(dist.centroids.shape[0]) - self.samples)
        X_background = np.tile(dist.centroids, (n_data, 1))[offset:offset+self.samples]
        X_background = X_background + np.random.normal(0, std, X_background.shape)
        alpha_background = np.random.uniform(0, 1, (self.samples, 1))
        exp_grad = (x0 - X_background) * grad_func(
                    X_background + alpha_background * (x0 - X_background)
                )
        return np.mean(exp_grad, axis=0)


# ---------- MULTIVARIATE METHODS ----------

class fANOVA(Explainer):
    '''
    Returns all marginal estimations.

    * residual: whether or not to substract children (yes: wfANOVA, no: GA^KM)
    '''
    def __init__(self, limit_order = -1, residual=False):
        super().__init__('fANOVA')
        self.limit_order = limit_order  # -1 for auto max dim
        self.residual = residual  # output should be decomposition or summed func?
        self.cache_subset = {}  # never flushed, always true
        self.cache_subset[0] = { 'l': [], 'children': [] }  # init with empty set
        self.cache_subset_ordering = []  # caches subsets to compute
        self.cache = {}  # caches fANOVA values

    def reset_cache(self):
        ''' We do not flush after each iteration. May be used by other methods. '''
        del self.cache
        self.cache = {}

    def reset_dim_cache(self):
        ''' To reset after each dim change. Do not clear before get_selection(). '''
        del self.cache_subset_ordering
        self.cache_subset_ordering = []
        self.reset_cache()  # just in case I forget

    def precompute_subset(self, d):
        ''' Cache subset bin correspondance and cardinal. '''
        for subset in range(2 ** d):
            if subset not in self.cache_subset:
                self.cache_subset[subset] = {'l': [k for k in range(d) if ((1<<k)&subset>0)]}

    def compute_subset_children(self, max_order):
        ''' Cache studied subset included children and topological ordering. '''
        for subset in self.cache_subset:
            if len(self.cache_subset[subset]['l']) <= max_order:
                self.cache_subset_ordering.append(subset)
                if 'children' not in self.cache_subset[subset]:
                    self.cache_subset[subset]['children'] = []
                    for child in self.cache_subset_ordering[:-1]:
                        if subset & child == child:  # inclusion
                            self.cache_subset[subset]['children'].append(child)

    def explain(self, x0, f_marginal, dist):
        if len(x0.shape) != 2:
            x0 = x0.reshape((1, -1))  # x is flat
        d = len(x0[0])
        max_order = min(self.limit_order, d)
        if max_order <= 0:
            max_order = d
        phis = {}

        if len(self.cache_subset_ordering) == 0:  # reset cache
            self.precompute_subset(d)
            self.compute_subset_children(max_order)

        for i, subset in enumerate(self.cache_subset_ordering):  # ordering => children all computed
            f_s = float(f_marginal(x0, self.cache_subset[subset]['l']))  # single point
            if self.residual:
                for child in self.cache_subset[subset]['children']:
                    f_s -= self.cache[child]  # additive model
            self.cache[subset] = f_s
            phis[subset] = f_s

        return phis


    @staticmethod
    def prob(x):
        '''
        Compute the probability that all points belong to one class (given the mean).
        That's the attribution measure we define for categorical tasks.
        '''
        if x > 0.5:
            return x
        return 1 - x


    def get_selection_from_subset(self, phi, limit_card=-1, threshold=0.85, infty = False):
        ''' For a multivariate model, find the subset with lowest cardinality
        such that its probability is higher than the threshold.

        * phi: conditional means, { int: float }
        * threshold: positive float, max required variance
        '''
        assert not self.residual

        max_card = limit_card
        if max_card < 0:
            max_card = len(self.cache_subset[max(list(phi.keys())) ]['l'])

        maxprob_subset = defaultdict(lambda: (0, 0.)) # [cardinality](bin, prob)
        maxprob_subset[max_card+1] = (-1, 1.)  # default if no subset found
        if infty: # can return no subset if found none
            candidate = max_card + 1
        else:  # return highest possible available subset
            candidate = max_card

        for bin_subset in phi:  # phi can contain cardinality limited subsets
            card = len(self.cache_subset[bin_subset]['l'])
            if card > min(candidate, max_card):  # max_card
                continue
            prob_subset = self.prob(phi[bin_subset])
            if prob_subset > maxprob_subset[card][1]:
                maxprob_subset[card] = (bin_subset, prob_subset)
                if prob_subset > threshold:
                    candidate = min(candidate, card)
        best_subset = maxprob_subset[candidate][0]

        return best_subset


    def get_selection_afchar(self, phi, threshold=0.85, mode='square'):
        '''
        Idea from afchar2020making: uses a custom function to weight subsets probs.
        and explicitly define the selection mechanism via boosting.
        '''
        assert not self.residual

        max_card = len(self.cache_subset[max(list(phi.keys())) ]['l'])
        candidate = int(max_card)  # copy

        g_subsets = {}
        max_alpha = 0.
        max_alpha_subset = -1

        for bin_subset in phi:  # phi can contain cardinality limited subsets
            card = len(self.cache_subset[bin_subset]['l'])
            if card > candidate:  # max_card
                continue

            if mode == 'square':
                g_subset = 4 * (phi[bin_subset] - 0.5) ** 2
            elif mode == 'abs':
                g_subset = 2 * abs(phi[bin_subset] - 0.5)
            else:
                raise ValueError('Invalid mode in parameters:', mode)
            g_subsets[bin_subset] = g_subset
            alpha_subset = g_subset

            if 'children' not in self.cache_subset[bin_subset]:  # can happen on rerun
                self.compute_subset_children(max_card)

            for child in self.cache_subset[bin_subset]['children']:  # boosting
                alpha_subset *= (1 - g_subsets[child])

            if alpha_subset > max_alpha:
                max_alpha = alpha_subset
                max_alpha_subset = bin_subset
                if alpha_subset > threshold:
                    candidate = card

        return max_alpha_subset



class Archipelago(Explainer):
    '''
    Implements the tsang2019ArchDetect that makes the set disjointness assumption
    to detect interactions.
    '''
    def __init__(self, anova):
        super().__init__('Archipelago')
        self.parent = anova  # use the cached set info, not so clean but yeah.

    @staticmethod
    def find_island(islands, n):
        ''' Return an island parent id, or create the entry islands[n] = n.'''
        while islands[n] != n:
            n = islands[n]
        return n

    def arch_detect(self, phi, threshold):
        ''' ArchDetect
        Looks at subset pairs and merge into islands, ie. union-find algorithm.
        Note: instead of using the TopK effects, we cut at a threshold, which is
        equivalent and allows a similar treatement as our other implementations.

        * phi: abs(arch_attribute) effects of (i,j) pairs.
        * threshold: positive, should probably be a fraction of max(phi.values())
        '''
        max_phi = max(phi.values())
        islands = defaultdict(int)  # island_id
        island_size = defaultdict(lambda: 1)

        for bin_subset in phi:
            card = len(self.parent.cache_subset[bin_subset]['l'])
            if card != 2:
                raise ValueError('Check phi, it should not have subset', bin_subset)

            if phi[bin_subset] > threshold * max_phi:
                # subset is relevant, add to islands
                f1, f2 = self.parent.cache_subset[bin_subset]['l']
                if f1 not in islands:
                    islands[f1] = f1
                if f2 not in islands:
                    islands[f2] = f2
                f1_par = self.find_island(islands, f1)
                f2_par = self.find_island(islands, f2)
                if f1_par != f2_par:  # not connected
                    if island_size[f1_par] > island_size[f2_par]:
                        f1_par, f2_par = f2_par, f1_par
                    islands[f1_par] = f2_par  # connect 1 and 2
                    island_size[f2_par] += island_size[f1_par]

        detected_binsubsets = defaultdict(int)
        for island in islands:
            detected_binsubsets[self.find_island(islands, island)] += 1 << island
        return detected_binsubsets


    def explain(self, x0, f_marginal, dist):
        ''' Uses the ANOVA cache for x0 if available.

        * x0: point
        * phi: relevance dict for x0 computed with anova
        * f_marginal: E(Y|X_S)

        In marginal langage, Archipelago omegas translate as
            w = 1/2 ( [ E(y|x) - E(y|x~i) - E(y|x~j) + E(y|x~(i,j)) ]^2
                    + [ E(y|x_(i,j)) - E(y|xi) - E(y|xj) + E(y) ]^2 )
        This is basically SHAP interact computed on two permutations only. '''
        if len(x0.shape) != 2:
            x0 = x0.reshape((1, -1))  # x is flat
        d = x0.shape[1]
        phis = {}

        for i in range(d-1):
            for j in range(i+1, d):
                # -- \omega'
                bin_pair = (1 << i) + (1 << j)
                if bin_pair in self.parent.cache:
                    fij_1 = self.parent.cache[bin_pair]
                else:
                    fij_1 = float(f_marginal(x0, [i,j]))
                    self.parent.cache[bin_pair] = fij_1
                    raise ValueError('should have used cache', (i, j))

                single_pair_i = 1 << i
                if single_pair_i in self.parent.cache:
                    fi_1 = self.parent.cache[single_pair_i]
                else:
                    fi_1 = float(f_marginal(x0, [i]))
                    self.parent.cache[single_pair_i] = fi_1
                    raise ValueError('should have used cache', i)

                single_pair_j = 1 << j
                if single_pair_j in self.parent.cache:
                    fj_1 = self.parent.cache[single_pair_j]
                else:
                    fj_1 = float(f_marginal(x0, [j]))
                    self.parent.cache[single_pair_j] = fj_1
                    raise ValueError('should have used cache', j)

                if 0 in self.parent.cache:
                    f0_1 = self.parent.cache[0]
                else:
                    f0_1 = float(f_marginal(x0, []))
                    self.parent.cache[0] = f0_1
                    raise ValueError('should have used cache', 0)

                omega_base = abs(fij_1 - fi_1 - fj_1 + f0_1)


                # -- \omega^*
                all_feat = ((1 << d) - 1)
                if all_feat in self.parent.cache:
                    f0_2 = self.parent.cache[all_feat]
                else:
                    f0_2 = float(f_marginal(x0, list(range(d))))
                    self.parent.cache[all_feat] = f0_2

                single_pair_i = all_feat ^ single_pair_i
                if single_pair_i in self.parent.cache:
                    fi_2 = self.parent.cache[single_pair_i]
                else:
                    subset_uni = list(range(d))
                    subset_uni.pop(i)
                    fi_2 = float(f_marginal(x0, self.parent.cache_subset[single_pair_i]['l']))
                    self.parent.cache[single_pair_i] = fi_2

                single_pair_j = all_feat ^ single_pair_j
                if single_pair_j in self.parent.cache:
                    fj_2 = self.parent.cache[single_pair_j]
                else:
                    subset_uni = list(range(d))
                    subset_uni.pop(j)
                    fj_2 = float(f_marginal(x0, self.parent.cache_subset[single_pair_j]['l']))
                    self.parent.cache[single_pair_j] = fj_2

                bin_pair_bar = all_feat ^ bin_pair
                if bin_pair_bar in self.parent.cache:
                    fij_2 = self.parent.cache[bin_pair_bar]
                else:
                    fij_2 = float(f_marginal(x0, self.parent.cache_subset[bin_pair_bar]['l']))
                    self.parent.cache[bin_pair_bar] = fij_2

                omega_star = abs(f0_2 - fj_2 - fi_2 + fij_2)

                phis[bin_pair] = 0.5 * (omega_base + omega_star)

        return phis

    def get_selection_from_subset(self, phi, x0, f_marginal,
                threshold=0.5, main_gam_sol=-1, ):
        ''' ArchAttribute

        Computed for disjoint subsets, with min card 2. Main effect can be added,
        and for that we cheat a little and use the result of the GAM computation.
         '''
        candidate_sets = self.arch_detect(phi, threshold)
        baseline = float(f_marginal(x0, []))

        max_prob = 0.
        max_subset = -1
        if main_gam_sol >= 0:
            max_subset = main_gam_sol
            max_prob = abs( float(f_marginal(x0, self.parent.cache_subset[main_gam_sol]['l']))
                    - baseline )

        for bin_subset in candidate_sets.values():
            prob_subset = abs( float(f_marginal(x0, self.parent.cache_subset[bin_subset]['l']) )
                    - baseline )
            if prob_subset > max_prob:
                max_prob = prob_subset
                max_subset = bin_subset
        return max_subset


## Exact function computation

def predict(X, mu, label, var=0.1):
    ''' py|x '''
    d = mu.shape[1]
    mu_ = np.expand_dims(mu.T, 0)  # (1, d, K)
    label_ = np.expand_dims(label, 0)  # (1, 1, K)
    distance = np.sum(np.square(mu_ - np.expand_dims(X, -1)), axis=1)  # (B, K)
    px = 1 / np.sqrt((2 * np.pi * var) ** d)* np.exp( - distance / (2 * var))
    pyx = np.sum(px * label_, axis=-1) / (np.abs(np.sum(px, axis=-1)) + np.finfo(np.float32).eps)
    return pyx

def predict_marginal(X, mu, label, subset_marg, var=0.1):
    '''
    * [same params as predict()]
    * subset_marg: list of indexes

    Because we use a mixture of indep gaussians and binary labels, we have
    E[Y|X_S] = \sum_y y P(Y = y|X_S) = P(Y = 1|X_S) = P(Y=1, X_S=x_s)/P(X_S=x_s)

    P(Y=1,X_S=x_s) = \int_xbar p(xs, xbar, y=1)
    = \int_xbar \sum_c p(c) p(xs, xbar, y=1 | c)
    = \int_xbar \sum_c p(c)p(xs|c)p(xbar|c)p(y|c)
    = 1/C \sum_c * p(xs|c) * \delta_{y_c = 1} * 1

    Similarly, P(X_S=x_s) = \sum_c 1/c * p(xs|c)
    where p(xs|c) is the gaussian marginal density with |s| dimensions.

    Everything happens as if we ignored the components of X_{\bar{S}}, that
    is thanks to the independant component of the normals.
    '''
    if len(subset_marg) == 0:
        return np.mean(label) * np.ones(X.shape[0])
    pyx_marginal = predict(X[:,subset_marg], mu[:,subset_marg], label, var)
    return pyx_marginal

def grad_predict(X, mu, label, var = 0.1):
    ''' \nabla_x py|x '''
    d = mu.shape[1]
    mu_ = np.expand_dims(mu.T, 0)  # (1, d, K)
    label_ = np.expand_dims(label, 0)  # (1, 1, K)
    distance = np.sum(np.square(mu_ - np.expand_dims(X, -1)), axis=1)  # (B, K)
    px_c = 1 / np.sqrt((2 * np.pi * var) ** d)* np.exp( - distance / (2 * var))
    px = np.mean(px_c, axis=-1, keepdims=True)  # \sum_c^C p(c)p(x|c), p(c) = 1/C
    pxy = np.mean(px_c * label_, axis=-1, keepdims=True)
    grad_denom = np.square(px) + np.finfo(np.float32).eps
    grad_num = np.mean(
        np.expand_dims(px_c * (-1 / var) * (label_ * px - pxy), 1)
        * (np.expand_dims(X, -1) - mu_),
        axis=-1)
    return grad_num / grad_denom
