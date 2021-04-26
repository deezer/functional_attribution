'''
Tune methods parameters.

Tuning can only be done in multidimensional case: eg. univariate methods would
else converge to star -- "always univariate" -- selection if tuned separately on
univariate tasks, which is not realistic.
'''
import sys
import time
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from collections import defaultdict

from methods import *
from multivariate_gen import gen_multivariate
from selector_predictor import *


MAX_DIM = 11
EXP_PER_DIM = 10
MAX_SHAPES_PER_EXP = 10
MAX_SHAPE_DIM = 5
MIN_DIST_BETWEEN_POINTS = 0.25

TASK_PATH = 'task_sets/tuning'

TIMESTAMP = str(round(time.time()))

# methods
lime_tab = LIME(continuous = False, samples=1000)
lime_cont = LIME(continuous = True, samples=1000)
shap_sample = Shapley_Sample(mode='mean_function', samples=128, use_exact=True)
shap_sample_real = Shapley_Sample(mode='function_mean', samples=128, use_exact=True)
gam_star = GAM()
saliency = Grad()
grad_x_input = GradXInput()
intgrad = IntGrad(m_steps=50)
expectedgrad = ExpectedGrad(samples=500)

uni_single_explainers = [
        lime_tab, lime_cont,
        gam_star,
        shap_sample, shap_sample_real
        ]

uni_grad_explainers = [
        saliency, grad_x_input, intgrad, expectedgrad,
]

fanova = fANOVA(limit_order=MAX_SHAPE_DIM)
archipelago = Archipelago(fanova)  # run independently
multi_uni_explainers = [ fanova ]

params_l2x = {
        'input_dim': 0,
        'F': 100,
        't': 0.1,
        'k': MAX_SHAPE_DIM,
        'threshold': 0.7,  # it's a training threshold used to check stability of the sel solution
    }

params_invase = {
        'input_dim': 0,
        'F': 100,
        'threshold': 0.7,
        'reg': 0.01,  # to tune
        'mode': 'l1',
    }

sel_pred_explainers = ['l2x', 'invase']


## Task set opening

tasks = {} # [task_id](x,y,s)
if TASK_PATH != '':
    tasks = np.load(TASK_PATH+'.npy', allow_pickle=True).item()
else:
    raise ValueError('Tuning requires a fixed dataset.')
print('Loaded', len(tasks), 'tasks.')


## Raw computations

outputs = {}  # hat_w[method][task_id]
last_dim = -1

for task in tqdm(tasks):
    # -- Get a task
    x, y, s, t = tasks[task]

    if task not in outputs:
        outputs[task] = {}

    task_predict = lambda X: predict(X, x, y,
                                    var=(MIN_DIST_BETWEEN_POINTS/2) ** 2)
    task_grad_predict = lambda X: grad_predict(X, x, y,
                                    var=MIN_DIST_BETWEEN_POINTS ** 2)
    task_predict_marginal = lambda X, S: predict_marginal(X, x, y, S,
                                    var=(MIN_DIST_BETWEEN_POINTS/2) ** 2)
    dist = Distribution(x)

    # -- Dim change event
    if (last_dim != x.shape[1]):  # task cache reset after a dimension change
        last_dim = x.shape[1]
        # reset caches
        for explainer in multi_uni_explainers:
            if (explainer.name == 'fANOVA'):
                explainer.reset_dim_cache()
        params_l2x['input_dim'] = last_dim
        params_invase['input_dim'] = last_dim

    # -- Univariate methods
    for j, explainer in enumerate(uni_single_explainers + uni_grad_explainers):
        explainer_id = 'uni_{}_{}'.format(j, explainer.name)
        hat_w = np.zeros_like(x)  # stores feature attribution values (size n)
        for i, point in enumerate(x):
            # choose the right function for each method
            if ((explainer.name == 'Shapley' and explainer.mode == 'mean_function')
                    or (explainer.name == 'Shapley Sampled' and explainer.mode == 'mean_function')
                    or (explainer.name == 'GAM')):
                w = explainer.explain(point, task_predict_marginal, dist)
            elif ((explainer.name == 'Gradient')
                    or (explainer.name == 'Gradient x Input')
                    or (explainer.name == 'Integrated Gradient')
                    or (explainer.name == 'Expected Gradient')):
                w = explainer.explain(point, task_grad_predict, dist)
            elif ((explainer.name == 'LIME')
                or (explainer.name == 'Shapley' and explainer.mode == 'function_mean')
                or (explainer.name == 'Shapley Sampled' and explainer.mode == 'function_mean')):
                w = explainer.explain(point, task_predict, dist)
            else:
                raise ValueError('Forgot to explicitely set one function!')
            hat_w[i,:] = w
        outputs[task][explainer_id] = hat_w

    # --- ANOVA like methods
    for j, explainer in enumerate(multi_uni_explainers):
        explainer_id = 'multi_{}_{}'.format(j, explainer.name)
        hat_phi = []  # stores subset attribution values (dict, up to 2**n entries)
        if (explainer.name == 'fANOVA'):
            explainer_id_archipelago = 'multi_{}_archipelago'.format(j, explainer.name)
            hat_phi_archipelago = []  # computed right after to use cache

        for i, point in enumerate(x):
            # point cache reset
            if (explainer.name == 'fANOVA'):
                explainer.reset_cache()
                phi = explainer.explain(point, task_predict_marginal, dist)
                phi_archipelago = archipelago.explain(point, task_predict_marginal, dist)
                hat_phi_archipelago.append(phi_archipelago)
            else:
                raise ValueError('Forgot to explicitely set one function!')
            hat_phi.append(phi)

        outputs[task][explainer_id] = hat_phi
        if (explainer.name == 'fANOVA'):
            outputs[task][explainer_id_archipelago] = hat_phi_archipelago

    # --- Selector-predictors methods
    if len(sel_pred_explainers) > 0:
        tf.keras.backend.clear_session()
        data_xy = make_generator(sample_data, { 'bs': 512, 'mu': x, 'label': y, 'std': MIN_DIST_BETWEEN_POINTS/2 })

    for j, explainer in enumerate(sel_pred_explainers):
        explainer_id = '{}_{}'.format(j, explainer)

        # compute
        if explainer == 'l2x':
            l2x = L2X(params_l2x)
            l2x.create_model()
            l2x.add_callback(x, patience=10, after=200)
            l2x.train(data_xy, n_epochs=500)
            logits = l2x.get_mask(x)
            _, explainer_acc = l2x.m.evaluate(data_xy, steps=100)

        elif explainer == 'invase':
            invase = INVASE(params_invase)
            invase.create_model()
            invase.add_callback(x, patience=10, after=200)
            invase.train(data_xy, n_epochs=500)
            logits = invase.get_mask(x).numpy()
            invase.m.evaluate(data_xy, steps=100)
            explainer_acc = invase.m.metrics[-1].result().numpy()

        # store
        if explainer in ['l2x', 'invase']:
            outputs[task][explainer_id] = { 'hat_s': logits,
                                            'acc': explainer_acc }
        else:
            if explainer != 'ignore':
                raise ValueError('Forgot to explicitely set one method!')

np.save('tuning/output_' + TIMESTAMP, outputs)


## TUNE from results

params = defaultdict(lambda: defaultdict(list))
tuned_params = {}

UNI_PARAMS =  list(np.arange(0.1, 0.96, 0.01))
MULTI_PARAMS = list(np.arange(0.5, 0.96, 0.01))

# -- Tune results UNIVARIATE
for task in tqdm(tasks):
    x, y, s, t = tasks[task]
    task_predict_marginal = lambda X, S: predict_marginal(X, x, y, S,
                                    var=(MIN_DIST_BETWEEN_POINTS/2) ** 2)  # for archipelago

    d = x.shape[1]  # dim of the problem
    bin_s = np.zeros(len(s), dtype='int')
    bin_s[:] = np.array([binarise_subset(true_subset) for true_subset in s], dtype='int')

    for uni_param in UNI_PARAMS:
        for j, explainer in enumerate(uni_single_explainers + uni_grad_explainers):
            explainer_id = 'uni_{}_{}'.format(j, explainer.name)
            hat_s = np.zeros(len(s), dtype='int')
            for i, w in enumerate(outputs[task][explainer_id]):
                subset_w = explainer.get_selection_from_feature(w,
                        level = uni_param, # * 1. / d
                        min_val=1e-4)
                hat_s[i] = binarise_subset(subset_w)
            accuracy = (bin_s == hat_s).mean()
            params[explainer_id][uni_param].append(accuracy)

    for k, multi_param in enumerate(MULTI_PARAMS):
        for j, explainer in enumerate(multi_uni_explainers):
            explainer_id = 'multi_{}_{}'.format(j, explainer.name)
            hat_s = np.zeros(len(s), dtype='int')
            if explainer.name == 'fANOVA':
                explainer_id_gam = 'multi_{}_GAM'.format(j, explainer.name)
                explainer_id_ga2m = 'multi_{}_GA^2M'.format(j, explainer.name)
                explainer_id_ga3m = 'multi_{}_GA^3M'.format(j, explainer.name)
                explainer_id_ga4m = 'multi_{}_GA^4M'.format(j, explainer.name)
                explainer_id_afchar = 'multi_{}_afchar'.format(j, explainer.name)
                hat_s_gam = np.zeros(len(s), dtype='int')
                hat_s_ga2m = np.zeros(len(s), dtype='int')
                hat_s_ga3m = np.zeros(len(s), dtype='int')
                hat_s_ga4m = np.zeros(len(s), dtype='int')
                hat_s_afchar = np.zeros(len(s), dtype='int')

                explainer_id_arch = 'multi_{}_archipelago'.format(j, explainer.name)
                hat_s_archipelago = np.zeros(len(s), dtype='int')

                ext_param = 2 * (multi_param - 0.5)  # explore [0,1] instead

            for i, phi in enumerate(outputs[task][explainer_id]):
                if (explainer.name == 'fANOVA'):
                    hat_s[i] = explainer.get_selection_from_subset(phi,
                                    threshold = multi_param)
                    hat_s_gam[i] = explainer.get_selection_from_subset(phi,
                                    threshold = multi_param, limit_card = 1)
                    hat_s_ga2m[i] = explainer.get_selection_from_subset(phi,
                                    threshold = multi_param, limit_card = 2)
                    hat_s_ga3m[i] = explainer.get_selection_from_subset(phi,
                                    threshold = multi_param, limit_card = 3)
                    hat_s_ga4m[i] = explainer.get_selection_from_subset(phi,
                                    threshold = multi_param, limit_card = 4)
                    hat_s_afchar[i] = explainer.get_selection_afchar(phi,
                                    threshold = ext_param)

                    hat_s_archipelago[i] = archipelago.get_selection_from_subset(
                                    outputs[task][explainer_id_arch][i],
                                    x[i:i+1],
                                    task_predict_marginal,
                                    threshold = ext_param,
                                    main_gam_sol = hat_s_gam[i])
                else:
                    raise ValueError('Forgot to explicitely set one function!')

            accuracy = (bin_s == hat_s).mean()
            params[explainer_id][multi_param].append(accuracy)

            if explainer.name == 'fANOVA':
                params[explainer_id_gam][multi_param].append((bin_s == hat_s_gam).mean())
                params[explainer_id_ga2m][multi_param].append((bin_s == hat_s_ga2m).mean())
                params[explainer_id_ga3m][multi_param].append((bin_s == hat_s_ga3m).mean())
                params[explainer_id_ga4m][multi_param].append((bin_s == hat_s_ga4m).mean())
                params[explainer_id_afchar][ext_param].append((bin_s == hat_s_afchar).mean())
                params[explainer_id_arch][ext_param].append((bin_s == hat_s_archipelago).mean())

    for k, multi_param in enumerate(MULTI_PARAMS):
        for j, explainer in enumerate(sel_pred_explainers):
            explainer_id = '{}_{}'.format(j, explainer)
            logit = outputs[task][explainer_id]['hat_s']
            hat_s = (logit >= (np.max(logit, axis=-1, keepdims=True) * multi_param) )
            bin_hat_s = binarise_solution_from_array(hat_s)
            accuracy = (bin_s == bin_hat_s).mean()
            params[explainer_id][multi_param].append(accuracy)



output_text = ''
def print_and_collect(s, output_text):
    output_text += s + '\n'
    return output_text

output_text = print_and_collect('Tuning results on ' + TASK_PATH, output_text)
for explainer in params:
    output_text = print_and_collect(explainer, output_text)
    best_param = -1
    best_param_val = 0.
    for param in params[explainer]:
        fact = 1.96 / np.sqrt(len(params[explainer][param]))
        acc_param = np.mean(params[explainer][param])
        if acc_param > best_param_val:
            best_param_val = acc_param
            best_param = param

        output_text = print_and_collect('  {:.2f}: {:.3f} ± {:.3f}'.format(param, acc_param,
                                fact * np.std(params[explainer][param]) ), output_text)
    tuned_params[explainer] = best_param
    output_text = print_and_collect('param* {:.2f} -> {:.5f}'.format(best_param, best_param_val),
                                    output_text)

print(output_text)

np.save('tuning/params_' + TIMESTAMP, dict(params))
np.save('tuning/tunedparams_' + TIMESTAMP, tuned_params)
