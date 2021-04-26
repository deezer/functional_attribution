'''
Evaluate multivariate methods.

Ex:
python3 eval_multivariate.py --gpu 0 --task_set A --start_dim 2 --end_dim 5 --output results_D --tuned_params tuning/tunedparams.npy
'''
import argparse
import os
import sys
import time
import glob
from tqdm import tqdm
import numpy as np
from collections import defaultdict

from methods import *
from multivariate_gen import gen_multivariate
from selector_predictor import *

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", help="GPU index to use", type=int, default=-1)
parser.add_argument("--task_set", help="Tasks set to choose", type=str, default='')
parser.add_argument("--tuned_params", help=".npy file with tuned parameters", type=str, default='')
parser.add_argument("--start_dim", help="Task dim range start", type=int, default=-1)
parser.add_argument("--end_dim", help="Task dim range end", type=int, default=100)
parser.add_argument("--output", help="Output directory with intermediate results", type=str, default='results')
parser.add_argument("--exp_name", help="Unique identifier for a given experiment", type=str, default='')
parser.add_argument("--mode", help="Execute all methods or given subset", type=str, default='all')
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu);
if args.mode not in ['all', 'uni', 'anova', 'selpred', 'l2x', 'invase']:
    raise ValueError('Mode param error')

MAX_DIM = 11
EXP_PER_DIM = 100
MAX_SHAPES_PER_EXP = 10
MAX_SHAPE_DIM = 5
MIN_DIST_BETWEEN_POINTS = 0.25
TASK_NAME = args.task_set
TASK_PATH = os.path.join('task_sets', TASK_NAME)
TIMESTAMP = str(round(time.time()))  # unique identifier for saved files
if args.exp_name != '':
    TIMESTAMP = args.exp_name   # i know, bad naming convention.

print('Starting experiment', TIMESTAMP)

## Methods instantiations

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
        shap_sample, shap_sample_real,
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
        'threshold': 0.7,  # tuned, training threshold used to check stability of the sel solution
    }

params_invase = {
        'input_dim': 0,
        'F': 100,
        'threshold': 0.7,  # tuned, training threshold used to check stability of the sel solution
        'reg': 0.01,  # tuned, but super unstable, that's a known issue with L1 reg
        'mode': 'l1',  # paper is with L0, but official repo L1 -> same performances.
    }

if args.mode == 'l2x':
    sel_pred_explainers = ['l2x']
elif args.mode == 'invase':
    sel_pred_explainers = ['ignore', 'invase']
else:
    sel_pred_explainers = ['l2x', 'invase']


## Task set opening or generation

tasks = {} # [task_id](x,y,s)

max_points = 0
if TASK_NAME != '':
    tasks = np.load(TASK_PATH+'.npy', allow_pickle=True).item()
    print('Loaded', len(tasks), 'tasks.')
else:
    for DIM in range(2, MAX_DIM+1):
        for j in range(EXP_PER_DIM+1):
            task_id = '{}_{}'.format(DIM, j)
            if j == 0:
                # sanity-check
                x, y, s, t = gen_multivariate(1, DIM,
                                        max_sel_dim = min(DIM, MAX_SHAPE_DIM),
                                        sigma = MIN_DIST_BETWEEN_POINTS,
                                        prob_simplify = 0.5)
            else:
                x, y, s, t = gen_multivariate(
                                        1 + (j * MAX_SHAPES_PER_EXP) // EXP_PER_DIM,  # adaptative
                                        DIM,
                                        max_sel_dim=min(DIM, MAX_SHAPE_DIM),
                                        sigma = MIN_DIST_BETWEEN_POINTS,
                                        prob_simplify = 0.5)
            max_points = max(x.shape[0], max_points)
            tasks[task_id] = (x, y, s, t) # input, target, selection
    print('Created', len(tasks), 'tasks.')
    filename = 'taskset_{}'.format(TIMESTAMP)
    np.save(os.path.join(args.output, filename), tasks)


## Load and merge multiple files if rerun

outputs = {}  # hat_w[method][task_id]
times = {}

outputs_files = glob.glob(os.path.join(args.output, 'outputs_*.npy'))
if args.mode != 'all':
    outputs_files = glob.glob(os.path.join(args.output, 'outputs_*' + args.mode + '.npy'))
for outputs_file in outputs_files:
    f = np.load(outputs_file, allow_pickle=True).item()
    for task in f:
        if task not in outputs:
            outputs[task] = f[task]
        else:
            outputs[task].update(f[task])  # don't overwrite subdict
    print('Loaded {} computed results on {} tasks'.format(outputs_file, len(f.keys()) ))

times_files = glob.glob(os.path.join(args.output, 'times_*.npy'))
if args.mode != 'all':
    times_files = glob.glob(os.path.join(args.output, 'times_*' + args.mode + '.npy'))
for time_file in times_files:
    f = np.load(time_file, allow_pickle=True).item()
    for task in f:
        if task not in times:
            times[task] = f[task]
        else:
            for method in f[task]:
                if method not in times[task]:
                    times[task][method] = f[task][method]
                else:
                    # take the max, there can be zeros due to reruns.
                    times[task][method] = max(times[task][method], f[task][method])
    print('Loaded {} measured times on {} tasks'.format(time_file, len(f.keys()) ))


## Raw output computation

last_dim = -1
ellapsed_time = time.time()
computed_something = False   # for multiple runs, avoid saving duplicate results

for task in tqdm(tasks):
    # -- Get a task
    x, y, s, t = tasks[task]

    if x.shape[1] < args.start_dim or x.shape[1] > args.end_dim: # unstudied dim
        continue
    if task not in outputs:
        outputs[task] = {}
    if task not in times:
        times[task] = {}

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

    # -- Auto-save results every 10 min
    if time.time() - ellapsed_time > 60 * 10:
        ellapsed_time = time.time()
        filename = 'outputs_{}_intermediate_{}'.format(TASK_NAME, TIMESTAMP)
        if args.mode != 'all':
            filename += '_' + args.mode
        np.save(os.path.join(args.output, filename), outputs)
        np.save(os.path.join(args.output, 'times' + filename[7:]), times)
        print('\n AUTOSAVE', os.path.join(args.output, filename) )

    # -- Univariate methods
    for j, explainer in enumerate(uni_single_explainers + uni_grad_explainers):
        if args.mode not in ['all', 'uni']:  # if selected mode
            continue
        explainer_id = 'uni_{}_{}'.format(j, explainer.name)
        if explainer_id in outputs[task]:  # already computed
            continue
        else:
            computed_something = True

        start_time = time.time()
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
        times[task][explainer_id] = time.time() - start_time

    # --- ANOVA like methods
    for j, explainer in enumerate(multi_uni_explainers):
        if args.mode not in ['all', 'anova']:  # if selected mode
            continue
        explainer_id = 'multi_{}_{}'.format(j, explainer.name)
        if explainer_id in outputs[task]:  # already computed
            continue
        else:
            computed_something = True

        start_time = time.time()
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
        times[task][explainer_id] = time.time() - start_time
        if (explainer.name == 'fANOVA'):
            outputs[task][explainer_id_archipelago] = hat_phi_archipelago

    # --- Selector-predictors methods
    if len(sel_pred_explainers) > 0:
        tf.keras.backend.clear_session()
        data_xy = make_generator(sample_data, { 'bs': 512, 'mu': x, 'label': y, 'std': MIN_DIST_BETWEEN_POINTS/2 })

    for j, explainer in enumerate(sel_pred_explainers):
        if args.mode not in ['all', 'selpred', 'l2x', 'invase']:  # if selected mode
            continue
        explainer_id = '{}_{}'.format(j, explainer)
        if explainer_id in outputs[task]:  # already computed
            continue
        else:
            computed_something = True

        # compute
        start_time = time.time()
        if explainer == 'l2x':
            if args.mode == 'invase':  # if selected mode
                continue
            l2x = L2X(params_l2x)
            l2x.create_model()
            l2x.add_callback(x, patience=10, after=200)
            l2x.train(data_xy, n_epochs=500)
            logits = l2x.get_mask(x)
            _, explainer_acc = l2x.m.evaluate(data_xy, steps=100)

        elif explainer == 'invase':
            if args.mode == 'l2x':  # if selected mode
                continue
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
            times[task][explainer_id] = time.time() - start_time
        else:
            if explainer != 'ignore':
                raise ValueError('Forgot to explicitely set one method!')

if computed_something:
    filename = 'outputs_{}_{}'.format(TASK_NAME, TIMESTAMP)
    if args.mode != 'all':
        filename += '_' + args.mode
    np.save(os.path.join(args.output, filename), outputs)
    np.save(os.path.join(args.output, 'times' + filename[7:]), times)
else:
    fanova.precompute_subset(MAX_DIM)

## Tuning file (should have all the methods tuned parameters, else it raises an error).

tuned_params = {}
if args.tuned_params != '':
    tuned_params = np.load(args.tuned_params, allow_pickle=True).item()


## Accuracy computation

accuracies = defaultdict(list)
selections = defaultdict(dict)
incomplete = False

# -- Tune results UNIVARIATE
for task in tqdm(tasks):
    if (task not in outputs) or (len(outputs[task]) == 0):  # uncomputed task, probably worrisome
        incomplete = True
        continue

    x, y, s, t = tasks[task]
    task_predict_marginal = lambda X, S: predict_marginal(X, x, y, S,
                                    var=(MIN_DIST_BETWEEN_POINTS/2) ** 2)  # for archipelago

    d = x.shape[1]  # dim of the problem
    bin_s = np.zeros(len(s), dtype='int')
    bin_s[:] = np.array([binarise_subset(true_subset) for true_subset in s], dtype='int')

    # -- Univariate
    for j, explainer in enumerate(uni_single_explainers + uni_grad_explainers):
        explainer_id = 'uni_{}_{}'.format(j, explainer.name)
        if args.mode not in ['all', 'uni']:  # if selected mode
            continue

        hat_s = np.zeros(len(s), dtype='int')
        for i, w in enumerate(outputs[task][explainer_id]):
            subset_w = explainer.get_selection_from_feature(w,
                    level = tuned_params[explainer_id], # * 1. / d
                    min_val=1e-2)
            hat_s[i] = binarise_subset(subset_w)
        accuracy = (bin_s == hat_s).mean()
        accuracies[explainer_id].append(accuracy)
        selections[task][explainer_id] = hat_s

    # -- ANOVA-like
    for j, explainer in enumerate(multi_uni_explainers):
        explainer_id = 'multi_{}_{}'.format(j, explainer.name)
        if args.mode not in ['all', 'anova']:  # if selected mode
            continue

        # init
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

        # compute
        for i, phi in enumerate(outputs[task][explainer_id]):
            if (explainer.name == 'fANOVA'):
                hat_s[i] = explainer.get_selection_from_subset(phi,
                                threshold = tuned_params[explainer_id],
                                infty=True)
                hat_s_gam[i] = explainer.get_selection_from_subset(phi,
                                threshold = tuned_params[explainer_id_gam],
                                limit_card = 1)
                hat_s_ga2m[i] = explainer.get_selection_from_subset(phi,
                                threshold = tuned_params[explainer_id_ga2m],
                                limit_card = 2)
                hat_s_ga3m[i] = explainer.get_selection_from_subset(phi,
                                threshold = tuned_params[explainer_id_ga3m],
                                limit_card = 3)
                hat_s_ga4m[i] = explainer.get_selection_from_subset(phi,
                                threshold = tuned_params[explainer_id_ga4m],
                                limit_card = 4)
                hat_s_afchar[i] = explainer.get_selection_afchar(phi,
                                threshold = tuned_params[explainer_id_afchar])

                hat_s_archipelago[i] = archipelago.get_selection_from_subset(
                                outputs[task][explainer_id_arch][i],
                                x[i:i+1],
                                task_predict_marginal,
                                threshold = tuned_params[explainer_id_arch],
                                main_gam_sol = hat_s_gam[i])
            else:
                raise ValueError('Forgot to explicitely set one function!')

        # store
        accuracy = (bin_s == hat_s).mean()
        accuracies[explainer_id].append(accuracy)
        selections[task][explainer_id] = hat_s
        if explainer.name == 'fANOVA':
            accuracies[explainer_id_gam].append((bin_s == hat_s_gam).mean())
            accuracies[explainer_id_ga2m].append((bin_s == hat_s_ga2m).mean())
            accuracies[explainer_id_ga3m].append((bin_s == hat_s_ga3m).mean())
            accuracies[explainer_id_ga4m].append((bin_s == hat_s_ga4m).mean())
            accuracies[explainer_id_afchar].append((bin_s == hat_s_afchar).mean())
            accuracies[explainer_id_arch].append((bin_s == hat_s_archipelago).mean())
            selections[task][explainer_id_gam] = (hat_s_gam)
            selections[task][explainer_id_ga2m] = (hat_s_ga2m)
            selections[task][explainer_id_ga3m] = (hat_s_ga3m)
            selections[task][explainer_id_ga4m] = (hat_s_ga4m)
            selections[task][explainer_id_afchar] = (hat_s_afchar)
            selections[task][explainer_id_arch] = (hat_s_archipelago)

    # -- Selector Predictor
    for j, explainer in enumerate(sel_pred_explainers):
        explainer_id = '{}_{}'.format(j, explainer)
        if args.mode not in ['all', 'selpred', 'l2x', 'invase']:
            continue

        if explainer != 'ignore':
            logit = outputs[task][explainer_id]['hat_s']
            if type(logit) != np.ndarray:  # then is a tf.tensor
                logit = logit.numpy()
            hat_s = (logit >= (np.max(logit, axis=-1, keepdims=True) * tuned_params[explainer_id]) )
            bin_hat_s = binarise_solution_from_array(hat_s)
            accuracy = (bin_s == bin_hat_s).mean()
            accuracies[explainer_id].append(accuracy)
            selections[task][explainer_id] = (bin_hat_s)


output_text = ''
def print_and_collect(s, output_text):
    output_text += s + '\n'
    return output_text

output_text = print_and_collect('Accuracy results on ' + TASK_PATH + 'with'
                    + str(len(outputs)) + 'tasks', output_text)
for explainer in accuracies:
    fact = 1.96 / np.sqrt(len(accuracies[explainer]))
    acc_param = np.mean(accuracies[explainer])
    output_text = print_and_collect('{} : {:.3f} +/- {:.3f}'.format(
                            explainer,
                            acc_param,
                            fact * np.std(accuracies[explainer]) ), output_text)
print(output_text)

filename = 'accuracy_{}_{}'.format(TASK_NAME, TIMESTAMP)
if args.mode != 'all':
    filename += '_' + args.mode
if incomplete:
    print('WARNING: not all tasks output of taskset were available.')
    filename += '_incomplete'

np.save(os.path.join(args.output, filename), accuracies)
np.save(os.path.join(args.output, 'summary_' + filename), output_text)
np.save(os.path.join(args.output, 'selection' + filename[8:]), selections)


## Eval property verification

properties = defaultdict(list)

for task in tqdm(tasks):
    x, y, s, t = tasks[task]
    tot_points = x.shape[0]
    d = x.shape[1]

    for method in selections[task]:
        count_verified = 0
        count_samples = 0
        for shape in t:
            for shape_index in shape:
                # for all x
                point = x[shape_index]
                # there exists I, x \in A_I(f)
                bin_sel = selections[task][method][shape_index]
                sel = [l for l in range(d) if ((1<<l)&bin_sel>0)]
                verifies_property = True
                for k in shape:
                    # then, for all x' (who could have P_I(x) = P_I(x') are in shape)
                    neighbor = x[k]
                    neighbor_bin_sel = selections[task][method][k]
                    if np.sum(np.abs(neighbor[sel] - point[sel])) < MIN_DIST_BETWEEN_POINTS/2:
                        # such that P_I(x) = P_I(x')
                        if neighbor_bin_sel & bin_sel != neighbor_bin_sel:
                            # x' \in A_I(f)? if J \subset I and x' \in A_J(f) then yes.
                            verifies_property = False
                            break
                count_verified += int(verifies_property)
                count_samples += 1
        properties[method].append(count_verified / tot_points)

output_text = ''
output_text = print_and_collect('Prop verification on ' + TASK_PATH + ' with '
                    + str(len(outputs)) + ' tasks', output_text)
for explainer in properties:
    fact = 1.96 / np.sqrt(len(properties[explainer]))
    output_text = print_and_collect('{} : {:.3f} +/- {:.3f}'.format(
                            explainer,
                            np.mean(properties[explainer]),
                            fact * np.std(properties[explainer]) ), output_text)
print(output_text)

filename = 'property_{}_{}'.format(TASK_NAME, TIMESTAMP)
if args.mode != 'all':
    filename += '_' + args.mode
if incomplete:
    print('WARNING: not all tasks output of task set were available.')
    filename += '_incomplete'

np.save(os.path.join(args.output, filename), properties)
np.save(os.path.join(args.output, 'summary_' + filename), output_text)
