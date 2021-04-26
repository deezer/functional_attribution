# Towards Rigorous Interpretations

Source code of our paper "*Towards Rigorous Interpretations: a Formalisation of Feature Attribution*", by Darius Afchar, Romain Hennequin and Vincent Guigue.

*Link to be added soon to Arxiv.*

### Install dependencies

```sh
pip install -r requirement.txt
```

### Task sets

We provide the three task sets we have used in our paper:

- `multivariate.npy`: 1010 tasks with dimension ranging from 2 to 11 ;
- `univariate.npy`: 1010 tasks with dimension ranging from 2 to 11 ;
- `tuning`: 110 tasks with dimension ranging from 2 to 11.

*i.e.* we provide **100** (respectively 10) supervised tasks to solve for the main experiments per dimension *d* (respectively for tuning). The provided files are dictionaries indexed by keys `{d}_1` to `{d}_100` + 1 simple sanity-check task `{d}_0`. Each corresponding dict value contains a tuple of the centroids coordinates, associated labels, and ground-truth selection. The associated continuous distributions *p'* are generated using the functions `predict`, `predict_marginal` and `grad_predict` in the `methods.py` file.

### Run the experiments

**Multivariate selection task**

Example command:
```sh
python eval_multivariate.py --task_set multivariate --output results --tuned_params tuning/tunedparams.npy
```

Documentation:

```sh
usage: eval_multivariate.py [-h] [--gpu GPU] [--task_set TASK_SET]
                            [--tuned_params TUNED_PARAMS]
                            [--start_dim START_DIM] [--end_dim END_DIM]
                            [--output OUTPUT] [--exp_name EXP_NAME]
                            [--mode MODE]
                            
optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             GPU index to use
  --task_set TASK_SET   Task set to choose
  --tuned_params TUNED_PARAMS .npy file with tuned parameters
  --start_dim START_DIM Task dim range start
  --end_dim END_DIM     Task dim range end
  --output OUTPUT       Output directory with intermediate results
  --exp_name EXP_NAME   Unique identifier for a given experiment
  --mode MODE           Execute all methods or given subset
```

**Univariate selection task**

Example command:
```sh
python eval_univariate.py --task_set univariate --output results --tuned_params tuning/tunedparams.npy
```

Intermediate results are saved every ten minutes.