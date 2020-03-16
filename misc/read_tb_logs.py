import numpy as np
import tensorboard
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tabulate import tabulate
from collections import OrderedDict

root_path = 'runs/criterion_experiment_no_bias/'


# Prepare regexes
re_crit = re.compile(r'crit=(.\w+)')
re_seed = re.compile(r'seed=(.\w+)')
re_pf = re.compile(r'pf=(\w+)')
re_model = re.compile(r' (\w+) ')
run_list = []

headers=['Model', 'Seed', 'PF', 'Prune crit.', 'Test acc.', 'Sparsity']
run_dict = {key:[] for key in headers}

for dirpath, dirs, files in os.walk(root_path):
    if dirpath==root_path:
        continue

    # Gather info about hparams
    model = re_model.search(dirpath).group(0).strip()
    pf = re_pf.search(dirpath).group(0).split('=')[1]
    seed = re_seed.search(dirpath).group(0).split('=')[1]
    crit = re_crit.search(dirpath).group(0).split('=')[1]
    
    # Open TB event file
    tb_event_file = os.path.join(dirpath, files[0])
    event_accum = EventAccumulator(tb_event_file)
    event_accum.Reload()
    
    test_acc = event_accum.Scalars('acc/test')[-1].value
    sparsity = event_accum.Scalars('sparsity/sparsity')[-1].value

    run_info = (model, seed, pf, crit, test_acc, sparsity)
    run_list.append(run_info)

    for key,elem in zip(run_dict.keys(), run_info):
        run_dict[key].append(elem)


# Turn the tb events file into df
df = pd.DataFrame.from_dict(run_dict)

# Calc mean and std over the seeds
groupby_keys = ['Model', 'Prune crit.', 'PF']
grouped = df.groupby(groupby_keys)

means = grouped.mean().reset_index()
means = means.rename(columns={'Test acc.': 'Mean acc.'})
stds = grouped.std().reset_index()
stds = stds.rename(columns={'Test acc.': 'Std. acc.'})
stds = stds.drop(labels='Sparsity', axis=1)

results = means.merge(stds, on=groupby_keys).sort_values('Sparsity')
results = results.sort_values('Sparsity')

# Build the plot dictionary
prune_crits = results['Prune crit.'].unique()
plot_dict = {key:{'means':[], 'stds':[], 'sparsities':[]} for key in prune_crits}

for idx, row in results.iterrows():
    prune_crit = row['Prune crit.']
    plot_dict[prune_crit]['means'].append(row['Mean acc.'])
    plot_dict[prune_crit]['stds'].append(row['Std. acc.'])
    plot_dict[prune_crit]['sparsities'].append(row['Sparsity'])

# Actually do the plots
for k, v in plot_dict.items():
    plt.plot(np.arange(len(v['sparsities'])), v['means'], label=k)

# Dirty hack for xticks
for k, v in plot_dict.items():
    truncated = list(map(lambda x: f"{x:.4f}", v['sparsities']))
    plt.xticks(np.arange(len(v['sparsities'])), truncated)
    break

plt.title('Sparsity vs. acc')
plt.legend()
plt.grid()
plt.savefig('fig.png')