import numpy as np
import tensorboard, os, re, sys
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tabulate import tabulate
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log_folder', type=str, help='Which folder to read for generating plots.')

config = vars(parser.parse_args())

root_path = config['log_folder']

# Prepare regexes
re_crit = re.compile(r'crit=(.\w+)')
re_seed = re.compile(r'seed=(.\w+)')
re_model = re.compile(r' (\w+) ')

headers=['Model', 'Seed', 'Prune crit.', 'Sparsity']
metrics = ['Test acc.']

run_dict = {key:[] for key in headers+metrics}
layer_sparsity_dict = {key:[] for key in headers}

for dirpath, dirs, files in os.walk(root_path):
    if dirpath==root_path:
        continue
    # Gather info about hparams
    model = re_model.search(dirpath).group(0).strip()
    seed = re_seed.search(dirpath).group(0).split('=')[1]
    crit = re_crit.search(dirpath).group(0).split('=')[1]
    
    # Open TB event file
    tb_event_file = os.path.join(dirpath, files[0])
    event_accum = EventAccumulator(tb_event_file)
    event_accum.Reload()

    test_acc = event_accum.Scalars('acc/test')[-1].value
    sparsity = event_accum.Scalars('sparsity/sparsity')[-1].value
    sparsity = round(sparsity, 4)

    run_keys = (model, seed, crit, sparsity)
    run_metrics = (test_acc,)

    # Add stuff to the run_dict
    for key,elem in zip(run_dict.keys(), run_keys+run_metrics):
        run_dict[key].append(elem)
    # Add stuff to the layerwise sparsity dict
    for key,elem in zip(headers, run_keys):
        layer_sparsity_dict[key].append(elem)


# Define the keys by which to groupby the dataframes
groupby_keys = ['Model', 'Prune crit.', 'Sparsity']

# Turn the tb events file into df
df = pd.DataFrame.from_dict(run_dict)
# Process the df of the metrics (test acc and final sparsity)
# Calc mean and std over the seeds

grouped = df.groupby(groupby_keys)

means = grouped.mean().reset_index()
means = means.rename(columns={'Test acc.': 'Mean acc.'})
stds = grouped.std().reset_index()
stds = stds.rename(columns={'Test acc.': 'Std. acc.'})
print(means)
print(stds)
results = means.merge(stds, on=groupby_keys)
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
    plot_length = np.arange(len(v['sparsities']))
    plt.errorbar(plot_length, v['means'], v['stds'], label=k)

# Dirty hack for xticks
for k, v in plot_dict.items():
    truncated = list(map(lambda x: f"{x:.4f}", v['sparsities']))
    plt.xticks(np.arange(len(v['sparsities'])), truncated)
    break

plt.title('Sparsity vs. acc (Resnet18-19 CIFAR-10) with bnorm pruning')
plt.legend()
plt.grid()
plt.xlabel('Sparsity')
plt.ylabel('Acc.')
plt.savefig('./misc/' + 'resnet18_prune_bnorm_results.png')