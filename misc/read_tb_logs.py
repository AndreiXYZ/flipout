import numpy as np
import tensorboard, os, re, sys
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tabulate import tabulate

root_path = './runs/criterion_experiment_no_bias/'

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
    
    # Gather info about layer sparsity
    for key in event_accum.scalars.Keys():
        if 'layerwise_sparsity' in key and ('linear' in key or 'conv' in key):
            # If key exists already, append to existing list
            # otherwise add it and create a 1-element list
            layer_sparsity = event_accum.Scalars(key)[-1].value
            layer_key = key.split('/')[1]

            if layer_key not in layer_sparsity_dict:
                layer_sparsity_dict[layer_key] = [layer_sparsity]
            else:
                layer_sparsity_dict[layer_key].append(layer_sparsity)


    test_acc = event_accum.Scalars('acc/test')[-1].value
    sparsity = event_accum.Scalars('sparsity/sparsity')[-1].value

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

# Process layerwise sparsities
# cand fac groupby stie sa faca media si std peste seed fiindca seed este string
df_layer_sparsities = pd.DataFrame.from_dict(layer_sparsity_dict)

grouped_layer_sparsities = df_layer_sparsities.groupby(groupby_keys)
layer_sparsity_means = grouped_layer_sparsities.mean().reset_index()
# Select just 1 level of sparsity
desired_sparsity = 0.984375
layer_sparsity_means = layer_sparsity_means[layer_sparsity_means['Sparsity']==desired_sparsity]
layer_sparsity_means.plot(kind='barh', x='Prune crit.', legend=False)
plt.grid()
plt.title('Layerwise sparsity across pruning techniques for sparsity={}'.format(desired_sparsity))
plt.savefig('misc/' + root_path.split('/')[2] + 
            '_layerwise_sparsity_at_' + str(desired_sparsity) + '.png')
plt.clf()
# Should also add errorbars
# layer_sparsity_stds = grouped_layer_sparsities.std().reset_index()

# Turn the tb events file into df
df = pd.DataFrame.from_dict(run_dict)
layerwise_sparsity_df = pd.DataFrame.from_dict(layer_sparsity_dict)

# Process the df of the metrics (test acc and final sparsity)
# Calc mean and std over the seeds
grouped = df.groupby(groupby_keys)

means = grouped.mean().reset_index()
means = means.rename(columns={'Test acc.': 'Mean acc.'})
stds = grouped.std().reset_index()
stds = stds.rename(columns={'Test acc.': 'Std. acc.'})
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

plt.title('Sparsity vs. acc')
plt.legend()
plt.grid()
plt.savefig('./misc/' + root_path.split('/')[2])