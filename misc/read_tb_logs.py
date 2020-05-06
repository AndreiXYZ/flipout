import numpy as np
import tensorboard, os, re, sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tabulate import tabulate
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--log_folder', type=str, help='Which folder to read for generating plots.')

config = vars(parser.parse_args())

root_path = config['log_folder']

# Prepare regexes
re_model = re.compile(r' (\w+)[_| ]')
re_seed = re.compile(r'seed=(.\w+)')
re_crit = re.compile(r'crit=.+[_| ]{1}')

headers=['Model', 'Seed', 'Prune crit.', 'Sparsity']
metrics = ['Test acc.']

run_dict = {key:[] for key in headers+metrics}
layer_sparsity_dict = {key:[] for key in headers}

hoyersquare_finetune_stats = []
hoyersquare_low_thresh_stats = []

for dirpath, dirs, files in os.walk(root_path):
    if dirpath==root_path:
        continue
    # Gather info about hparams
    model = re_model.search(dirpath).group(0)[:-1].strip()
    seed = re_seed.search(dirpath).group(0).split('=')[1].strip()
    crit = re_crit.search(dirpath).group(0).split('=')[1].strip()

    # Split from left to right by space or underscore, grab last elem
    # then re-reverse string
    crit = re.split('[ |_]', crit[::-1], maxsplit=1)[-1][::-1]
    
    # Open TB event file
    tb_event_file = os.path.join(dirpath, files[0])
    event_accum = EventAccumulator(tb_event_file)
    event_accum.Reload()

    test_acc = event_accum.Scalars('acc/test')[-1].value
    sparsity = event_accum.Scalars('sparsity/sparsity')[-1].value
    sparsity = round(sparsity, 4)

    # Do this for the case where we get NaNs in ResNet18
    if sparsity==0:
        continue
    # Remove some unwanted criteria
    if crit in ['magnitude', 'topflip', 'weight_div_flips', 'weight_div_squared_flips']:
        continue
    
    if 'weight_squared_div_flips' in crit and ('1.30' in crit or '1.4' in crit or '1.35' in crit):
        continue
    
    # Skip stuff that I do in the ablation studies#
    if 'noisy_global_magnitude' in crit:
        continue
    
    if 'scaling_factor' in crit:
        continue

    if 'hoyersquare_lower_thresh' in crit:
        continue
    ################################################

    if 'hoyersquare_threshold_finetuned' in crit:
        if 'thresh_' not in crit:
            hoyersquare_finetune_stats.append((sparsity, test_acc))
        else:
            hoyersquare_low_thresh_stats.append((sparsity, test_acc))
        continue
    
    # Skip the old hoyersquare method:
    if 'hoyer_square_lambda' in crit:
        continue
    
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

print(means.to_string())
print(stds.to_string())

results = means.merge(stds, on=groupby_keys)
# Build the plot dictionary
prune_crits = results['Prune crit.'].unique()
plot_dict = {key:{'means':[], 'stds':[], 'sparsities':[]} for key in prune_crits}

for idx, row in results.iterrows():
    prune_crit = row['Prune crit.']
    plot_dict[prune_crit]['means'].append(row['Mean acc.'])
    plot_dict[prune_crit]['stds'].append(row['Std. acc.'])
    plot_dict[prune_crit]['sparsities'].append(row['Sparsity'])


# Plot horizontal line of unpruned baseline
unpruned_baselines = {'resnet18': 0.9537,
                      'vgg19': 0.9358,
                      'densenet121': 0.9162}


plt.axhline(y=unpruned_baselines[model.lower()], linestyle='--', color='k', label='unpruned')

# Actually do the plots
for k, v in plot_dict.items():
    plt.errorbar(v['sparsities'], v['means'], v['stds'], label=k, marker='s', capsize=6)

# Plot hoyersquare finetune stuff
hoyersquare_finetune_stats = sorted(hoyersquare_finetune_stats, key=lambda x: x[0])
sparsities = [elem[0] for elem in hoyersquare_finetune_stats]
accs = [elem[1] for elem in hoyersquare_finetune_stats]
if len(hoyersquare_finetune_stats) > 0:
    plt.plot(sparsities, accs, 's-', label='hoyersquare_finetuned')


# Repeat for the lower sparsity ones
hoyersquare_low_thresh_stats = sorted(hoyersquare_low_thresh_stats, key=lambda x: x[0])
sparsities = [elem[0] for elem in hoyersquare_low_thresh_stats]
accs = [elem[1] for elem in hoyersquare_low_thresh_stats]
if len(hoyersquare_low_thresh_stats) > 0:
    plt.plot(sparsities, accs, 's-', label='hoyersquare_lower_thresh')
    print('-'*30)
    for elem in hoyersquare_low_thresh_stats:
        print(elem)

# Boilerplate stuff for plot to look good
plt.title('Sparsity vs. acc (VGG19 CIFAR10)')
plt.legend()
# plt.ylim(bottom=0.75, top=0.96)
# plt.xlim(left=0.75)
plt.grid()
plt.xscale('logit')
plt.minorticks_off()
ax = plt.gca()
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.xticks(v['sparsities'], v['sparsities'])
plt.minorticks_off()

plt.xlabel('Sparsity (%)')
plt.ylabel('Acc. (%)')
# plt.savefig('./misc/' + 'resnet18_results.png')
plt.show()