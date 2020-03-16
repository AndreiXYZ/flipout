import numpy as np
import tensorboard
import os
import re
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tabulate import tabulate

root_path = 'runs/criterion_experiment_no_bias/'
headers=['Model', 'Seed', 'PF', 'Prune crit.', 'Test acc.', 'Sparsity']

# Prepare regexes
re_crit = re.compile(r'crit=(.\w+)')
re_seed = re.compile(r'seed=(.\w+)')
re_pf = re.compile(r'pf=(\w+)')
re_model = re.compile(r' (\w+) ')
run_list = []

run_dict = {key:[] for key in headers}

for dirpath, dirs, files in os.walk(root_path):
    if dirpath==root_path:
        continue

    # Gather info about hparams
    model = re_model.search(dirpath).group(0)
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

df = pd.DataFrame.from_dict(run_dict)
print(df[df['Prune crit.']=='global_magnitude'])