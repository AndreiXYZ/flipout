import csv
import os
import numpy as np
import matplotlib.pyplot as plt

thresholds = ['threshold1', 'threshold40', 'threshold65', 'threshold100', 'threshold200']

sparsities = {}
accs = {}

for fname in os.listdir('./sparsities'):
    with open('./sparsities/' + fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # skip header
        next(csv_reader)
        sparsities[fname[:-4]] = [float(row[2]) for row in csv_reader]

    with open('./test_accs/' + fname) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        # skip header
        next(csv_reader)
        accs[fname[:-4]] = [float(row[2]) for row in csv_reader]

last_vals = {}
for threshold in thresholds:
    sparsity = sparsities[threshold]
    acc = accs[threshold]

    # Build the plot
    plt.title(threshold)
    plt.xlabel('epochs')
    plt.ylabel('value')
    plt.plot(np.arange(150), sparsity, label='sparsity')
    plt.plot(np.arange(150), acc, label='acc')
    plt.legend()
    plt.grid()
    # plt.savefig('./plots/' + threshold); plt.show();
    plt.clf()

    # Now grab the last val of sparsity and acc
    last_vals[threshold] = (acc[-1], sparsity[-1])

# last_vals['no_pruning_final'] = (accs['no_pruning'][-1], sparsities['no_pruning'][-1])
# last_vals['no_pruning_earlystop'] = (max(accs['no_pruning']), max(sparsities['no_pruning']))

for key in last_vals:
    acc = last_vals[key][0]
    sparsity = last_vals[key][1]
    plt.plot([sparsity], [acc], marker='o', markersize=10, label=key)

plt.legend()
plt.grid()
plt.xlabel('Sparsity')
plt.ylabel('Test acc.')
plt.title('Sparsity/accuracy tradeoff for various thresholds')
plt.show()