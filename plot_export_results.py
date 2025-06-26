import os
import json
import matplotlib.pyplot as plt
import numpy as np
import mplcursors

# Set the base directory
base_dir = "my_models/export_results"

# Collect data
results = {}
for class_name in os.listdir(base_dir):
    class_dir = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    results[class_name] = {}
    for fname in os.listdir(class_dir):
        if fname.endswith('.json'):
            prune_rate = fname.replace('.json', '')
            with open(os.path.join(class_dir, fname), 'r') as f:
                data = json.load(f)
            results[class_name][prune_rate] = data

# Prepare colormap
class_names = list(results.keys())
num_classes = len(class_names)
cmap = plt.get_cmap('hsv')
colors = [cmap(i / (num_classes + 1)) for i in range(1, num_classes + 1)]

# Plotting
for idx, (class_name, prunes) in enumerate(results.items()):
    prune_rates = []
    accuracies = []
    for prune_rate, data in sorted(prunes.items(), key=lambda x: float(x[0])):
        prune_rates.append(data.get('FPS', 0))
        accuracies.append(data.get('mAP50-95', 0))
    plt.plot(prune_rates, accuracies, marker='o', label=class_name, color=colors[idx])

plt.xlabel('FPS')
plt.ylabel('mAP50-95')
plt.title('mAP50-95 vs FPS')
plt.legend()
plt.grid(True)
mplcursors.cursor(hover=True)

# Add tooltips showing prune_rate value
for idx, (class_name, prunes) in enumerate(results.items()):
    prune_rates = []
    accuracies = []
    prune_rate_labels = []
    for prune_rate, data in sorted(prunes.items(), key=lambda x: float(x[0])):
        prune_rates.append(data.get('FPS', 0))
        accuracies.append(data.get('mAP50-95', 0))
        prune_rate_labels.append(prune_rate)
    for x, y, label in zip(prune_rates, accuracies, prune_rate_labels):
        plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

plt.show()