import os
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
import argparse

# CLI arguments
parser = argparse.ArgumentParser(description="Plot export results with selectable variables.")
parser.add_argument('-x', type=str, default='FPS', help='Variable for x-axis (default: FPS)')
parser.add_argument('-y', type=str, default='mAP50-95', help='Variable for y-axis (default: mAP50-95)')
parser.add_argument('-c', '--cursor', action='store_true', help='Enable cursor for hover tooltips')
parser.add_argument('-b', '--base_dir', type=str, default='my_models/sloppy_results', help='Base directory for results')
args = parser.parse_args()

# Set the base directory
base_dir = args.base_dir

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

print(f"Collected results for {len(results)} classes from {base_dir}")

# Pretty print first few results
# print(results)

# Prepare colormap
class_names = list(results.keys())
num_classes = len(class_names)
cmap = plt.get_cmap('hsv')
colors = [cmap(i / (num_classes + 1)) for i in range(1, num_classes + 1)]

# Plotting
for idx, (class_name, prunes) in enumerate(results.items()):
    x_vals = []
    y_vals = []
    for prune_rate, data in sorted(prunes.items(), key=lambda x: float(x[0])):
        x_vals.append(data.get(args.x, 0))
        y_vals.append(data.get(args.y, 0))
    plt.plot(x_vals, y_vals, marker='o', label=class_name, color=colors[idx])

plt.xlabel(args.x)
plt.ylabel(args.y)
plt.title(f'{args.y} vs {args.x}')
plt.legend()
plt.grid(True)

if args.cursor:
    mplcursors.cursor(hover=True)

    # Add tooltips showing prune_rate value
    for idx, (class_name, prunes) in enumerate(results.items()):
        x_vals = []
        y_vals = []
        prune_rate_labels = []
        for prune_rate, data in sorted(prunes.items(), key=lambda x: float(x[0])):
            x_vals.append(data.get(args.x, 0))
            y_vals.append(data.get(args.y, 0))
            prune_rate_labels.append(prune_rate)
        for x, y, label in zip(x_vals, y_vals, prune_rate_labels):
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=8)

plt.show()