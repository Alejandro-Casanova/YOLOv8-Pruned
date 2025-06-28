import os
import json
import sys
import matplotlib.pyplot as plt
import numpy as np
import mplcursors
import argparse

label_names = {
    'FPS': 'FPS',
    'mAP50-95': 'mAP50-95 (%)',
    'mAP50': 'mAP50 (%)',
    'mAP75': 'mAP75 (%)',
    'model_size_kb': 'Model Size (KB)',
    'inference speed (ms)': 'Inference Speed (ms)',
}

label_names_for_title = {
    'FPS': 'FPS',
    'mAP50-95': 'mAP50-95',
    'mAP50': 'mAP50',
    'mAP75': 'mAP75',
    'model_size_kb': 'Model Size',
    'inference speed (ms)': 'Inference Speed',
}

# CLI arguments
parser = argparse.ArgumentParser(description="Plot export results with selectable variables.")
parser.add_argument('-x', type=str, default='FPS', help='Variable for x-axis (default: FPS)')
parser.add_argument('-y', type=str, default='mAP50-95', help='Variable for y-axis (default: mAP50-95)')
parser.add_argument('-c', '--cursor', action='store_true', help='Enable cursor for hover tooltips')
parser.add_argument('-b', '--base-dir', type=str, default='my-models/sloppy_results', help='Base directory for results')
parser.add_argument('-s', '--save-fig', action='store_true', help='Save the figure to a file')
parser.add_argument('-i', '--interactive', action='store_true', help='Enable interactive mode') 
parser.add_argument('-a', '--axis_limits', action='store_true', help='Set axis limits for the plot')
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
    plt.plot(x_vals, y_vals, marker='o', label=class_name, color=colors[idx], markersize=4)

plt.xlabel(label_names[args.x])
plt.ylabel(label_names[args.y])
plt.title(f'{label_names_for_title[args.y]} vs {label_names_for_title[args.x]}')

# Reorder legend handles and labels
handles, labels = plt.gca().get_legend_handles_labels()
order = [6,7,4,5,2,3,0,1]  # Custom order for legend
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order])

plt.grid(True)

if args.cursor:
    # mplcursors.cursor(hover=True)

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
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(0,5), ha='center', fontsize=6)

if args.axis_limits:
    # Set axis limits if specified
    if args.x == 'FPS':
        plt.xlim(0, 60)
        if args.y == 'mAP50':
            plt.ylim(0.4, 0.95)
    elif args.x == 'model_size_kb':
        plt.xlim(500, 4000)
        if args.y == 'mAP50':
            plt.ylim(0.0, 0.95)

if args.save_fig:
    # Save the plot as an eps file
    output_path = os.path.join(
        "export-results", 
        f"{args.y}_vs_{args.x}" + 
        ("_cursors" if args.cursor else "") + 
        ("_zoomed" if args.axis_limits else "") +
        ".eps"
    )
    plt.gcf().set_size_inches(11.06, 5.49)  # Set figure dimensions (width, height) in inches
    plt.savefig(output_path, format='eps', dpi=300, bbox_inches='tight')

if args.interactive:
    plt.show()