import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Set the base directory
base_dir = "my-models/export_results"

# Collect data
results = {}
for class_name in os.listdir(base_dir):
    class_dir = os.path.join(base_dir, class_name)
    if not os.path.isdir(class_dir):
        print(f"Skipping {class_dir}, not a directory.")
        continue
    results[class_name] = {}
    for prune_ratio in os.listdir(class_dir):
        prune_dir = os.path.join(class_dir, prune_ratio)
        if not os.path.isdir(class_dir):
            print(f"Skipping {prune_dir}, not a directory.")
            continue
        results[class_name][prune_ratio] = {}
        for fname in os.listdir(prune_dir):
            if fname.endswith('.json'):
                seed = fname.replace('.json', '')
                with open(os.path.join(prune_dir, fname), 'r') as f:
                    data = json.load(f)
                results[class_name][prune_ratio][seed] = data

# Average final data across seeds
averaged_results = {}
for class_name, prune_ratios in results.items():
    averaged_results[class_name] = {}
    for prune_ratio, seeds in prune_ratios.items():
        fps_list = []
        latency_list = []
        for seed, data in seeds.items():
            fps_list.append(data.get('FPS', 0))
            latency_list.append(data.get('inference speed (ms)', 0))
        averaged_results[class_name][prune_ratio] = {
            'FPS': np.mean(fps_list),
            'inference speed (ms)': np.mean(latency_list)
        }

# Save averaged results to a new JSON file
output_file = os.path.join("my-models/aggregated_results", 'averaged_results.json')
with open(output_file, 'w') as f:
    json.dump(averaged_results, f, indent=4)
print(f"Averaged results saved to {output_file}")
