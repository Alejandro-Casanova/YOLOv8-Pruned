import json

# Open averaged results file
averaged_results_file = 'my_models/aggregated_results/averaged_results.json'
with open(averaged_results_file, 'r') as file:
    averaged_results = json.load(file)

for run_class, prune_ratio_dict in averaged_results.items():
    for prune_ratio, results in prune_ratio_dict.items():
        inference_time = results['inference speed (ms)']
        fps = results['FPS']

        # Build path to file to correct
        file_path = f'my_models/sloppy_results/{run_class}/{prune_ratio}.json'
        with open(file_path, 'r') as file:
            sloppy_results = json.load(file)
        # Correct the inference time and FPS
        sloppy_results['inference speed (ms)'] = inference_time
        sloppy_results['FPS'] = fps
        # Write the corrected results back to the file
        with open(file_path, 'w') as file:
            json.dump(sloppy_results, file, indent=4)
    