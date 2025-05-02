import pandas as pd
import matplotlib.pyplot as plt
import json
import glob
import os
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot Prune Graph")
    parser.add_argument('-dir', '--directory', type=str, default=".\\selected_prune_results", 
                        help='Pretrained pruning target model file')
    parser.add_argument('-y', '--y-axis', type=str, default='speedup',
                        choices=['speedup', 'prune_ratio', 'accuracy', 'acc_drop'],
                        help='Set the logging level')
    
    args = parser.parse_args()

    # Step 1: Define the directory containing the JSON files
    directory_path = args.directory

    # Step 2: Get all .json files in the directory
    json_files = glob.glob(os.path.join(directory_path, '*.json'))

    # Step 3: Initialize a list to hold the dictionaries
    data = []

    # Step 4: Read each JSON file and extract the dictionaries
    for json_file in json_files:
        with open(json_file, 'r') as file:
            # Load the JSON data
            json_data = json.load(file)
            # Assuming each json_data is a dictionary with 'x' and 'y' keys
            data.append(json_data)

    # Step 5: Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(data)
    df['prune_ratio'] = df['x'].apply(lambda x: x[0] - x[1])
    df['acc_drop'] = df['y1'].apply(lambda x: (x[1] - x[0])*100.0)
    df['accuracy'] = df['y1'].apply(lambda x: x[1] * 100.0)
    df['speedup'] = df['y2'].apply(lambda x: x[0] / x[1])

    # Step 6: Print the DataFrame (optional)
    print(df)

    # Step 7: Plot the points
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(df['prune_ratio'], df[args.y_axis], alpha=0.7, cmap='viridis', edgecolors='w', linewidth=2.5)
    # scatter = plt.scatter(df['prune_ratio'], df['accuracy'])

    # Step 8: Style

    # # Add a colorbar
    # plt.colorbar(scatter, label='Color Scale')

    # Title and Axis Labels
    plt.title('Prune Results', fontsize=16, fontweight='bold')
    plt.xlabel('Prune Ratio', fontsize=14)
    plt.ylabel(args.y_axis, fontsize=14)

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add annotations for each point
    for i in range(len(df)):
        plt.annotate(f'({df["prune_ratio"][i]:.2f}, {df[args.y_axis][i]:.2f})', (df['prune_ratio'][i], df[args.y_axis][i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

    # Show the plot
    plt.tight_layout()
    plt.show()
