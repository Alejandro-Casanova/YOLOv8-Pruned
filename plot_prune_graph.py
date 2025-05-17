import pandas as pd
import matplotlib.pyplot as plt
import json
import glob
import os
import argparse

axis_names_spanish = {
    'speedup': 'Aceleración',
    'prune_ratio': 'Ratio de Poda (%)',
    'accuracy': 'Precisión mAP50-95 (%)',
    'acc_drop': 'Diferencia de Precisión (%)'
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Plot Prune Graph")
    parser.add_argument('-dir', '--directory', type=str, default=".\\selected_prune_results", 
                        help='Pretrained pruning target model file')
    parser.add_argument('-y', '--y-axis', type=str, default='speedup',
                        choices=['speedup', 'prune_ratio', 'accuracy', 'acc_drop'],
                        help='Set the logging level')
    parser.add_argument('-x', '--x-axis', type=str, default='prune_ratio',
                        choices=['speedup', 'prune_ratio', 'accuracy', 'acc_drop'],
                        help='Variable to plot on the x-axis')
    parser.add_argument('-s', '--spanish', action='store_true', 
                        help='Use Spanish labels for the plot')
    parser.add_argument('-i', '--interactive', action='store_true',
                        help='Use interactive mode for the plot')
    parser.add_argument('-o', '--output', type=str, default='plot.eps',
                        help='Output file name for the plot')
    
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

    # Sort the DataFrame by 'prune_ratio'
    df = df.sort_values(by=args.x_axis)
    # Reset the index
    df = df.reset_index(drop=True)

    # Step 7: Plot the points
    plt.figure(figsize=(10, 6))
    plt.plot(df[args.x_axis], df[args.y_axis], marker='o', linestyle='-', color='#00aaff', linewidth=2, alpha=0.8)

    # Step 8: Style

    # # Add a colorbar
    # plt.colorbar(scatter, label='Color Scale')

    # Title and Axis Labels
    plt.title(
        'Prune Results' if not args.spanish else 'Resultados de la Poda', 
        fontsize=16, fontweight='bold')
    plt.xlabel(
        args.x_axis if not args.spanish else axis_names_spanish[args.x_axis], 
        fontsize=14)
    plt.ylabel(
        args.y_axis if not args.spanish else axis_names_spanish[args.y_axis], 
        fontsize=14)

    # Add grid lines
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add annotations for each point
    for i in range(len(df)):
        plt.annotate(
            f'({df[args.x_axis][i]:.2f}, {df[args.y_axis][i]:.2f})', 
            (df[args.x_axis][i], df[args.y_axis][i]), 
            textcoords="offset points", 
            xytext=(0,10), 
            ha='center', 
            fontsize=8)

    # Show the plot
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(args.output, format=args.output.split('.')[-1], dpi=300, bbox_inches='tight')

    if args.interactive:
        plt.show()
