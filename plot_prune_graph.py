import pandas as pd
import matplotlib.pyplot as plt
import json
import glob
import os

# Step 1: Define the directory containing the JSON files
directory_path = '/home/alex/Desktop/ultralytics/selected_prune_results'

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
df['prune-ratio'] = df['x'].apply(lambda x: x[0] - x[1])
df['acc_drop'] = df['y1'].apply(lambda x: (x[1] - x[0])*100.0)
df['acc'] = df['y1'].apply(lambda x: x[1] * 100.0)
df['speedup'] = df['y2'].apply(lambda x: x[0] / x[1])

# Step 6: Print the DataFrame (optional)
print(df)

Y_VAR_TO_PLOT = 'speedup'
Y_LABEL = "Speed Up"

# Step 7: Plot the points
plt.figure(figsize=(10, 6))
scatter = plt.scatter(df['prune-ratio'], df[Y_VAR_TO_PLOT], alpha=0.7, cmap='viridis', edgecolors='w', linewidth=2.5)
# scatter = plt.scatter(df['prune-ratio'], df['acc'])

# Setp 8: Style

# # Add a colorbar
# plt.colorbar(scatter, label='Color Scale')

# Title and Axis Labels
plt.title('Prune Results', fontsize=16, fontweight='bold')
plt.xlabel('Prune Ratio', fontsize=14)
plt.ylabel(Y_LABEL, fontsize=14)

# Add gridlines
plt.grid(True, linestyle='--', alpha=0.7)

# Add annotations for each point
for i in range(len(df)):
    plt.annotate(f'({df["prune-ratio"][i]:.2f}, {df[Y_VAR_TO_PLOT][i]:.2f})', (df['prune-ratio'][i], df[Y_VAR_TO_PLOT][i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)

# Show the plot
plt.tight_layout()
plt.show()
