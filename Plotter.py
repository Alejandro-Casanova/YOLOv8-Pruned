import argparse
import glob
import json
import logging
import os
import pprint
import sys
from matplotlib import pyplot as plt
import numpy as np


class Plotter:

    def __init__(self):
        self._currentPlot = 0 # Plot index corresponding to current run
        self._out_dir = "prune-results"
        if not os.path.exists(self._out_dir):
            os.makedirs(self._out_dir)
        self._filename = f"{self._out_dir}/pruning_perf_change"

        # Find next free index to save plot
        while glob.glob('{}_{:d}.*'.format(self._filename, self._currentPlot)):
            self._currentPlot += 1
        
        # Append index to filename
        self._filename = '{}_{:d}'.format(self._filename, self._currentPlot)

        self._logger = logging.getLogger(__name__)

    def append_dict_to_log(
        self, 
        dict: dict = {},
        description: str = "" 
    ):
        # Save the console args passed to script
        with open('{}.txt'.format(self._filename), 'a') as f:
            #f.write(json.dumps(dict))
            f.write("\n" + description + "\n")
            pprint.pprint(dict, f)

    def save_pruning_performance_graph(
        self,
        x, y1, y2, y3,
        subTitleStr: str = "",
        save_results_json: bool = True
    ) -> None: 
        """
        Draw performance change graph
        Each call will overwrite the graph corresponding to the current run. On the next run, the graph index will be incremented.
        Also saves plot data as json file.

        Parameters
        ----------
        x : List
            Parameter numbers of all pruning steps
        y1 : List
            mAPs after fine-tuning of all pruning steps
        y2 : List
            FLOPs of all pruning steps
        y3 : List
            mAPs after pruning (not fine-tuned) of all pruning steps

        Returns
        -------

        """
        try:
            plt.style.use("ggplot")
        except:
            pass

        x, y1, y2, y3 = np.array(x), np.array(y1), np.array(y2), np.array(y3)
        y2_ratio = y2 / y2[0]

        # create the figure and the axis object
        fig, ax = plt.subplots(figsize=(8, 6))

        # plot the pruned mAP and recovered mAP
        ax.set_xlabel('Tamaño del model (1 - ratio de poda) %')
        ax.set_ylabel('mAP')
        ax.plot(x, y1, label='mAP tras re-entrenar')
        ax.scatter(x, y1)
        ax.plot(x, y3, color='tab:gray', label='mAP tras podar')
        ax.scatter(x, y3, color='tab:gray')

        # create a second axis that shares the same x-axis
        ax2 = ax.twinx()

        # plot the second set of data
        ax2.set_ylabel('FLOPs')
        ax2.plot(x, y2_ratio, color='tab:orange', label='FLOPs')
        ax2.scatter(x, y2_ratio, color='tab:orange')

        # add a legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='best')

        # set plot limits
        ax.set_xlim(105, -5)
        ax.set_ylim(0, max(y1) + 0.05)
        ax2.set_ylim(0.05, 1.05)

        # calculate the highest and lowest points for each set of data
        max_y1_idx = np.argmax(y1)
        min_y1_idx = np.argmin(y1)
        max_y2_idx = np.argmax(y2)
        min_y2_idx = np.argmin(y2)
        max_y1 = y1[max_y1_idx]
        min_y1 = y1[min_y1_idx]
        max_y2 = y2_ratio[max_y2_idx]
        min_y2 = y2_ratio[min_y2_idx]

        # add text for the highest and lowest values near the points
        ax.text(x[max_y1_idx], max_y1 - 0.05, f'max mAP = {100.0*max_y1:.2f}%', fontsize=10)
        ax.text(x[min_y1_idx], min_y1 + 0.02, f'min mAP = {100.0*min_y1:.2f}%', fontsize=10)
        ax2.text(x[max_y2_idx], max_y2 - 0.05, f'max FLOPs = {max_y2 * y2[0] / 1e9:.2f}G', fontsize=10)
        ax2.text(x[min_y2_idx], min_y2 + 0.02, f'min FLOPs = {min_y2 * y2[0] / 1e9:.2f}G', fontsize=10)

        plt.title('Comparación del mAP y los FLOPs con respecto al ratio de poda')
        plt.suptitle(subTitleStr)
        
        # Write/overwrite the plot corresponding to current run
        plt.savefig('{}.png'.format(self._filename))

        if not save_results_json:
            return
        
        # Save plot data as json file too
        dict_to_save = {
            "x": x.tolist(),
            "y1": y1.tolist(),
            "y2": y2.tolist(),
            "y3": y3.tolist()
        }

        with open('{}.json'.format(self._filename), "w") as fp:
            json.dump(dict_to_save , fp, indent=4) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pruning Plotter")

    # Default parameters from Ultralytics Yolo config file
    parser.add_argument('--results-json', type=str, required=True,
                        help='Pruning results json file')
    parser.add_argument('--subtitle', type=str, default="", 
                        help='Plot subtitle string.')
    
    args = parser.parse_args()
    
    json_dict = None

    with open(args.results_json, "r") as fp:
        json_dict = json.load(fp) 

    if json_dict is None:
        sys.exit(1)
    
    plotter = Plotter()

    plotter.save_pruning_performance_graph(
        json_dict['x'], json_dict['y1'], json_dict['y2'], json_dict['y3'],
        args.subtitle, False
    )

    sys.exit(0)