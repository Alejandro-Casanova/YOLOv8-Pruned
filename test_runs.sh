#!/bin/bash

: ' pruning_perf_change 11-15 (coco-minitrain-10k)
python yolov8_pruning.py --lrf=1.0 
python yolov8_pruning.py 
python yolov8_pruning.py --epochs=200 
python yolov8_pruning.py --epochs=200 --target-prune-rate=0.75
python yolov8_pruning.py --iterative-steps=2
'

: ' pruning_perf_change 16 (coco-minitrain-25k)
python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml
'

python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --prune-method=group_norm --epochs=200
python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml
