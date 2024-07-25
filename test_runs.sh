#!/bin/bash

: ' pruning_perf_change 11-15 (coco-minitrain-10k)
python yolov8_pruning.py --lrf=1.0 
python yolov8_pruning.py  # lrf = 0.01 around .5% worse
python yolov8_pruning.py --epochs=200  # under 1% acc improved for twice epochs
python yolov8_pruning.py --epochs=200 --target-prune-rate=0.75
python yolov8_pruning.py --iterative-steps=2 # over 1% improved for iterative
'

: ' pruning_perf_change 16 (coco-minitrain-25k)
python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml
# Over 5% improved from using larger dataset
'

: ' pruning_perf_change 17 (coco-minitrain-25k) (twice as mamy epochs as before)
python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --prune-method=group_norm --epochs=200
# Over 1% improved by using x2 epochs
'

: ' pruning_perf_change 18 (coco-minitrain-25k) (sparse learning, group_sl method) 100+100 epochs
python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml
# Initial drop in mAP, but end mAP is .5% better
'

: '
25/07/2024
'

: ' pruning_perf_change 19 (coco-minitrain-25k) (sparse learning, group_sl method) 
Same as 18, no weight decay
'
python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --weight-decay=0.0

: ' pruning_perf_change 20 (coco-minitrain-25k) (sparse learning, group_sl method) 
Same as 18, smaller reg coef
'
python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --reg=1e-5

: ' pruning_perf_change 21 (coco-minitrain-25k) (sparse learning, group_sl method) 
19 and 20 together
'
python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --reg=1e-5 --weight-decay=0.0

: ' pruning_perf_change 22 (coco-minitrain-25k)
Same as 16, half batch size
'
python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --batch=16

: ' pruning_perf_change 23 (coco-minitrain-25k)
Same as 17, twice the epochs
'
python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --prune-method=group_norm --epochs=400

: ' pruning_perf_change 24 (coco-minitrain-25k)
Same as 17, no lr decay
'
python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --prune-method=group_norm --epochs=200 --lrf=1.0

: ' pruning_perf_change 25 (coco-minitrain-25k)
Same as 23, no data augmentation
'
python yolov8_pruning.py --cfg-file=my_config_files/no_augment.yaml --data=my_datasets/coco-minitrain-25k.yaml --prune-method=group_norm --epochs=200

: ' pruning_perf_change 26 (coco-minitrain-25k) 
Same as 17, but iterative (equivalent epochs)
'
python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --prune-method=group_norm --epochs=50 --iterative-steps=4

: ' pruning_perf_change 27 (coco full dataset) 
Same as 16, full dataset
'
python yolov8_pruning.py --data=coco.yaml --prune-method=group_norm

: ' pruning_perf_change 28 (coco full dataset) 
Same as 27, twice the epochs
'
python yolov8_pruning.py --data=coco.yaml --prune-method=group_norm --epochs=200

: ' pruning_perf_change 29 (coco full dataset) 
Same as 27, constant lr
'
python yolov8_pruning.py --data=coco.yaml --prune-method=group_norm --lrf=1.0

: ' pruning_perf_change 17 (coco-minitrain-25k) 
Same as 17 but 8 worker threads for dataloaders
'
python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --prune-method=group_norm --epochs=200 --workers=8
