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
# .3% better, .15% better after sparse learn
'
# python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --weight-decay=0.0


: ' pruning_perf_change 20 (coco-minitrain-25k) (sparse learning, group_sl method) 
Same as 18, smaller reg coef
# .05% worse, .09% better after sparse learn (leave coefficient as it was)
'
# python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --reg=1e-5


: ' pruning_perf_change 21 (coco-minitrain-25k) (sparse learning, group_sl method) 
19 and 20 together
# .7% better than 20 and .3% better than 19 (actually best option!!)
'
# python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --reg=1e-5 --weight-decay=0.0


: ' pruning_perf_change 22 (coco-minitrain-25k)
Same as 18, half batch size
# .4% worse and 20 minutes slower
'
# python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --batch=16


: ' pruning_perf_change 23 (coco-minitrain-25k)
Same as 17, twice the epochs
# Over .7% better
'
# python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --prune-method=group_norm --epochs=400


: ' pruning_perf_change 24 (coco-minitrain-25k)
Same as 17, no lr decay
# Over 1% WORSE
'
# python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --prune-method=group_norm --epochs=200 --lrf=1.0


: ' pruning_perf_change 25 (coco-minitrain-25k)
Same as 17, no data augmentation
# Around 9% WORSE!!!! WOW (Will there be so much difference with full dataset?)
'
# python yolov8_pruning.py --cfg-file=my_config_files/no_augment.yaml --data=my_datasets/coco-minitrain-25k.yaml --prune-method=group_norm --epochs=200


: ' pruning_perf_change 26 (coco-minitrain-25k) 
Same as 17, but iterative (equivalent epochs)
# Around .1% worse hummmm... And 1hr longer runtime
'
# python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --prune-method=group_norm --epochs=50 --iterative-steps=4


: ' pruning_perf_change 27 (coco full dataset) 
Same as 16, full dataset (100 epochs)
# Over 5% better (28.95%) (runtime: 1 day 7:46)
'
# python yolov8_pruning.py --data=coco.yaml --prune-method=group_norm


: ' pruning_perf_change 28 (coco full dataset) 
Same as 27, twice the epochs
# Around .7% better (29.66%) (almost twice runtime)
'
# python yolov8_pruning.py --data=coco.yaml --prune-method=group_norm --epochs=200

: ' pruning_perf_change 29 (coco full dataset) 
Same as 27, constant lr
# Over 3% WORSE!
'
# python yolov8_pruning.py --data=coco.yaml --prune-method=group_norm --lrf=1.0

: ' pruning_perf_change 30 (coco-minitrain-25k) 
Same as 17 but 8 worker threads for dataloaders
# Around .3% better? Normal deviation between runs? Runtime is the same => FIX workers=8
'
# python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --prune-method=group_norm --epochs=200 --workers=8


: '
05/08/2024
'

: ' pruning_perf_change 31 (coco-minitrain-25k)
Group-sl, 50+50 epochs, baseline for batch size analysis (batch=32)
'
# python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --epochs=50

: ' pruning_perf_change 32 (coco-minitrain-25k)
Same as 31, half batch size
# Same precision and runtime (not better or worse)
'
# python yolov8_pruning.py --data=my_datasets/coco-minitrain-25k.yaml --epochs=50 --batch=32

# Following runs with bigger batch sizes crashed from CUDA out of memory