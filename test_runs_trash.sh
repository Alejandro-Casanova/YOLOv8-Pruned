: ' pruning_perf_change 33 (trash dataset) 
Prune 50% and retrain for 200 epochs. Pretrained model, no sparsity learning
'
# python yolov8_pruning.py --model=my_models/trash_yolov8_pretrained.pt --cfg-file=my_config_files/trash_config.yaml --prune-method=group_norm

: ' pruning_perf_change 34 (trash dataset) 
Sparsity learn for 200 epochs, prune 50% and retrain for 200 epochs
'
# python yolov8_pruning.py --cfg-file=my_config_files/trash_config.yaml

: ' pruning_perf_change 35 (trash dataset) 
Prune 50% and retrain for 300 epochs. Pretrained model, WITH sparsity learning
'
python yolov8_pruning.py --model=my_models/trash_yolov8_pretrained_sl.pt --cfg-file=my_config_files/trash_config.yaml --prune-method=group_norm


# NEXT try global pruning