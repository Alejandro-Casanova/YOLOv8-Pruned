: ' pruning_perf_change 33 (trash dataset) 
Prune 50% and retrain for 200 epochs. Pretrained model, no sparsity learning
# mAP: 66.13% -> 54.51%
'
# python yolov8_pruning.py --model=my_models/trash_yolov8_pretrained.pt --cfg-file=my_config_files/trash_config.yaml --prune-method=group_norm

: ' pruning_perf_change 34 (trash dataset) 
Sparsity learn for 200 epochs, prune 50% and retrain for 200 epochs
# mAP: 65.64% -> 53.66%
'
# python yolov8_pruning.py --cfg-file=my_config_files/trash_config.yaml

: ' pruning_perf_change 35 (trash dataset) 
Prune 50% and retrain for 300 epochs. Pretrained model, WITH sparsity learning
# mAP: 65.62% -> 55.98%
'
# python yolov8_pruning.py --model=my_models/trash_yolov8n_pretrained_sl.pt --cfg-file=my_config_files/trash_config.yaml --prune-method=group_norm

: ' pruning_perf_change 36 (trash dataset) 
Prune 50% and retrain for 300 epochs. Pretrained model, WITHOUT sparsity learning
# mAP: 
'
python yolov8_pruning.py --cfg-file=my_config_files/trash_config.yaml --prune-method=group_norm

: ' pruning_perf_change 37 (trash dataset) 
Prune 50% and retrain for 300 epochs. Pretrained model, WITH sparsity learning and Global Pruning
# mAP: 
'
python yolov8_pruning.py --model=my_models/trash_yolov8n_pretrained_sl.pt --cfg-file=my_config_files/trash_config.yaml --prune-method=group_norm --global-pruning

: ' pruning_perf_change 38 (trash dataset) 
Prune 10% and retrain for 300 epochs. Pretrained model, WITH sparsity learning and Global Pruning
# mAP: 
'
python yolov8_pruning.py --target-prune-rate=0.1 --model=my_models/trash_yolov8n_pretrained_sl.pt --cfg-file=my_config_files/trash_config.yaml --prune-method=group_norm --global-pruning

: ' pruning_perf_change 39 (trash dataset) 
Prune 20% and retrain for 300 epochs. Pretrained model, WITH sparsity learning and Global Pruning
# mAP: 
'
python yolov8_pruning.py --target-prune-rate=0.2 --model=my_models/trash_yolov8n_pretrained_sl.pt --cfg-file=my_config_files/trash_config.yaml --prune-method=group_norm --global-pruning

: ' pruning_perf_change 40 (trash dataset) 
Prune 30% and retrain for 300 epochs. Pretrained model, WITH sparsity learning and Global Pruning
# mAP: 
'
python yolov8_pruning.py --target-prune-rate=0.3 --model=my_models/trash_yolov8n_pretrained_sl.pt --cfg-file=my_config_files/trash_config.yaml --prune-method=group_norm --global-pruning

: ' pruning_perf_change 41 (trash dataset) 
Prune 60% and retrain for 300 epochs. Pretrained model, WITH sparsity learning and Global Pruning
# mAP: 
'
python yolov8_pruning.py --target-prune-rate=0.6 --model=my_models/trash_yolov8n_pretrained_sl.pt --cfg-file=my_config_files/trash_config.yaml --prune-method=group_norm --global-pruning

: ' pruning_perf_change 42 (trash dataset) 
Prune 70% and retrain for 300 epochs. Pretrained model, WITH sparsity learning and Global Pruning
# mAP: 
'
python yolov8_pruning.py --target-prune-rate=0.7 --model=my_models/trash_yolov8n_pretrained_sl.pt --cfg-file=my_config_files/trash_config.yaml --prune-method=group_norm --global-pruning

: ' pruning_perf_change 43 (trash dataset) 
Prune 80% and retrain for 300 epochs. Pretrained model, WITH sparsity learning and Global Pruning
# mAP: 
'
python yolov8_pruning.py --target-prune-rate=0.8 --model=my_models/trash_yolov8n_pretrained_sl.pt --cfg-file=my_config_files/trash_config.yaml --prune-method=group_norm --global-pruning

: ' pruning_perf_change 44 (trash dataset) 
Prune 90% and retrain for 300 epochs. Pretrained model, WITH sparsity learning and Global Pruning
# mAP: 
'
python yolov8_pruning.py --target-prune-rate=0.9 --model=my_models/trash_yolov8n_pretrained_sl.pt --cfg-file=my_config_files/trash_config.yaml --prune-method=group_norm --global-pruning
