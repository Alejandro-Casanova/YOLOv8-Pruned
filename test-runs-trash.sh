: ' pruning_perf_change 33 (trash dataset) 
Prune 50% and retrain for 200 epochs. Pretrained model, no sparsity learning
# mAP: 66.13% -> 54.51%
'
# python yolov8-pruning.py --model=my-models/trash_yolov8_pretrained.pt --cfg-file=my-config-files/trash_config.yaml --prune-method=group_norm

: ' pruning_perf_change 34 (trash dataset) 
Sparsity learn for 200 epochs, prune 50% and retrain for 200 epochs
# mAP: 65.64% -> 53.66%
'
# python yolov8-pruning.py --cfg-file=my-config-files/trash_config.yaml

: ' pruning_perf_change 35 (trash dataset) 
Prune 50% and retrain for 300 epochs. Pretrained model, WITH sparsity learning
# mAP: 65.62% -> 55.98%
'
# python yolov8-pruning.py --model=my-models/trash_yolov8n_pretrained_sl.pt --cfg-file=my-config-files/trash_config.yaml --prune-method=group_norm

: ' pruning_perf_change 36 (trash dataset) 
Prune 50% and retrain for 300 epochs. Pretrained model, WITHOUT sparsity learning
# mAP: 66.13% -> 56.20%
'
# python yolov8-pruning.py --cfg-file=my-config-files/trash_config.yaml --prune-method=group_norm

: ' pruning_perf_change 37 (trash dataset) 
Prune 50% and retrain for 300 epochs. Pretrained model, WITH sparsity learning and Global Pruning
# mAP: 65.62% -> 43.46%
'
# python yolov8-pruning.py --model=my-models/trash_yolov8n_pretrained_sl.pt --cfg-file=my-config-files/trash_config.yaml --prune-method=group_norm --global-pruning

: ' pruning_perf_change 38 (trash dataset) 
Prune 10% and retrain for 300 epochs. Pretrained model, WITH sparsity learning and Global Pruning
# mAP: 65.62% -> 62.11%
'
# python yolov8-pruning.py --target-prune-rate=0.1 --model=my-models/trash_yolov8n_pretrained_sl.pt --cfg-file=my-config-files/trash_config.yaml --prune-method=group_norm --global-pruning

: ' pruning_perf_change 39 (trash dataset) 
Prune 20% and retrain for 300 epochs. Pretrained model, WITH sparsity learning and Global Pruning
# mAP: 65.62% -> 61.90%
'
# python yolov8-pruning.py --target-prune-rate=0.2 --model=my-models/trash_yolov8n_pretrained_sl.pt --cfg-file=my-config-files/trash_config.yaml --prune-method=group_norm --global-pruning

: ' pruning_perf_change 40 (trash dataset) 
Prune 30% and retrain for 300 epochs. Pretrained model, WITH sparsity learning and Global Pruning
# mAP: 65.62% -> 58.23%
'
# python yolov8-pruning.py --target-prune-rate=0.3 --model=my-models/trash_yolov8n_pretrained_sl.pt --cfg-file=my-config-files/trash_config.yaml --prune-method=group_norm --global-pruning

: ' pruning_perf_change 41 (trash dataset) 
Prune 60% and retrain for 300 epochs. Pretrained model, WITH sparsity learning and Global Pruning
# mAP: 65.62% -> 27.20%
'
# python yolov8-pruning.py --target-prune-rate=0.6 --model=my-models/trash_yolov8n_pretrained_sl.pt --cfg-file=my-config-files/trash_config.yaml --prune-method=group_norm --global-pruning

: ' pruning_perf_change 42 (trash dataset) 
Prune 70% and retrain for 300 epochs. Pretrained model, WITHOUT sparsity learning
# mAP: 66.13% -> 43.50%
'
# python yolov8-pruning.py --target-prune-rate=0.7 --model=my-models/trash_yolov8n_pretrained.pt --cfg-file=my-config-files/trash_config.yaml --prune-method=group_norm

: ' pruning_perf_change 43 (trash dataset) 
Prune 80% and retrain for 300 epochs. Pretrained model, WITHOUT sparsity learning
# mAP: 66.13% -> 30.13%
'
# python yolov8-pruning.py --target-prune-rate=0.8 --model=my-models/trash_yolov8n_pretrained.pt --cfg-file=my-config-files/trash_config.yaml --prune-method=group_norm

: ' pruning_perf_change 44 (trash dataset) 
Prune 90% and retrain for 300 epochs. Pretrained model, WITHOUT sparsity learning
# mAP: 66.13% -> 3.58%
'
# python yolov8-pruning.py --target-prune-rate=0.9 --model=my-models/trash_yolov8n_pretrained.pt --cfg-file=my-config-files/trash_config.yaml --prune-method=group_norm

: ' pruning_perf_change 45 (trash dataset) 
Prune 10% and retrain for 300 epochs. Pretrained model, WITHOUT sparsity learning
# mAP: 66.13% -> 64.98% 
'
# python yolov8-pruning.py --target-prune-rate=0.1 --model=my-models/trash_yolov8n_pretrained.pt --cfg-file=my-config-files/trash_config.yaml --prune-method=group_norm

: ' pruning_perf_change 46 (trash dataset) 
Prune 20% and retrain for 300 epochs. Pretrained model, WITHOUT sparsity learning
# mAP: 66.13% -> 65.00%
'
# python yolov8-pruning.py --target-prune-rate=0.2 --model=my-models/trash_yolov8n_pretrained.pt --cfg-file=my-config-files/trash_config.yaml --prune-method=group_norm

: ' pruning_perf_change 47 (trash dataset) 
Prune 30% and retrain for 300 epochs. Pretrained model, WITHOUT sparsity learning
# mAP: 66.13% -> 63.84%
'
python yolov8-pruning.py --target-prune-rate=0.3 --model=my-models/trash_yolov8n_pretrained.pt --cfg-file=my-config-files/trash_config.yaml --prune-method=group_norm

: ' pruning_perf_change 48 (trash dataset) 
Prune 60% and retrain for 300 epochs. Pretrained model, WITHOUT sparsity learning
# mAP: 66.13% -> 51.34%
'
python yolov8-pruning.py --target-prune-rate=0.6 --model=my-models/trash_yolov8n_pretrained.pt --cfg-file=my-config-files/trash_config.yaml --prune-method=group_norm
