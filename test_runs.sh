#!/bin/bash

python yolov8_pruning.py --max-map-drop=0.5 
python yolov8_pruning.py --max-map-drop=0.5 --epochs=25

python yolov8_pruning.py --model=yolov8s.pt --iterative-steps=20 --target-prune-rate=0.5 --max-map-drop=0.8 --epochs=100
python yolov8_pruning.py --model=yolov8n.pt --iterative-steps=5 --target-prune-rate=0.9 --max-map-drop=0.8 --epochs=10

python yolov8_pruning.py --model=yolov8n.pt --data=coco.yaml --iterative-steps=1 --target-prune-rate=0.2 --max-map-drop=0.5 --epochs=100