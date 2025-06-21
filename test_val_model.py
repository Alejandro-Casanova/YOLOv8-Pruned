from copy import deepcopy
import sys
import torch
from ultralytics import YOLO
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml
from yolov8_c2v_patch import C2f_v2
import argparse
import pprint
import os
import json

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="YOLOv8 validation and training script")
    parser.add_argument('-d', '--dataset', type=str, default="my_config_files/trash_config.yaml", help='Path to dataset config YAML')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to YOLOv8 model weights')
    # parser.add_argument('-i', '--imgsz', type=int, default=640, help='Image size for validation')

    args = parser.parse_args()

    # Extract model class from suffix (e.g., imgsz320_quant8)
    a = args.model.split('.')[-2]
    b = a.split('_')
    img_size = int(b[3].replace('imgsz', '')) if 'imgsz' in b[3] else 640  # Default to 640 if not specified
    model_class = b[3] + '_' + b[4] if len(b) > 4 else b[3] 
    prune_rate = b[2]
    
    # Load Config
    cfg = yaml_load(check_yaml(args.dataset))
    batch = cfg['batch']

    # Load a pretrained model
    model = YOLO(args.model)  

    cfg['batch'] = 1
    cfg['imgsz'] = img_size
    cfg['task'] = 'detect'

    # cfg['split'] = 'test' # Default is val
    validation_model = deepcopy(model)
    metric = validation_model.val(**cfg)
    print(f"mAP50-95: {metric.box.map}")
    print(f"mAP50: {metric.box.map50}")
    print(f"mAP75: {metric.box.map75}")
    print(f"mAP50-95 per class: {metric.box.maps}")

    if hasattr(metric, 'speed'):
        print(f"Inference speed: {metric.speed['inference']} ms per image")
        print(f"FPS: {1 / metric.speed['inference'] * 1000}")
    else:
        print("Inference speed information not available.")

    # Print the model size in KB
    model_size_kb = float(os.path.getsize(args.model)) / 1024.0
    print(f"Model size: {model_size_kb:.2f} KB")

    # Save validation results to a json file in my_models/export_results/model_class.json
    results = {
        'mAP50-95': float(metric.box.map),
        'mAP50': float(metric.box.map50),
        'mAP75': float(metric.box.map75),
        'mAP50-95 per class': metric.box.maps.tolist() if hasattr(metric.box.maps, 'tolist') else list(metric.box.maps),
        'inference speed (ms)': float(metric.speed['inference']) if hasattr(metric, 'speed') else None,
        'FPS': float(1 / metric.speed['inference'] * 1000) if hasattr(metric, 'speed') else None,
        'model_size_kb': model_size_kb,
        'model_class': model_class,
    }

    os.makedirs(f'my_models/export_results/{model_class}', exist_ok=True)
    with open(f'my_models/export_results/{model_class}/{prune_rate}.json', 'w') as f:
        json.dump(results, f, indent=4)
    sys.exit()

    # cfg['batch'] = batch

    # results = model.train(**cfg)
    # print(f"RESULTS: ")
    # print(results)