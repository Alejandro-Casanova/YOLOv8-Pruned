from copy import deepcopy
import sys
import torch
from ultralytics import YOLO
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml
from yolov8_pruning import C2f_v2
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="YOLOv8 validation and training script")
    parser.add_argument('-d', '--dataset', type=str, default="my_config_files/trash_config.yaml", help='Path to dataset config YAML')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to YOLOv8 model weights')

    args = parser.parse_args()
    
    # Load Config
    cfg = yaml_load(check_yaml(args.dataset))
    batch = cfg['batch']

    # Load a pretrained model
    model = YOLO(args.model)  

    cfg['batch'] = 1

    # cfg['split'] = 'test' # Default is val
    validation_model = deepcopy(model)
    metric = validation_model.val(**cfg)
    print(f"mAP50-95: {metric.box.map}")
    print(f"mAP50: {metric.box.map50}")
    print(f"mAP75: {metric.box.map75}")
    print(f"mAP50-95 per class: {metric.box.maps}")

    sys.exit()

    # cfg['batch'] = batch

    # results = model.train(**cfg)
    # print(f"RESULTS: ")
    # print(results)