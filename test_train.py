from copy import deepcopy
import torch
from ultralytics import YOLO
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_yaml
from yolov8_pruning import C2f_v2

# Load Config
cfg = yaml_load(check_yaml("my_config_files/trash_config.yaml"))
batch = cfg['batch']

# Load a pretrained model
model = YOLO(cfg['model'])  

cfg['batch'] = 1
validation_model = deepcopy(model)
metric = validation_model.val(**cfg)
results = metric.box.maps
print(f"RESULTS: ")
print(results)
exit()

cfg['batch'] = batch

results = model.train(**cfg)
print(f"RESULTS: ")
print(results)