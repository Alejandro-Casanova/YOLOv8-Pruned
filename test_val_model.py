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
model = YOLO('/home/alex/Desktop/ultralytics/my_models/trash_yolov8n_pretrained_sl.pt')  

cfg['batch'] = 1

# cfg['split'] = 'test' # Default is val
validation_model = deepcopy(model)
metric = validation_model.val(**cfg)
print(f"mAP50-95: {metric.box.map}")
print(f"mAP50: {metric.box.map50}")
print(f"mAP75: {metric.box.map75}")
print(f"mAP50-95 per class: {metric.box.maps}")

exit()

cfg['batch'] = batch

results = model.train(**cfg)
print(f"RESULTS: ")
print(results)