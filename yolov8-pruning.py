# This code is adapted from Issue [#147](https://github.com/VainF/Torch-Pruning/issues/147), implemented by @Hyunseok-Kim0.
import argparse
from functools import partial
import glob
import inspect
import math
import os
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
import pprint
import time
from typing import List, Union
import json

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from ultralytics.models import YOLO
from ultralytics import __version__
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck
from ultralytics.nn.tasks import attempt_load_one_weight
#from ultralytics.yolo.engine.model import TASK_MAP
#from ultralytics.yolo.engine.trainer import BaseTrainer
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import yaml_load, LOGGER, RANK, DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS
# from ultralytics.yolo.utils.checks import check_yaml
from ultralytics.utils.checks import check_yaml
#from ultralytics.yolo.utils.torch_utils import initialize_weights, de_parallel
from ultralytics.utils.torch_utils import initialize_weights, de_parallel
from ultralytics.utils import checks
from ultralytics.cfg import TASK2DATA
import logging
from Plotter import Plotter
import torch_pruning as tp
from yolov8_c2v_patch import replace_c2f_with_c2f_v2

def save_model_v2(self: BaseTrainer):
    """
    Disabled half precision saving. originated from ultralytics/engine/trainer.py
    """
    import io

    import pandas as pd  # scope for faster 'import ultralytics'

    # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
    buffer = io.BytesIO()
    torch.save(
        {
            "epoch": self.epoch,
            "best_fitness": self.best_fitness,
            # 'model': deepcopy(de_parallel(self.model)), ?? 
            "model": None,  # resume and final checkpoints derive from EMA
            "ema": deepcopy(self.ema.ema),
            "updates": self.ema.updates,
            "optimizer": self.optimizer.state_dict(),
            "train_args": vars(self.args),  # save as dict
            "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
            "train_results": {k.strip(): v for k, v in pd.read_csv(self.csv).to_dict(orient="list").items()},
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
        },
        buffer,
    )
    serialized_ckpt = buffer.getvalue()  # get the serialized content to save

    # Save checkpoints
    self.last.write_bytes(serialized_ckpt)  # save last.pt
    if self.best_fitness == self.fitness:
        self.best.write_bytes(serialized_ckpt)  # save best.pt
    if (self.save_period > 0) and (self.epoch > 0) and (self.epoch % self.save_period == 0):
        (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'
    
    return
    ckpt = {
        'epoch': self.epoch,
        'best_fitness': self.best_fitness,
        'model': deepcopy(de_parallel(self.model)),
        'ema': deepcopy(self.ema.ema),
        'updates': self.ema.updates,
        'optimizer': self.optimizer.state_dict(),
        'train_args': vars(self.args),  # save as dict
        'date': datetime.now().isoformat(),
        'version': __version__}

    # Save last, best and delete
    torch.save(ckpt, self.last)
    if self.best_fitness == self.fitness:
        torch.save(ckpt, self.best)
    if (self.epoch > 0) and (self.save_period > 0) and (self.epoch % self.save_period == 0):
        torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
    del ckpt


def final_eval_v2(self: BaseTrainer):
    """
    originated from ultralytics/engine/trainer.py
    """
    for f in self.last, self.best:
        if f.exists():
            strip_optimizer_v2(f)  # strip optimizers
            if f is self.best:
                LOGGER.info(f'\nValidating {f}...')
                self.validator.args.plots = self.args.plots # Added this
                self.metrics = self.validator(model=f)
                self.metrics.pop('fitness', None)
                self.run_callbacks('on_fit_epoch_end')


def strip_optimizer_v2(f: Union[str, Path] = 'best.pt', s: str = '') -> None:
    """
    Disabled half precision saving. originated from ultralytics/utils/torch_utils.py
    """

    try:
        x = torch.load(f, map_location=torch.device("cpu"))
        assert isinstance(x, dict), "checkpoint is not a Python dictionary"
        assert "model" in x, "'model' missing from checkpoint"
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ Skipping {f}, not a valid Ultralytics model: {e}")
        return

    updates = {
        "date": datetime.now().isoformat(),
        "version": __version__,
        "license": "AGPL-3.0 License (https://ultralytics.com/license)",
        "docs": "https://docs.ultralytics.com",
    }

    # Update model
    if x.get("ema"):
        x["model"] = x["ema"]  # replace model with EMA
    if hasattr(x["model"], "args"):
        x["model"].args = dict(x["model"].args)  # convert from IterableSimpleNamespace to dict
    if hasattr(x["model"], "criterion"):
       x["model"].criterion = None  # strip loss criterion
    #x["model"].half()  # to FP16
    for p in x["model"].parameters():
        p.requires_grad = False

    # Update other keys
    args = {**DEFAULT_CFG_DICT, **x.get("train_args", {})}  # combine args
    for k in "optimizer", "best_fitness", "ema", "updates":  # keys
        x[k] = None
    x["epoch"] = -1
    x["train_args"] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # strip non-default keys
    # x['model'].args = x['train_args']

    # Save
    torch.save({**updates, **x}, s or f, use_dill=False)  # combine dicts (prefer to the right)
    mb = os.path.getsize(s or f) / 1e6  # file size
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")

    return
    x = torch.load(f, map_location=torch.device('cpu'))
    args = {**DEFAULT_CFG_DICT, **x['train_args']}  # combine model args with default args, preferring model args
    if x.get('ema'):
        x['model'] = x['ema']  # replace model with ema
    for k in 'optimizer', 'ema', 'updates':  # keys
        x[k] = None
    for p in x['model'].parameters():
        p.requires_grad = False
    x['train_args'] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}  # strip non-default keys
    # x['model'].args = x['train_args']
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    LOGGER.info(f"Optimizer stripped from {f},{f' saved as {s},' if s else ''} {mb:.1f}MB")


def train_v2(self, trainer=None, pruning=False, **kwargs):
    """
    Disabled loading new model when pruning flag is set. originated from ultralytics/engine/model.py
    """
    self._check_is_pytorch_model()
    if hasattr(self.session, "model") and self.session.model.id:  # Ultralytics HUB session with loaded model
        if any(kwargs):
            LOGGER.warning("WARNING ⚠️ using HUB training arguments, ignoring local training arguments.")
        kwargs = self.session.train_args  # overwrite kwargs

    checks.check_pip_update_available()

    overrides = yaml_load(checks.check_yaml(kwargs["cfg"])) if kwargs.get("cfg") else self.overrides
    custom = {
        # NOTE: handle the case when 'cfg' includes 'data'.
        "data": overrides.get("data") or DEFAULT_CFG_DICT["data"] or TASK2DATA[self.task],
        "model": self.overrides["model"],
        "task": self.task,
    }  # method defaults
    args = {**overrides, **custom, **kwargs, "mode": "train"}  # highest priority args on the right
    if args.get("resume"):
        args["resume"] = self.ckpt_path

    self.trainer = (trainer or self._smart_load("trainer"))(overrides=args, _callbacks=self.callbacks)
    
    if not pruning:
        if not args.get("resume"):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model
    else:
        # pruning mode
        self.trainer.pruning = True
        self.trainer.model = self.model

        # replace some functions to disable half precision saving
        self.trainer.save_model = save_model_v2.__get__(self.trainer)
        self.trainer.final_eval = final_eval_v2.__get__(self.trainer)

    self.trainer.hub_session = self.session  # attach optional HUB session
    self.trainer.train()
    # Update model and cfg after training
    if RANK in {-1, 0}:
        ckpt = self.trainer.best if self.trainer.best.exists() else self.trainer.last
        self.model, _ = attempt_load_one_weight(ckpt)
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, "metrics", None)  # TODO: no metrics returned by DDP
    return self.metrics

    self._check_is_pytorch_model()
    if self.session:  # Ultralytics HUB session
        if any(kwargs):
            LOGGER.warning('WARNING ⚠️ using HUB training arguments, ignoring local training arguments.')
        kwargs = self.session.train_args
    overrides = self.overrides.copy()
    overrides.update(kwargs)
    if kwargs.get('cfg'):
        LOGGER.info(f"cfg file passed. Overriding default params with {kwargs['cfg']}.")
        overrides = yaml_load(check_yaml(kwargs['cfg']))
    overrides['mode'] = 'train'
    if not overrides.get('data'):
        raise AttributeError("Dataset required but missing, i.e. pass 'data=coco128.yaml'")
    if overrides.get('resume'):
        overrides['resume'] = self.ckpt_path

    self.task = overrides.get('task') or self.task
    self.trainer = TASK_MAP[self.task][1](overrides=overrides, _callbacks=self.callbacks)

    if not pruning:
        if not overrides.get('resume'):  # manually set model only if not resuming
            self.trainer.model = self.trainer.get_model(weights=self.model if self.ckpt else None, cfg=self.model.yaml)
            self.model = self.trainer.model

    else:
        # pruning mode
        self.trainer.pruning = True
        self.trainer.model = self.model

        # replace some functions to disable half precision saving
        self.trainer.save_model = save_model_v2.__get__(self.trainer)
        self.trainer.final_eval = final_eval_v2.__get__(self.trainer)

    self.trainer.hub_session = self.session  # attach optional HUB session
    self.trainer.train()
    # Update model and cfg after training
    if RANK in (-1, 0):
        self.model, _ = attempt_load_one_weight(str(self.trainer.best))
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, 'metrics', None)

def setup_pruner(args):
    #args.sparsity_learning = False
    if args.prune_method == "random":
        imp = tp.importance.RandomImportance()
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.prune_method == "l1":
        imp = tp.importance.MagnitudeImportance(p=1)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.prune_method == "l2":
        imp = tp.importance.MagnitudeImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.prune_method == "fpgm":
        imp = tp.importance.FPGMImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.prune_method == "obdc":
        imp = tp.importance.OBDCImportance(group_reduction='mean', num_classes=args.num_classes)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.prune_method == "lamp":
        imp = tp.importance.LAMPImportance(p=2)
        pruner_entry = partial(tp.pruner.MagnitudePruner, global_pruning=args.global_pruning)
    elif args.prune_method == "slim":
        args.sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning)
    elif args.prune_method == "group_slim":
        args.sparsity_learning = True
        imp = tp.importance.BNScaleImportance()
        pruner_entry = partial(tp.pruner.BNScalePruner, reg=args.reg, global_pruning=args.global_pruning, group_lasso=True)
    elif args.prune_method == "group_norm":
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GroupNormPruner, global_pruning=args.global_pruning)
    elif args.prune_method == "group_sl":
        args.sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2, normalizer='max') # normalized by the maximum score for CIFAR
        pruner_entry = partial(tp.pruner.GroupNormPruner, reg=args.reg, global_pruning=args.global_pruning)
    elif args.prune_method == "growing_reg":
        args.sparsity_learning = True
        imp = tp.importance.GroupNormImportance(p=2)
        pruner_entry = partial(tp.pruner.GrowingRegPruner, reg=args.reg, delta_reg=args.delta_reg, global_pruning=args.global_pruning)
    else:
        raise NotImplementedError
    pruner_entry = partial(pruner_entry, importance=imp)
    #args.is_accum_importance = is_accum_importance
    # unwrapped_parameters = []
    # ignored_layers = []
    # pruning_ratio_dict = {}
    # # ignore output layers
    # for m in model.modules():
    #     if isinstance(m, torch.nn.Linear) and m.out_features == args.num_classes:
    #         ignored_layers.append(m)
    #     elif isinstance(m, torch.nn.modules.conv._ConvNd) and m.out_channels == args.num_classes:
    #         ignored_layers.append(m)
    
    # Here we fix iterative_steps=200 to prune the model progressively with small steps 
    # until the required speed up is achieved.
    # pruner = pruner_entry(
    #     model,
    #     example_inputs,
    #     importance=imp,
    #     iterative_steps=args.iterative_steps,
    #     # pruning_ratio=1.0,
    #     # pruning_ratio_dict=pruning_ratio_dict,
    #     max_pruning_ratio=args.max_pruning_ratio,
    #     # ignored_layers=ignored_layers,
    #     # unwrapped_parameters=unwrapped_parameters,
    # )
    return pruner_entry

## Does not work
# def fixed_step_scheduler(pruning_ratio_dict, steps):
#     # prune same ratio of filter based on initial size
#     pruning_ratio = 1 - math.pow((1 - args.target_prune_rate), 1 / args.iterative_steps)
#     return [pruning_ratio * pruning_ratio_dict for i in range(steps)]

def train(args, plotter: Plotter):
    # load trained YOLOv8 model
    model = YOLO(args.model)

    # Append tweaked training function to model
    model.__setattr__("train_v2", train_v2.__get__(model))

    # Load Config
    pruning_cfg = yaml_load(check_yaml(args.cfg_file))
    pruning_cfg['project'] = "runs/" + pruning_cfg["task"] + "/" + time.strftime("%Y-%m-%d-%H:%M") # Save each run in separate folder
    batch_size = pruning_cfg['batch']

    if args.data is not None: # Overwrite choice from config file, if script argument is provided
        pruning_cfg['data'] = args.data # "coco128.yaml"
    pruning_cfg['epochs'] = args.epochs

    # Save configuration to log
    plotter.append_dict_to_log(pruning_cfg)

    model.model.train()
    for name, param in model.model.named_parameters():
        param.requires_grad = True

    example_inputs = torch.randn(1, 3, pruning_cfg["imgsz"], pruning_cfg["imgsz"]).to(model.device)
    flops_list, num_params_list, map_list, pruned_map_list = [], [], [], [] # Will store metrics during iterative pruning process
    base_flops, base_num_params = tp.utils.count_ops_and_params(model.model, example_inputs) # Baseline metrics

    # do validation before pruning model
    pruning_cfg['name'] = f"baseline_val"
    pruning_cfg['batch'] = 1
    validation_model = deepcopy(model)

    #pruning_cfg['data'] = "coco.yaml" # TODO temporal, remove this
    metric = validation_model.val(**pruning_cfg)
    #pruning_cfg['data'] = "coco128.yaml"

    init_map = metric.box.map
    flops_list.append(base_flops)
    num_params_list.append(100) # save as % of baseline
    map_list.append(init_map)
    pruned_map_list.append(init_map)
    LOGGER.info(f"Before Train: mAP={init_map: .5f}")

    model.model.train()
    for name, param in model.model.named_parameters():
        param.requires_grad = True

    # fine-tuning
    for _, param in model.model.named_parameters():
        param.requires_grad = True
    pruning_cfg['name'] = f"train"
    pruning_cfg['batch'] = batch_size  # restore batch size

    #pruning_cfg['data'] = "coco128.yaml" # TODO Reduced dataset just for training (Temporal)
    model.train_v2(pruning=True, **pruning_cfg)

    # post fine-tuning validation
    pruning_cfg['name'] = f"train_post_val"
    pruning_cfg['batch'] = 1
    validation_model = YOLO(model.trainer.best)
    metric = validation_model.val(**pruning_cfg)
    current_map = metric.box.map
    LOGGER.info(f"After fine tuning mAP={current_map}")

    # Save post fine-tuning validation metrics
    flops_list.append(base_flops)
    num_params_list.append(100) # save as % of baseline
    pruned_map_list.append(init_map)
    map_list.append(current_map)

    # Plot results for each iteration
    plotter.save_pruning_performance_graph(
        num_params_list, 
        map_list, flops_list, 
        pruned_map_list,
        subTitleStr=f"{pruning_cfg['project']} : {args.model} - steps: {args.iterative_steps} - target: {args.target_prune_rate}"
    )

    exported_path = model.export(format='onnx')
    LOGGER.info(f"Final model saved at: {exported_path}")

def inspect_attributes_and_methods(obj: object):
    # Get all attributes and methods
    all_attributes_and_methods = dir(obj)

    # Filter out built-in attributes and methods
    user_defined_attributes_and_methods = [attr for attr in all_attributes_and_methods if not attr.startswith('__')]

    # Print user-defined attributes and methods
    print(user_defined_attributes_and_methods)

def overwrite_dict(target_dict: dict, source_dict: dict):
    LOGGER.info("Over-writing config file parameters with provided cli arguments...")
    # print(f"SRC BEFORE: {source_dict}")
    # print(f"DST BEFORE: {target_dict}")
    for key, value in source_dict.items():
        if key in target_dict and value is not None:
            print(f"Overwrote: ({key}: {value})")
            target_dict[key] = value

def progressive_pruning(
        pruner, 
        model, 
        target_prune_rate, 
        example_inputs):
    
    model.eval()
    base_flops, base_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
    current_prune_rate = 0.0
    current_compression_rate = 1.0
    while current_prune_rate < target_prune_rate:
        pruner.step()
        pruned_flops, pruned_params = tp.utils.count_ops_and_params(model, example_inputs=example_inputs)
        current_compression_rate = float(base_params) / pruned_params
        current_prune_rate = ( float(base_params) - pruned_params ) / float(base_params)
        LOGGER.info(f"Progressive Pruning. Current prune rate: {current_prune_rate} \t - current compression: {current_compression_rate}")
        
        if pruner.current_step == pruner.iterative_steps:
            LOGGER.warning("WARNING⚠️: Reached max iterative step before desired compression was achieved.")
            break
    return current_compression_rate

# def my_linear_scheduler(pruning_ratio_dict, steps):
#     print(f"Dict: {pruning_ratio_dict} - Steps: {steps}")
#     schedule_list = [((1) / float(steps)) * pruning_ratio_dict for i in range(steps+1)]
#     #schedule_list[0] = pruning_ratio_dict * 0.3
#     schedule_list[1] = pruning_ratio_dict * 0.3
#     schedule_list[2] = pruning_ratio_dict * 0.1
#     print(f"Schedule Steps List: {schedule_list}")
#     return schedule_list

# Handles regularization callback creation. These callbacks will then be passed to ultralytics
# trainer, so they are called internally when required
class RegularizationCallbacks:
    def __init__(self, pruner_obj: tp.pruner.MetaPruner):
        self.pruner = pruner_obj

    def on_update_regularizer(self, trainer: BaseTrainer):
        LOGGER.debug("UPDATE REG")
        self.pruner.update_regularizor() ## TODO TYPO IN FUNCTION CALL IN CURRENT VERSION OF LIBRARY

    def on_regularize(self, trainer: BaseTrainer):
        LOGGER.debug("ON REG")
        self.pruner.regularize(trainer.model)


def prune(args, plotter: Plotter):

    # Check which arguments were passed by the user (non default), that are relevant or will overwrite config file parameters
    keys_to_extract = ['cfg_file', 'model', 'data', 'epochs', 'lrf', 'prune_method', 'iterative_steps', 'target_prune_rate', 'sparsity_learning']
    filtered_dict = {k: vars(args)[k] for k in keys_to_extract if k in vars(args) and vars(args)[k] is not None}
    plotter.append_dict_to_log(dict = filtered_dict, description="Main arguments:")

    # Save script arguments to log
    plotter.append_dict_to_log(dict = vars(args), description="All script arguments:") # Convert to dict with "vars"

    # Load Config
    pruning_cfg = yaml_load(check_yaml(args.cfg_file))

    # Overwrite config file arguments with those parsed from command line
    overwrite_dict(pruning_cfg, vars(args))

    # Save path for results
    pruning_cfg['project'] = os.path.join("runs", pruning_cfg["task"], time.strftime("%Y-%m-%d-%H-%M")) # Save each run in separate folder
    # pruning_cfg['project'] = os.path.abspath(pruning_cfg['project'])
    batch_size = pruning_cfg['batch'] # Save original batch size

    # load trained YOLOv8 model
    model = YOLO(pruning_cfg["model"])
    # inspect_attributes_and_methods(model)

    # Append tweaked training function to model
    model.__setattr__("train_v2", train_v2.__get__(model))

    # use coco128 dataset for 10 epochs fine-tuning each pruning iteration step
    # this part is only for sample code, number of epochs should be included in config file
    # if args.data is not None: # Overwrite choice from config file, if script argument is provided
    #     pruning_cfg['data'] = args.data # "coco128.yaml"
    # pruning_cfg['epochs'] = args.epochs
    # TODO LR?

    # Save configuration to log
    plotter.append_dict_to_log(pruning_cfg, description="Final Configuration:")

    model.model.train()
    replace_c2f_with_c2f_v2(model.model) # Prevents depGraph error (caused by layer split and concatenation)
    initialize_weights(model.model)  # set BN.eps, momentum, ReLU.inplace

    for name, param in model.model.named_parameters():
        param.requires_grad = True

    example_inputs = torch.randn(1, 3, pruning_cfg["imgsz"], pruning_cfg["imgsz"]).to(model.device)
    flops_list, num_params_list, map_list, pruned_map_list = [], [], [], [] # Will store metrics during iterative pruning process
    base_flops, base_num_params = tp.utils.count_ops_and_params(model.model, example_inputs) # Baseline metrics

    # do validation before pruning model
    pruning_cfg['name'] = f"baseline_val"
    pruning_cfg['batch'] = 1
    validation_model = deepcopy(model)

    #pruning_cfg['data'] = "coco.yaml" # TODO temporal, remove this
    metric = validation_model.val(**pruning_cfg)
    #pruning_cfg['data'] = "coco128.yaml"

    init_map = metric.box.map
    flops_list.append(base_flops)
    num_params_list.append(100) # save as % of baseline
    map_list.append(init_map)
    pruned_map_list.append(init_map)
    LOGGER.info(f"Before Pruning: FLOPs={base_flops / 1e9: .5f} G, #Params={base_num_params / 1e6: .5f} M, mAP={init_map: .5f}")

    # prune same ratio of filter based on initial size
    pruning_ratio = 1 - math.pow((1 - args.target_prune_rate), 1 / args.iterative_steps) if args.iterative_steps > 0 else 1
    LOGGER.info(f"PRUNE RATIO: {pruning_ratio}")
    fine_steps = round(100.0 / pruning_ratio) if pruning_ratio != 0 else 1 # Steps taken inside torch-pruning to approach desired prune ratio at give global step
    LOGGER.info(f"SMOOTH STEPS: {fine_steps}")
    #pruning_ratio = args.target_prune_rate # FOR TESTING: should cut model by half twice if iterative_steps is 2
    
    #for i in range(args.iterative_steps):
    i = 0
    pretraining_done_flag = False # Indicates pretraining on first iteration is done
    while i < args.iterative_steps:
        model.model.train()
        for name, param in model.model.named_parameters():
            param.requires_grad = True

        ignored_layers = []
        unwrapped_parameters = []
        pruning_ratio_dict = {}
        for m in model.model.modules():
            if isinstance(m, (Detect,)):
                ignored_layers.append(m)

        # Setup pruner on each iteration 
        pruner = setup_pruner(args)

        # Disable weight decay if sparsity learning is used (sparsity learning already involves a form of weight decay regularization)
        if args.sparsity_learning:
            pruning_cfg['weight_decay'] = 0.0

        example_inputs = example_inputs.to(model.device)
        # pruner = tp.pruner.GroupNormPruner(
        pruner = pruner(
            model.model,
            example_inputs,
            #importance=tp.importance.GroupNormImportance(),  # L2 norm pruning,
            #global_pruning
            # pruning_ratio=pruning_ratio,
            pruning_ratio=1.0,
            #pruning_ratio_dict=pruning_ratio_dict,
            #max_pruning_ratio=args.max_pruning_ratio,
            iterative_steps=fine_steps, # was 400 before
            #iterative_pruning_ratio_scheduler=my_linear_scheduler, # linear is default
            ignored_layers=ignored_layers,
            unwrapped_parameters=unwrapped_parameters,
        )

        # TODO Regularization
        # Test regularization
        #output = model.model(example_inputs)
        #(output[0].sum() + sum([o.sum() for o in output[1]])).backward()
        #pruner.regularize(model.model)

        # Configure Sparsity Learning
        if(args.sparsity_learning):
            # Add regularization callbacks to model (will be automatically called when training)
            regularization_callbacks = RegularizationCallbacks(pruner)
            model.reset_callbacks() # Clear callbacks from previous iterations
            model.add_callback("on_after_model_train_mode", regularization_callbacks.on_update_regularizer)
            model.add_callback("on_before_optimizer_step", regularization_callbacks.on_regularize)

        # Prune if sparsity learning is not activated, or if first iteration (just pre-train) already passed 
        if(not args.sparsity_learning or pretraining_done_flag):
            i += 1 # Increment Iteration
            LOGGER.info(f"Started pruning for iter {i}")
            #pruner.step()
            progressive_pruning(pruner=pruner, 
                                model=model.model, 
                                target_prune_rate=pruning_ratio, #args.target_prune_rate, 
                                example_inputs=example_inputs)

        # pre fine-tuning validation
        pruning_cfg['name'] = f"step_{i}_pre_val"
        pruning_cfg['batch'] = 1
        validation_model.model = deepcopy(model.model)
        metric = validation_model.val(**pruning_cfg)
        pruned_map = metric.box.map
        pruned_flops, pruned_num_params = tp.utils.count_ops_and_params(pruner.model, example_inputs.to(model.device))
        current_speed_up = float(flops_list[0]) / pruned_flops

        if(not args.sparsity_learning or pretraining_done_flag):
            LOGGER.info(f"After pruning iter {i + 1}: FLOPs={pruned_flops / 1e9} G, #Params={pruned_num_params / 1e6} M, "
                f"mAP={pruned_map}, speed up={current_speed_up}")
        else:
            LOGGER.info(f"After Pre-train iter {i}: mAP={pruned_map}")
            
        # post-training (or pre-training if i == 0)
        for _, param in model.model.named_parameters():
            param.requires_grad = True
        pruning_cfg['name'] = f"step_{i}_finetune"
        pruning_cfg['batch'] = batch_size  # restore batch size

        #pruning_cfg['data'] = "coco128.yaml" # TODO Reduced dataset just for training (Temporal)
        model.train_v2(pruning=True, **pruning_cfg)
        #pruning_cfg['data'] = "coco.yaml"

        # Signal pre-training on first iteration (for sparsity learning) is done
        if(args.sparsity_learning and i == 0):
            pretraining_done_flag = True

        #print(model.model.criterion)
        #LOGGER.error("ERROR: ", str(model.model.criterion))

        # post fine-tuning validation
        pruning_cfg['name'] = f"step_{i}_post_val"
        pruning_cfg['batch'] = 1
        validation_model = YOLO(model.trainer.best)
        metric = validation_model.val(**pruning_cfg)
        current_map = metric.box.map
        LOGGER.info(f"After fine tuning mAP={current_map}")

        # Save post fine-tuning validation metrics
        flops_list.append(pruned_flops)
        num_params_list.append(pruned_num_params / base_num_params * 100)
        pruned_map_list.append(pruned_map)
        map_list.append(current_map)

        # remove pruner after single iteration
        del pruner

        # Plot results for each iteration
        if not args.test_run:
            plotter.save_pruning_performance_graph(
                num_params_list, 
                map_list, flops_list, 
                pruned_map_list,
                subTitleStr=f"{pruning_cfg['project']} : {pruning_cfg['model']} - steps: {args.iterative_steps} - target: {args.target_prune_rate} - epochs: {pruning_cfg['epochs']}"
            )

        if init_map - current_map > args.max_map_drop:
            LOGGER.info("Pruning early stop: Reached max mAP drop")
            break

        if ( 1.0 / current_speed_up ) < ( 1 - args.target_prune_rate ):
            LOGGER.info("Pruning early stop: Reached target speedup")
            break

    exported_path = model.export(format='onnx')
    LOGGER.info(f"Final model saved at: {exported_path}")

def float_range(mini,maxi):
    """Return function handle of an argument type function for 
       ArgumentParser checking a float range: mini <= arg <= maxi
         mini - minimum acceptable argument
         maxi - maximum acceptable argument"""

    # Define the function with default arguments
    def float_range_checker(arg):
        """New Type function for argparse - a float within predefined range."""

        try:
            f = float(arg)
        except ValueError:    
            raise argparse.ArgumentTypeError("must be a floating point number")
        if f < mini or f > maxi:
            raise argparse.ArgumentTypeError("must be in range [" + str(mini) + " .. " + str(maxi)+"]")
        return f

    # Return function handle to checking function
    return float_range_checker

def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 Pruning")

    # Default parameters from Ultralytics Yolo config file
    parser.add_argument('--cfg-file', default='my-config-files/base_cfg.yaml',
                        help='Pruning config file.'
                             ' This file should have same format with ultralytics/yolo/cfg/default.yaml')
    
    # Arguments for Ultralytics YOLO Library Configuration
    # Will overwrite config file parameters if not None
    parser.add_argument('--model', type=str, default=None, 
                        help='Pretrained pruning target model file')
    parser.add_argument('--data', type=str, default=None,
                        #choices=['coco128.yaml', 'coco8.yaml', 'coco.yaml'],
                        help='Set the desired dataset')
    parser.add_argument("--device", type=str, default='cuda',
                        choices=['cpu', 'cuda'],
                        help='Set the desired target device')
    parser.add_argument('--epochs', default=None, type=int, help='Training epochs')
    parser.add_argument("--lrf", type=float, default=None, # Final Learning Rate Multiplier
                        help="Final Learning Rate MULTIPLIER"
                        "Final learning rate = lr0 * lrf"
                        "For constant learning rate: lrf = 1"
                        "lr0 is initial learning rate (default is 0.01 for SGD)") 
    parser.add_argument('--weight-decay', default=None, type=float, 
                        help='Optimizer Weight Decay for L2 regularization during training.'
                        'Default in config_file is 1e-5')
    parser.add_argument('--batch', default=None, type=int, 
                        help='Training Batch Size.'
                        'Default in config_file is 32')
    parser.add_argument('--workers', default=None, type=int, 
                        help='Number of worker threads for data loading.'
                        'Default in config_file is 4')
    
    # Arguments for Pruning Script
    parser.add_argument('--script-mode', type=str, default='prune',
                        choices=['prune', 'train'],
                        help='Set the script mode')
    parser.add_argument("--prune-method", type=str, default='group_sl',
                        choices=['group_norm', 'group_sl'],
                        help="Pruning method")
    parser.add_argument("--sparsity-learning", action="store_true", default=False,
                        help='Apply sparsity regularization on pre-training and post-training steps'
                        'Will be set to true automatically if group_sl method is selected.')
    parser.add_argument("--reg", type=float, default=1e-5, # Was 5e-4 before
                        help='Regularization Coefficient for Sparsity Learning') 
    parser.add_argument("--global-pruning", action="store_true", default=False)
    parser.add_argument('--iterative-steps', default=1, type=int, 
                        help='Total pruning iteration step')
    parser.add_argument('--target-prune-rate', default=0.5, type=float, 
                        help='Target pruning rate (proportion of removed parameters i.e. 1 - final_params / initial_params)')
    parser.add_argument('--max-map-drop', default=1.0, type=float_range(0, 1), 
                        help='Allowed maximum map drop after fine-tuning (absolute percentage drop in the range [0,1])')


    parser.add_argument('--log-level', type=str, default='INFO', # DEBUG LEVEL DOESN'T WORK
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    parser.add_argument("--test-run", action="store_true", default=False,
                        help="Indicates this is a test run, so there's no need to save pruning results")
    
    return parser.parse_args()

def set_logger_level(log_level):
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        LOGGER.error(f'Invalid log level: {log_level}')
        LOGGER.setLevel(logging.INFO)
        LOGGER.info(f'Logger level set to {log_level}')
    else:
        LOGGER.info(f'Logger level set to {log_level} {(numeric_level)}')
        LOGGER.setLevel(numeric_level)

    # Test each level
    # LOGGER.debug("Logger Debug Message Test")
    # LOGGER.info("Logger Info Message Test")
    # LOGGER.warning("Logger Warning Message Test")
    # LOGGER.error("Logger Error Message Test")
    # LOGGER.critical("Logger Critical Message Test")

if __name__ == "__main__":
    
    args = parse_args()
    set_logger_level(args.log_level) # Debug level doesn't work

    # Instantiate plotter for pruning results
    plotter = Plotter()
    
    # Save start time
    start_time = time.time()
    
    if args.script_mode == "prune":
        # Run pruning algorithm
        prune(args, plotter)
    elif args.script_mode == "train":
        # Just run training
        train(args, plotter)

    # Print runtime
    runtime = str(timedelta( seconds=(time.time() - start_time) ))
    LOGGER.info(f"--- Total runtime: {runtime} (hours:min:sec) ---")
    plotter.append_dict_to_log({"runtime": runtime}, description="Runtime:")
