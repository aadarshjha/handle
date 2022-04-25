import sys
import os
import json
import numpy as np
import torch
import shutil
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split

import warnings
import copy
import yaml

# Custom imports
from network import *
from data_pipeline import *
from parameter_sets import *
from evaluation import *
from tools import *


MANUAL_SEED = 1

warnings.filterwarnings("ignore")


# Config class to load in YAML configurations
class Config:
    def __init__(self, in_dict: dict):
        for key, val in in_dict.items():
            if isinstance(val, (list, tuple)):
                setattr(
                    self, key, [DictObj(x) if isinstance(x, dict) else x for x in val]
                )
            else:
                setattr(self, key, DictObj(val) if isinstance(val, dict) else val)


def save_model(model, save_pth):
    torch.save(model.state_dict(), save_pth)


def run():
    if len(sys.argv) != 2:
        print("Usage: python run.py name-of-config-file.yaml")
        return 0

    # Load config file for model and training parameters
    print("Loading configuration file...")
    config_file = sys.argv[1]
    config_dict = {}
    torch.manual_seed(MANUAL_SEED)

    try:
        with open(config_file, "r") as f:
            user_config = yaml.safe_load(f)
            config_dict.update(user_config)
    except Exception:
        print(f"Error: failed to open file {config_file}")
        return -1

    config_dict["device"] = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    config = Config(config_dict)

    print("Setting configuration...")
    device = config.device

    if config.model == "resnext":
        model, parameters = load_resnext101(config, device)
    elif config.model == "cnn-lstm":
        model, parameters = load_cnn_lstm(config, device)
    elif config.model == "timesformer":
        model, parameters = load_timesformer(config, device)
    else:
        print(f"Model {config.model} not implemented")
        return -1

    criterion = nn.CrossEntropyLoss().to(device)

    if config.model == "cnn-lstm":
        train_params = LSTM_Parameters(config)
    else:
        train_params = IPN_Parameters(config)

    print("Loading training data...")
    training_data = get_ipn_data(
        config,
        "training",
        train_params.train_spatial_transform,
        train_params.train_temporal_transform,
    )
    train_loader = DataLoader(
        training_data,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.n_threads,
        pin_memory=True,
    )
    train_log_pth = os.path.join(
        config.result_path, f"train_%s.log" % (config.save_name)
    )
    train_logger = Logger(
        train_log_pth, ["epoch", "loss", "acc", "precision", "recall", "bacc"]
    )

    print("Loading validation data...")
    val_data = get_ipn_data(
        config,
        "validation",
        train_params.val_spatial_transform,
        train_params.val_temporal_transform,
    )

    split_1_ct = int(0.5 * len(val_data))
    split_2_ct = len(val_data) - split_1_ct
    val_data, test_data = random_split(val_data, [split_1_ct, split_2_ct], generator=torch.Generator().manual_seed(42))

    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.n_threads,
        pin_memory=True,
    )
    val_log_pth = os.path.join(config.result_path, f"val_%s.log" % (config.save_name))
    val_logger = Logger(val_log_pth, ["epoch", "loss", "acc", "precision", "recall", "bacc"])

    
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.n_threads,
        pin_memory=True,
    )
    test_log_pth = os.path.join(config.result_path, f"test_%s.log" % (config.save_name))
    test_logger = Logger(test_log_pth, ["loss", "acc", "precision", "recall", "bacc"])

    if train_params.optimizer_type == "SGD":
        print("Using SGD optimizer...")
        optimizer = optim.SGD(
            parameters,
            lr=train_params.lr,
            momentum=train_params.momentum,
            dampening=train_params.dampening,
            weight_decay=train_params.weight_decay,
        )
    else:
        print("Error: Optimizer not implemented")
        return -1

    best_acc = 0

    # return 0

    for i in range(1, config.n_epochs + 1):
        # Run training for this epoch
        train_epoch(i, train_loader, model, device, criterion, optimizer, train_logger)

        # Validate for this epoch
        _, val_acc, _, _, _ = validate_epoch(
            i, val_loader, model, device, criterion, val_logger
        )

        if val_acc > best_acc:
            best_acc = val_acc

            # Save model here
            # Save epoch and metric as well for reference
            print(f"Saving model w/ acc {best_acc}")
            save_pth = os.path.join(config.result_path, f"%s.pth" % (config.save_name))
            save_model(model, save_pth)

        print(f"\nBest model so far: Validation Acc: {best_acc}\n")


    test_model(test_loader, model, device, criterion, test_logger)

    return 0


if __name__ == "__main__":
    code = run()
    print("Exiting {} w/ code {}".format(__file__, code))
