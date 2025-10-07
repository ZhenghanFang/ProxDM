import glob
import logging
import os

import torch
from torch import nn


def setup_logging(debug: bool):
    level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger()  # root logger
    logger.setLevel(level)

    # Clear existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter("[%(name)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def has_nan_in_torch_model(model: nn.Module) -> bool:
    for param in model.parameters():
        if param is not None and torch.isnan(param).any():
            return True
    return False


def get_ckpt_num(ckpt_path: str) -> int:
    """
    Extract the checkpoint number from the checkpoint path.
    Example: path/to/ckpt_100000.pt -> 100000
    """
    ckpt_num = os.path.basename(ckpt_path).split("_")[1].split(".")[0]
    return int(ckpt_num)


def scan_ckpt_nums(ckpt_dir):
    """
    Scan folder for ckpt numbers
    Example: ckpt_dir/ckpt_{10, 20, 30}.pt -> [10, 20, 30]
    """
    ckpt_files = glob.glob(os.path.join(ckpt_dir, "ckpt_*.pt"))
    ckpt_nums = [get_ckpt_num(x) for x in ckpt_files]
    return ckpt_nums
