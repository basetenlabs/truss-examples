import logging
import torch
import os
import os.path as osp
import sys
from torch import optim
import numpy as np
import random


def setup_logging(filename: str, level_str: str, filemode: str):
    """
    Setup logging configuration
    :param filename: Log file
    :param level_str:
    :param filemode:
    :return:
    """
    if level_str == "error":
        level = logging.ERROR
    elif level_str == "warning":
        level = logging.WARNING
    elif level_str == "info":
        level = logging.INFO
    else:
        raise ValueError(
            'Unknown logging level {}. Expected one of ("error", "warning", "info")'
        )

    log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    file_handler = logging.FileHandler(filename, mode=filemode)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)


def load_checkpoint(filename: str):
    """
    Load model, optimizer and scheduler state from a file
    :param filename: Input state filename
    :return: Dictionary with 'model', 'optimizer', 'scheduler' and 'args' as keys and their states as values
    """
    return torch.load(filename)


def save_checkpoint(filename, model, args, optimizer, scheduler, current_epoch):

    torch.save(
        {
            "model": model.state_dict(),
            "args": args,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "current_epoch": current_epoch,
        },
        "{}".format(filename),
    )


def get_optimizer_model(name: str, model, lr, **kwargs):
    """
    Create and initialize a PyTorch optimizer
    :param name: Name of the optimizer. One of ('SGD', 'Adam')
    :param model: PyTorch model (torch.nn.Module) whose parameters will be optimized
    :param lr: Learning rate
    :param kwargs: other keyword arguments to the optimizer
    :return: Optimizer
    """
    params = list(model.parameters())
    if name == "SGD":
        if lr is None:
            lr = 0.1
        return optim.SGD(params, lr=lr, **kwargs)
    if name == "Adam":
        if lr is None:
            lr = 0.0001
        return optim.Adam(params, lr=lr, weight_decay=1e-5)
    raise ValueError("Unknown optimizer: " + name)


def create_dir(path: str):
    """
    Create a directory of it does not exist
    :param path: Path to directory
    :return: None
    """
    if not osp.exists(path):
        os.makedirs(path, exist_ok=True)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


def get_device(args):
    gpu_string = "cuda:"
    gpu_array = []
    length = 1
    for i in args.gpu:
        if length == len(args.gpu):
            gpu_string = gpu_string + i
        else:
            gpu_string = gpu_string + i + ","
        gpu_array.append(int(i))
        length = length + 1
    return gpu_string, gpu_array


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(manualSeed):
    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def create_directories_level2(args, exp_name, exp_name_2):
    log_filename = osp.join("dump", exp_name, exp_name_2, "log.txt")
    args.experiment_dir = osp.join("dump", exp_name, exp_name_2)
    args.experiment_dir_base = osp.join("dump", exp_name)
    args.checkpoint_dir = osp.join("dump", exp_name, exp_name_2, "checkpoints")
    args.checkpoint_dir_base = osp.join("dump", exp_name, "checkpoints")
    args.vis_dir = osp.join("dump", exp_name, exp_name_2, "vis_dir") + "/"
    args.generate_dir = osp.join("dump", exp_name, exp_name_2, "generate_dir") + "/"

    create_dir(args.checkpoint_dir)
    create_dir(args.vis_dir)
    create_dir(args.generate_dir)

    if args.train_mode != "test":
        setup_logging(log_filename, args.log_level, "w")
    else:
        test_log_filename = osp.join("dump", exp_name, exp_name_2, "test_log.txt")
        setup_logging(test_log_filename, args.log_level, "w")
        args.query_generate_dir = (
            osp.join("dump", exp_name, exp_name_2, "query_generate_dir") + "/"
        )
        create_dir(args.query_generate_dir)
        args.vis_gen_dir = osp.join("dump", exp_name, exp_name_2, "vis_gen_dir") + "/"
        create_dir(args.vis_gen_dir)

    return args
