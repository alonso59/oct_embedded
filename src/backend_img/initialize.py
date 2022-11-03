import logging
import datetime
import os
import sys
import numpy as np
import torch
import yaml

def initialize(cfg):
    """Directories"""
    now = datetime.datetime.now()
    version = 'logs/' + str(now.strftime("%Y-%m-%d_%H_%M_%S")) + '/'
    checkpoint_path = version + "checkpoints/"
    create_dir(checkpoint_path)

    with open(version + 'experiment_cfg.yaml', 'w') as yaml_config:
        yaml.dump(cfg, yaml_config)

    """logging"""
    logging.basicConfig(filename=version + "info.log",
                        format='%(asctime)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler(sys.stdout))

    """ Seeding """
    # seeding(43)  # 42
    return logger, checkpoint_path, version

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def seeding(seed):
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True