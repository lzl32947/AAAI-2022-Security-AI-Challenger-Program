import argparse
import os.path
import time

from bases.train import train
from util.logger.logger import GlobalLogger
from util.tools.file_util import global_init

if __name__ == '__main__':
    config, identifier = global_init()
    train(dataset_path=config['runtime']['data_train'],
          checkpoint_path=os.path.join(config['runtime']['output_checkpoint'], config["runtime"]["log_name"],
                                       identifier))
