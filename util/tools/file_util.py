import argparse
import os
import time
import warnings
from typing import Dict

import yaml

from util.log.logger import GlobalLogger


def read_config(config_path: str) -> Dict:
    """
    Read config.yaml to program
    :param config_path: str, the path to the config files
    :return: dict, the value of config
    """
    fin = open(config_path, encoding="utf-8")
    data = yaml.load(fin, Loader=yaml.FullLoader)
    return data


def create_dir(*target: str) -> bool:
    """
    Create the dir and return whether the dir is created.
    :param target: str, the path to dir.
    :return: bool, whether the dir is successfully created.
    """
    # if the directory not exist then create the directory
    if os.path.exists(os.path.join(*target)):
        return True
    else:
        try:
            tmp = 1
            while tmp <= len(target):
                if not os.path.exists(os.path.join(*target[:tmp])):
                    os.mkdir(os.path.join(*target[:tmp]))
                tmp += 1
            return True
        except IOError:
            return False


def set_ignore_warning(close: bool) -> None:
    """
    Close the UserWarning by Python
    :param close: bool, whether to close the UserWarning
    :return: None
    """
    if close:
        warnings.filterwarnings("ignore")


def parse_opt():
    parser = argparse.ArgumentParser(description='Options')
    parser.add_argument('--log_name', type=str, help='The name of the log')
    parser.add_argument('--data_train', type=str, help='The path to input directory')
    parser.add_argument('--data_eval', type=str, help='The path to evaluation directory')
    parser.add_argument('--output_checkpoint', type=str, help='The path to evaluation directory', default="checkpoint")
    return parser.parse_args()


def clear_error():
    pass


def global_init():
    opt = vars(parse_opt())
    config = read_config(os.path.join("configs", "config.yaml"))
    config["runtime"] = opt
    run_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    if 'warning' in config.keys():
        set_ignore_warning(config['warning']['ignore'])
    create_dir(config['log']['log_dir'], config["runtime"]["log_name"], run_time)
    GlobalLogger().init_config(config=config['log'], store_name=os.path.join(config["runtime"]["log_name"], run_time))
    GlobalLogger().get_logger().info("Running with identifier {}".format(run_time))
    GlobalLogger().get_logger().info("Saving log to {}".format(
        os.path.join(config['log']['log_dir'], config["runtime"]["log_name"], run_time, "run.log")))
    create_dir(config['runtime']['output_checkpoint'], config["runtime"]["log_name"], run_time)
    return config, run_time
