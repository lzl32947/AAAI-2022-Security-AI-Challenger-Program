import os
import shutil
import warnings
from typing import Dict

import yaml

from util.logger.logger import GlobalLogger


# DEPRECATED
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


# NOT TESTED!
def remove_dir(*target):
    try:
        if not os.path.exists(os.path.join(*target)):
            return True
        else:
            files = os.path.join(*target)
            if os.path.islink(files):
                os.remove(files)
            elif os.path.isdir(files):
                os.remove(files)
            elif os.path.isdir(files):
                shutil.rmtree(files)
            else:
                GlobalLogger().get_logger().error("Unable to delete {}".format(os.path.join(*target)))
            if len(target) != 1:
                i = len(files) - 2
                while i >= 0:
                    path = os.path.join(*target[:i])
                    if os.path.exists(path):
                        try:
                            os.remove(path)
                            i -= 1
                        except IOError:
                            break
                    else:
                        break
    except IOError:
        GlobalLogger().get_logger().error("Unable to delete {}".format(os.path.join(*target)))


# NOT TESTED
def rename_file(target, *source):
    try:
        if len(source) > 1:
            os.rename(os.path.join(*source), os.path.join(*source[:-1], target))
        else:
            os.rename(*source, target)
    except IOError:
        GlobalLogger().get_logger().error("Unable to rename {} to {}".format(os.path.join(*source),
                                                                             os.path.join(*source[:-1], target) if len(
                                                                                 source) > 1 else target))


# NOT TESTED
def clear_error(opt, run_time, error_name):
    remove_dir(opt.output_checkpoint_dir, opt.log_name, run_time)
    rename_file(run_time + "_" + error_name, opt.log_dir, opt.log_name, run_time)
