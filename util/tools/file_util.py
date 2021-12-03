import argparse
import os
import shutil
import warnings
from typing import Dict

import yaml

from util.logger.logger import GlobalLogger

# DEPRECATED
from util.logger.tensorboards import GlobalTensorboard


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


def remove_dir(*target: str) -> None:
    """
    Delete all files in the directory and then delete the directory
    :param target: str, the path to the directory to remove, if length is 1, target will be recognized as the FULL PATH,
    otherwise will be recognized as the path tuple
    :return: None
    """
    try:
        if not os.path.exists(os.path.join(*target)):
            return
        else:
            files = os.path.join(*target)
            if os.path.islink(files):
                os.remove(files)
            elif os.path.isfile(files):
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
    except IOError as e:
        print("Unable to delete {} with error {}".format(os.path.join(*target), e))


def rename_file(target: str, *source: str) -> None:
    """
    Rename files to the target name
    :param target: str, the name of the target, NOT THE FULL PATH
    :param source: tuple of str, when length is 1, source will be recognized as the FULL PATH, otherwise will be the str
    tuple, which iteratively lead to the target path.
    """
    try:
        if len(source) > 1:
            shutil.move(os.path.join(*source), os.path.join(*source[:-1], target))
        else:
            shutil.move(*source, target)
    except IOError as e:
        print("Unable to rename {} to {} with error {}".format(os.path.join(*source),
                                                               os.path.join(*source[:-1],
                                                                            target) if len(
                                                                   source) > 1 else target,
                                                               e))


def delete_if_empty(*target: str) -> None:
    """
    Remove directory if is empty
    :param target: tuple of str, when length is 1, source will be recognized as the FULL PATH, otherwise will be the str
    tuple, which iteratively lead to the target path.
    :return: None
    """
    try:
        if not os.path.exists(os.path.join(*target)):
            return
        else:
            files = os.path.join(*target)
            if os.path.isdir(files) and len(os.listdir(files)) == 0:
                shutil.rmtree(files)
            else:
                return
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
    except IOError as e:
        print("Unable to delete {} with error {}".format(os.path.join(*target), e))


def on_train_finish(opt: argparse.Namespace, runtime: str) -> None:
    """
    When the procedures finish, the function will run.
    :param opt: argparse.Namespace, the runtime config
    :param runtime: str, the global identifier
    :return: None
    """
    # Copy the train config to the log directory
    shutil.copy2(os.path.join("configs", "train_config.py"), os.path.join(opt.log_dir, opt.log_name, runtime))
    # Delete the output if empty
    delete_if_empty(opt.output_dir, opt.log_name, runtime)
    # Delete the tensorboard files if empty
    delete_if_empty(opt.tf_dir, opt.log_name, runtime)


def on_train_error(opt: argparse.Namespace, runtime: str, error_name: str) -> None:
    """
    When the procedures end with interrupt or error, the function will run.
    :param error_name: str, the name of the error
    :param opt: argparse.Namespace, the runtime config
    :param runtime: str, the global identifier
    :return: None
    """
    # Remove all files in checkpoints, mainly weight files
    remove_dir(opt.output_checkpoint_dir, opt.log_name, runtime)
    # Close logging module
    GlobalLogger().close()
    GlobalTensorboard().close()
    # Remove the log output
    rename_file(runtime + "_" + error_name, opt.log_dir, opt.log_name, runtime)
    # Remove the tensorboard output
    rename_file(runtime + "_" + error_name, opt.tf_dir, opt.log_name, runtime)
    # Delete the tensorboard files if empty
    delete_if_empty(opt.tf_dir, opt.log_name, runtime + "_" + error_name)
    # Delete the output if empty
    delete_if_empty(opt.output_dir, opt.log_name, runtime)
