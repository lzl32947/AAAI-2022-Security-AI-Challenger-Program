import threading
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class GlobalTensorboard(object):
    """
    This is the class of the global tensorboard and is set to be the single-instance when running
    """
    # Add lock in the instance for threading in case
    _instance_lock = threading.Lock()

    def __init__(self) -> None:
        # Check the attr in __init__ for not initializing the parameters in re-creating the instance
        if not hasattr(self, "writer"):
            self.writer = None
        if not hasattr(self, "tf_path"):
            self.tf_path = None
        if not hasattr(self, "log_name"):
            self.log_name = None

    def __new__(cls, *args, **kwargs):
        # This function is used to lock the instance for only create the single instance when running
        if not hasattr(GlobalTensorboard, "_instance"):
            with GlobalTensorboard._instance_lock:
                if not hasattr(GlobalTensorboard, "_instance"):
                    GlobalTensorboard._instance = object.__new__(cls)
        return GlobalTensorboard._instance

    def get_writer(self) -> SummaryWriter:
        """
        Get the global tensorboard writer.
        :return: SummaryWriter instance, and in class return the instance of tensorboard
        """
        return self.writer

    def init_config(self, tf_path: str, store_name: str) -> None:
        """
        Init the logger with the given parameters.
        :param tf_path: str, the path to tensorboard
        :param store_name: str, the name of the saving name
        :return: None
        """
        self.tf_path = tf_path
        self.log_name = store_name
        self.writer = SummaryWriter(os.path.join(tf_path, store_name))

    def close(self):
        if self.writer is not None:
            self.writer.close()
