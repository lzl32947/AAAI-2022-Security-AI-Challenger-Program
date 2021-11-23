import argparse
import os.path
import time

from bases.train import train
from util.logger.logger import GlobalLogger
from util.tools.file_util import global_init, clear_error

if __name__ == '__main__':
    config, identifier = global_init()
    try:
        train(dataset_path=config['runtime']['data_train'],
              checkpoint_path=os.path.join(config['runtime']['output_checkpoint'], config["runtime"]["log_name"],
                                           identifier))
    except Exception as e:
        GlobalLogger().get_logger().error(e)
        clear_error(config, identifier, "fail")
    except KeyboardInterrupt as k:
        GlobalLogger().get_logger().info("Keyboard Interrupt")
        clear_error(config, identifier, "interrupt")
