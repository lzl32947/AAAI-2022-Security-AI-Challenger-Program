import argparse
import os.path
import time

from bases.train import train
from util.logger.logger import GlobalLogger
from util.tools.args_util import parse_opt
from util.tools.file_util import clear_error, read_config, set_ignore_warning, create_dir


def global_init() -> (argparse.Namespace, str):
    # Get the configs from the command line
    opt = parse_opt()
    # Generate the run time as the identifier
    run_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # Whether to ignore warning raised by torch
    if not opt.not_ignore_warning:
        set_ignore_warning(True)
    # Create the log directory, e.g. "log/base/2021123_223310"
    create_dir(opt.log_dir, opt.log_name, run_time)
    GlobalLogger().init_config(log_path=opt.log_dir,
                               store_name=os.path.join(opt.log_name, run_time))
    # Write log
    GlobalLogger().get_logger().info("Running with identifier {}".format(run_time))
    GlobalLogger().get_logger().info("Saving log to {}".format(
        os.path.join(opt.log_dir, opt.log_name, run_time, "run.log")))
    # Create for checkpoint
    create_dir(opt.output_checkpoint_dir, opt.log_name, run_time)
    return opt, run_time


if __name__ == '__main__':
    args, identifier = global_init()
    try:
        train(args, identifier)
    except Exception as e:
        GlobalLogger().get_logger().error(e)
        clear_error(args, identifier, "fail")
    except KeyboardInterrupt as k:
        GlobalLogger().get_logger().info("Keyboard Interrupt")
        clear_error(args, identifier, "interrupt")
