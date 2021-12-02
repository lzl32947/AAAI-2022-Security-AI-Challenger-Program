import argparse
import os.path
import time

from bases.train import train
from util.logger.logger import GlobalLogger
from util.logger.tensorboards import GlobalTensorboard
from util.tools.args_util import parse_train_opt
from util.tools.file_util import on_error, read_config, set_ignore_warning, create_dir, on_finish


def global_init() -> (argparse.Namespace, str):
    # Get the configs from the command line
    opt = parse_train_opt()
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
    GlobalLogger().get_logger().info("Using args: {}".format([(i, vars(opt)[i]) for i in vars(opt).keys()]))
    # Create for checkpoint
    create_dir(opt.output_checkpoint_dir, opt.log_name, run_time)
    # Create fot tensorboard
    create_dir(opt.tf_dir, opt.log_name, run_time)
    # Create for output
    create_dir(opt.output_dir, opt.log_name, run_time)
    if opt.enable_tensorboard:
        create_dir(opt.tf_dir, opt.log_name, run_time)
        GlobalTensorboard().init_config(tf_path=opt.tf_dir, store_name=os.path.join(opt.log_name, run_time))
        GlobalLogger().get_logger().info(
            "Using tensorboard, store at: {}".format(os.path.join(opt.tf_dir, opt.log_name, run_time)))
    GlobalLogger().get_logger().info("Using training data: {}".format(opt.data_train))
    GlobalLogger().get_logger().info("Pack this file using this command if training finished: {}".format(
        "--logdir " + opt.log_dir + "--log_name " + opt.log_name + "--identifier " + run_time + "--output_checkpoint_dir " + opt.output_checkpoint_dir + "--data_dir " + opt.data_train
    ))
    return opt, run_time


if __name__ == '__main__':
    args, identifier = global_init()
    try:
        train(args, identifier)
        on_finish(args, identifier)

    except Exception as e:
        GlobalLogger().get_logger().error(e)
        on_error(args, identifier, "fail")
        raise e
    except KeyboardInterrupt as k:
        GlobalLogger().get_logger().info("Keyboard Interrupt")
        on_error(args, identifier, "interrupt")

    if args.pack:
        import subprocess

        subprocess.run(
            ["python", "pack_upload.py", "--logdir", args.log_dir, "--log_name ", args.log_name, "--identifier ",
             identifier, "--output_checkpoint_dir ", args.output_checkpoint_dir, "--data_dir ", args.data_train])
