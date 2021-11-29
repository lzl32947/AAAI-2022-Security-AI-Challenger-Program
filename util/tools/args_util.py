import argparse
import os


def parse_opt() -> argparse.Namespace:
    """
    Parse the argument when the program starts
    :return: argparse.Namespace, the parsed arguments
    """
    parser = argparse.ArgumentParser(description='Options')
    # The name of this log, e.g. "base"
    parser.add_argument('--log_name', type=str, help='The name of the log')
    # The path to the training dataset, default is "dataset/baseline"
    parser.add_argument('--data_train', type=str, help='The path to input directory',
                        default=os.path.join("dataset", "baseline"))
    # The path to the evaluation dataset, default is "dataset/baseline", and is the same of the training dataset
    parser.add_argument('--data_eval', type=str, help='The path to evaluation directory',
                        default=os.path.join("dataset", "baseline"))

    # The path to the checkpoint, default is "checkpoint"
    parser.add_argument('--output_checkpoint_dir', type=str, help='The path to evaluation directory',
                        default="checkpoint")
    # The internal interval of the training procedures, namely how many epochs should be passed when training
    parser.add_argument('--eval_per_epoch', type=int, help='The internal evaluation in training', default=20)
    # The path to the log directory
    parser.add_argument('--log_dir', type=str, help="The path to the log directory", default="log")
    # The path to tensorboard
    parser.add_argument('--tf_dir', type=str, help="The path to the tensorboard directory", default="tensorboard")
    # The path to output
    parser.add_argument('--output_dir', type=str, help="The path to the output directory", default="output")

    # Whether to ignore warning
    parser.add_argument("--not_ignore_warning", action="store_true",
                        help="if given, not to ignore warnings raised by torch")
    # Enable tensorboard
    parser.add_argument("--enable_tensorboard", action="store_true",
                        help="if given, tensorboard will be used to record the processes")

    return parser.parse_args()
