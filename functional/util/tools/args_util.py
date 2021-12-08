import argparse
import os


def parse_train_opt() -> argparse.Namespace:
    """
    Parse the argument when the training procedures starts
    :return: argparse.Namespace, the parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train options')
    # The name of this log, e.g. "base"
    parser.add_argument('--log_name', type=str, help='The name of the log')
    # The path to the training dataset, default is "dataset/baseline"
    parser.add_argument('--data_train', type=str, help='The path to input directory')
    # The path to the evaluation dataset, default is "dataset/baseline", and is the same of the training dataset
    parser.add_argument('--data_eval', type=str, help='The path to evaluation directory')

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
    # Whether to automatically pack the output when finished
    parser.add_argument("--pack", action="store_true")

    return parser.parse_args()


def parse_pack_opt() -> argparse.Namespace:
    """
    Parse the argument when the packing up procedures starts
    :return: argparse.Namespace, the parsed arguments
    """
    parser = argparse.ArgumentParser(description='Pack options')
    # THe path to the log directory, which contains the "train_config.py"
    parser.add_argument('--log_dir', type=str, help='The path to log directory', default="log")
    # The name of the log, used to identify the log and the checkpoints
    parser.add_argument('--log_name', type=str, help='The name of the log')
    # The time identifier, used to identify the log and the checkpoints
    parser.add_argument('--identifier', type=str, help='The identifier string for pack')
    # The path to the checkpoint directory, used to get the output
    parser.add_argument('--output_checkpoint_dir', type=str, help='The path to evaluation directory',
                        default="checkpoint")
    # The path to the dataset, which contains the "*.npy"
    parser.add_argument('--data_dir', type=str, help='The path to the data set')
    return parser.parse_args()


def parse_plot_opt() -> argparse.Namespace:
    """
    Parse the argument when the plotting the dataset procedures starts
    :return: argparse.Namespace, the parsed arguments
    """
    parser = argparse.ArgumentParser(description='Pack options')
    # The path to output
    parser.add_argument('--output_dir', type=str, help="The path to the output directory", default="output")
    # The path to the dataset, which contains the "*.npy"
    parser.add_argument('--data_dir', type=str, help='The path to the data set')
    # The name of the log, used to identify the log and the checkpoints
    parser.add_argument('--log_name', type=str, help='The name of the log')
    # The batch to show
    parser.add_argument('--batch_size', type=int, help='The size of a batch', default=16)
    # The row to plot
    parser.add_argument('--row', type=int, help='The size of a batch', default=4)
    parser.add_argument('--class_first', action="store_true", help='Show images with same class')
    return parser.parse_args()


def parse_generation_opt() -> argparse.Namespace:
    """
    Parse the argument when the generating the dataset
    :return: argparse.Namespace, the parsed arguments
    """
    parser = argparse.ArgumentParser(description='Generate options')
    # The path to the output directory
    parser.add_argument('--output_data_path', type=str, help="The path to the output directory", default="dataset")
    # The base dataset to use, the given should be a str with the same name in "functional.generator_function.dataset_function"
    parser.add_argument('--base_dataset', type=str,
                        help="The basic dataset to use, see \"functional.generator_function.dataset_function\"",
                        default="cifar10_test")
    # The name of the generated dataset
    parser.add_argument('--store_name', type=str, help="The name of the dataset", required=True)
    # The max size of the generated dataset
    parser.add_argument('--max_length', type=int, help="The max size of the generated dataset", default=50000)
    # The config to use, the given should be a str with the same name in "configs.generate_config"
    parser.add_argument('--config', type=str,
                        help="The function to use for generation, see \"configs.generate_config\"", required=True)
    # If the previous data exist, then delete it if given the parameter
    parser.add_argument('--cover', action="store_true", help="Whether to cover the dataset if exists")
    return parser.parse_args()
