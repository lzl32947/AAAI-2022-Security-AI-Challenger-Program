import importlib
import os
import random

import numpy as np
import torch

from functional.generator_function.global_definition import ArgumentRunnable
from functional.util.tools.args_util import parse_generation_opt
from functional.util.tools.file_util import create_dir, remove_dir

# Init the random values
from functional.util.tools.generator_util import compose_config

seed = 11037
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # Parse arguments
    opt = parse_generation_opt()
    # Create the directory for storing
    create_dir(opt.output_data_path)
    create_dir("temp")
    # Generate the target path
    target_dir = os.path.join(opt.output_data_path, opt.store_name)
    # Create the directory for storage
    if not os.path.exists(os.path.join(opt.output_data_path, opt.store_name)):
        create_dir(opt.output_data_path, opt.store_name)
    else:
        print("{} already exist!".format(target_dir))
        if opt.cover:
            # Remove the original files and re-generate the new one
            print("Cover enable, replace the previous one...")
            remove_dir(opt.output_data_path, opt.store_name)
            create_dir(opt.output_data_path, opt.store_name)
        else:
            print("Change it to another one!")
            exit(-1)
    max_length = opt.max_length
    config = opt.config
    try:
        # Dynamically load the config files
        module = importlib.import_module("configs.generate_config")
        # Get the configs
        if hasattr(module, config):
            config = getattr(module, config)
        else:
            print("{} not found!".format(config))
            raise RuntimeError
        # Dynamically load the dataset
        dataset_module = importlib.import_module("functional.generator_function.dataset_function")
        if hasattr(dataset_module, opt.base_dataset):
            datasets = getattr(dataset_module, opt.base_dataset)()
        else:
            print("{} not found!".format(opt.base_dataset))
            raise RuntimeError
        # Compose the transforms
        composed_transform_list, runnable_config = compose_config(config)
        # Set the description
        general_description = []
        # Set the data and label
        general_data = []
        general_label = []
        global_counter = 0
        # Perform the transform
        for index in range(len(composed_transform_list)):
            composed_transform = composed_transform_list[index]
            # Generate the runnable
            runnable = ArgumentRunnable(datasets, composed_transform)
            # Execute the runnable
            data, label = runnable(**runnable_config)
            # Get the description
            description = str(composed_transform)
            # Add the description and data etc.
            general_description.append(description)
            general_label.append(label)
            general_data.append(data)
            # Add the counter
            global_counter += len(data)
            # If over-create
            if global_counter > max_length:
                raise RuntimeError("Create more data than the maximum count!")
        # Transform the data and label
        general_data = np.concatenate(general_data, axis=0)
        general_label = np.concatenate(general_label, axis=0)

        # Save data
        np.save(os.path.join(target_dir, "data.npy"), general_data)
        np.save(os.path.join(target_dir, "label.npy"), general_label)
        # Save description
        with open(os.path.join(target_dir, "description.txt"), "w", encoding="utf-8") as fout:
            fout.write("\n".join(general_description))
    except (NameError, ValueError,TypeError, FileNotFoundError, FileExistsError, RuntimeError, KeyboardInterrupt,AssertionError) as e:
        print("Fail to generate the dataset!")
        remove_dir(opt.output_data_path, opt.store_name)
        raise e
