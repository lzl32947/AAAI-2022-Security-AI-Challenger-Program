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
        composed_transform = compose_config(config)
        # Generate the runnable
        runnable = ArgumentRunnable(datasets, composed_transform)
        # Execute the runnable
        data, label = runnable()
        # Get the description
        description = str(composed_transform)
        # Save data
        np.save(os.path.join(target_dir, "data.npy"), data)
        np.save(os.path.join(target_dir, "label.npy"), label)
        # Save description
        with open(os.path.join(target_dir, "description.txt"), "w", encoding="utf-8") as fout:
            fout.write(description)
    except (NameError, ValueError, FileNotFoundError, FileExistsError, RuntimeError, KeyboardInterrupt) as e:
        print("Fail to generate the dataset!")
        remove_dir(opt.output_data_path, opt.store_name)
        print(e)
