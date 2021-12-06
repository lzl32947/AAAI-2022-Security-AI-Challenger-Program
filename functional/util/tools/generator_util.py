import importlib
from typing import List, Dict

from functional.generator_function.global_definition import ImageCompose


def compose_config(config_list: List) -> ImageCompose:
    """
    The compose function to generate the sequential transform
    :param config_list: str, the name of the config
    :return: ImageCompose, the composed transform
    """
    # Load module dynamically
    m1 = importlib.import_module("functional.generator_function.transforms.iaa_transform")
    m2 = importlib.import_module("functional.generator_function.transforms.custom_transform")
    target_list = []
    # "item" in format [str(name), float(possibility), Dict(kwargs)]
    for item in config_list:
        if hasattr(m1, item[0]):
            target_list.append(getattr(m1, item[0])(item[1], **item[2]))
        elif hasattr(m2, item[0]):
            target_list.append(getattr(m2, item[0])(item[1], **item[2]))
    return ImageCompose(target_list)
