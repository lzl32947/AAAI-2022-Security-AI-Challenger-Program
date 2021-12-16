import importlib
from typing import List, Dict, Tuple

from functional.generator_function.global_definition import ImageCompose


def compose_config(config_dict: Dict) -> Tuple[List[ImageCompose], Dict]:
    """
    The compose function to generate the sequential transform
    :param config_dict: Dict, the parsed configs for generations
    :return: ImageCompose, the composed transform
    """
    # Load module dynamically
    m1 = importlib.import_module("functional.generator_function.transforms.iaa_transform")
    m2 = importlib.import_module("functional.generator_function.transforms.custom_transform")
    target_list = []
    extra_config = config_dict["config"]
    config_list = config_dict["procedures"]
    # how many times to compose
    separate = len(config_list)
    for index in range(0, separate):
        inner_target = []
        # "item" in format [str(name), float(possibility), Dict(kwargs)]
        for item in config_list[index]:
            if hasattr(m1, item[0]):
                inner_target.append(getattr(m1, item[0])(item[1], **item[2]))
            elif hasattr(m2, item[0]):
                inner_target.append(getattr(m2, item[0])(item[1], **item[2]))
        target_list.append(ImageCompose(inner_target))
    return target_list, extra_config
