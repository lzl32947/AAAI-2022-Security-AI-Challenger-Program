import importlib

from functional.generator_function.global_definition import ImageCompose


def compose_config(config_list):
    m = importlib.import_module("functional.generator_function.transforms.iaa_transform")
    target_list = []
    for item in config_list:
        if hasattr(m, item[0]):
            target_list.append(getattr(m, item[0])(item[1], **item[2]))
    return ImageCompose(target_list)
