from typing import Union, List, Tuple

import numpy as np
import torch

global_label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def _process_single_onehot_label(inputs: np.ndarray, top_k):
    inputs = np.squeeze(inputs)
    if len(inputs.shape) > 1:
        return None
    else:
        answer_dict = {}
        labels = inputs.copy()
        labels.sort()
        labels = labels[::-1]
        confidence_value = labels[:top_k]
        for i in range(top_k):
            confidence = confidence_value[i]
            if confidence > 0:
                index = np.where(inputs == confidence)[0][0]

                answer_dict[global_label[index]] = confidence
        return answer_dict


def get_label_name(inputs: Union[int, np.ndarray, List, Tuple, torch.Tensor], top_k=3):
    if isinstance(inputs, int):
        if 0 <= inputs < len(global_label):
            return [{global_label[inputs]: 1.0}]
        else:
            return None
    if isinstance(inputs, torch.Tensor):
        if inputs.is_cuda:
            inputs = inputs.detach().cpu()
        inputs = inputs.numpy()
    if isinstance(inputs, (list, tuple)):
        if isinstance(inputs[0], (float, int)):
            return [_process_single_onehot_label(np.array(inputs), top_k)]
        else:
            inputs = np.array(inputs)
    result_list = []
    for item in inputs:
        result_list.append(_process_single_onehot_label(item, top_k))
    return result_list


def get_class(inputs: Union[int, np.ndarray, List, Tuple, torch.Tensor]):
    if isinstance(inputs, int):
        if 0 <= inputs < len(global_label):
            return inputs
        else:
            return None
    if isinstance(inputs, torch.Tensor):
        if inputs.is_cuda:
            inputs = inputs.detach().cpu()
        inputs = inputs.numpy()
    if isinstance(inputs, (list, tuple)):
        if isinstance(inputs[0], (float, int)):
            return np.argmax(np.array(inputs))
        else:
            inputs = np.array(inputs)
    return np.argmax(inputs, axis=1)
