import math
from typing import List

import torch
from AudioFragmentDataset import AudioFragmentDataset
import torch.nn as nn


def get_num_model_params(module: nn.Module):
    params = 0
    for p in list(module.parameters()):
        num_params = 1
        for s in list(p.size()):
            num_params = num_params * s
        params += num_params

    return params


def calc_config_param_cost_and_viability(window_size: int, conv_config: List) -> (int, int):
    cost = 0
    in_chans = 1
    input_size = window_size

    for layer_config in conv_config:
        out_chans, kernel_size, pool_stride = layer_config
        stride = kernel_size_to_stride(kernel_size, input_size)
        padding = same_size_padding(kernel_size, stride)
        layer_cost = kernel_size // stride * (input_size + 2 * padding) * in_chans * (2 ** out_chans)
        input_size = calc_output_size(input_size, kernel_size, stride, pool_stride, padding)
        if input_size <= 0:
            return -1, False
        cost += layer_cost

    return cost, True


def same_size_padding(kernel_size:int, stride:int):
    return math.ceil(kernel_size - stride) // 2


def kernel_size_to_stride(kernel_size: int, input_size: int):
    stride = 1 if input_size == 1 else max(1, round(math.floor(math.sqrt(kernel_size))))
    return stride


def calc_output_size(input_size, kernel_size, stride, pool_stride, padding):
    return math.floor((input_size + 2 * padding - 1 * (kernel_size - 1) - 1) // stride + 1) // pool_stride


def arch_hash(nas_config: dict) -> int:
    hsh: int = 67
    for conv_block in nas_config["conv_config"]:
        for var in conv_block:
            hsh *= var + 1919941
            hsh *= hsh * 16457
            hsh %= 381837

    for resid_pairs in nas_config["resid_cxns"]:
        hsh *= (-130307 * resid_pairs[0] + 817513 * resid_pairs[1])
        hsh %= 717031

    return hsh


def calc_conv_ops(input_size: int, in_chans: int, out_chans: int, kernel_size: int, stride: int):
    return input_size // stride * (in_chans * kernel_size + 1) * out_chans


# globally define these, but lazy load them so they don't waste time during tests and evals
train_dataset = None
eval_dataset = None
train_data = None
eval_data = None


def load_datasets(device: torch.device,
                  max_train_samples: int = 512 * 1000,
                  max_train_files: int = 30,
                  max_eval_samples: int = 512 * 100,
                  max_eval_files: int = 5,
                  dyn_range_squash_amt: int = 8,
                  **model_config) -> (torch.Tensor, torch.Tensor, AudioFragmentDataset):
    global train_dataset, eval_dataset, train_data, eval_data

    if train_dataset is None:
        train_dataset = AudioFragmentDataset(root_dir="audio/train/",
                                             max_samples=max_train_samples,
                                             max_files=max_train_files,
                                             data_squash_amt=dyn_range_squash_amt,
                                             **model_config)
        train_data = train_dataset.make_tensor_dataset().to(device)

    if eval_dataset is None:
        eval_dataset = AudioFragmentDataset(root_dir="audio/eval/",
                                            max_samples=max_eval_samples,
                                            max_files=max_eval_files,
                                            data_squash_amt=dyn_range_squash_amt,
                                            **model_config)
        eval_data = eval_dataset.make_tensor_dataset().to(device)

    return train_data, eval_data, train_dataset


def print_architecture(arch: dict):
    print(arch)
    print()

    conv_blocks = arch["conv_blocks"]
    resid_blocks = arch["resid_blocks"]

    for cblock in conv_blocks:
        for layer in cblock.layers:
            print(layer)
        print()

    for rblock in resid_blocks:
        print(rblock)


def get_device(debug=False) -> (bool, torch.device):
    use_cuda = torch.cuda.is_available() and not debug
    device = torch.device("cuda:0" if use_cuda else "cpu")

    return use_cuda, device