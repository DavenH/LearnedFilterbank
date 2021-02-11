from typing import List, Iterator
import math
import torch.nn as nn
import Utils as util


class ConvBlock(nn.Module):
    def __init__(self, in_chan: int,
                 input_size: int,
                 out_chan_log2: int,
                 kernel_size: int,
                 pool_stride: int):
        super().__init__()

        self.in_chan = in_chan
        self.out_chan = 2 ** out_chan_log2
        self.kernel_size = 1 if input_size == 1 else kernel_size
        self.pool_stride = pool_stride
        self.computed_input_size = input_size
        self.stride = util.kernel_size_to_stride(self.kernel_size, input_size)
        self.padding = util.same_size_padding(self.kernel_size, self.stride)

        self.conv = None
        self.pool = None
        self.norm = nn.BatchNorm1d(num_features=self.out_chan)
        self.relu = nn.ReLU(True)

    # decoupled from init because sometimes we need to do param
    # adjustments for compatibility before committing to them
    def create_layers(self):
        self.conv = nn.Conv1d(in_channels=self.in_chan, out_channels=self.out_chan,
                              kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)

        if 1 < self.pool_stride <= self.computed_input_size:
            self.pool = nn.MaxPool1d(kernel_size=self.pool_stride, stride=self.pool_stride)

    def get_output_size(self) -> int:
        return math.floor((self.computed_input_size + 2 * self.padding - 1 * (
                self.kernel_size - 1) - 1) // self.stride + 1) // self.pool_stride

    def get_compact_conv_config(self) -> List:
        return [self.out_chan.bit_length() - 1, self.kernel_size, self.pool_stride]

    def forward(self, x):
        x = self.conv(x)
        if self.pool:
            x = self.pool(x)
        x = self.norm(x)
        x = self.relu(x)
        return x

    def set_device(self, device):
        self.to(device)

