import math

import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self,
                 src_idx: int, src_chans: int, src_size: int,
                 dst_idx: int, dst_chans: int, dst_size: int):
        """
        These indices model the conv processing of tensors as a chain of tensor nodes connected by conv-block edges:

            T[0] -> conv_block[0] -> T[1] -> conv_block[1] -> ... conv_block[n-1] -> T[n]
                                      ↑
             ╰──────── (0, 1) ────────╯
                                                                                      ↑
                                      ╰─────────────────── (1, n) ────────────────────╯

        Naturally, there are n - 1 conv blocks sandwiched between n tensors.
        Here, the convention of tensor shapes is [N, C, L] for batch size, channels, and spatial sizes respectively.

        To add tensors of varying dimensions, a few cases will be handled separately:
            • if the C and L dimensions are equal, they are compatible so just add them
            • if the C_src and L_dst are equal, or L_src and C_dst are equal,
                reshape to make the common lengths compatible dim, then apply (un)pooling to the inequal dim

            • otherwise we'll reshape and 1x1-conv, preserving as much information as possible.
                "Preserving information" means doing the least resizing of tensors possible.
                Considering the set of dimensions C and L as x,y for cartesian points:
                1. if |src^T - dst| < |src - dst| (i.e., transposing first point is closer
                                                  to the second point than if it were not transposed)
                    then transpose C and L in source tensor
                2. if C_src != C_dst, add a 1x1 conv layer to make the C dims compatible
                3. if L_src != L_dst, add an (un)pool layer to make the L dims compatible


        :param src_idx:     the source tensor index in the chain
        :param src_chans:   the C dimension of the source tensor:
                            number of output filter maps in the conv_blocks[src_index - 1] convolution block,
                            or 1 if src_index is 0 (i.e. no preceding conv blocks)
        :param src_size:    the L dimension of the source tensor
        :param dst_idx:     the dest tensor index the chain
        :param dst_chans:   the C dimension of the dest tensor:
                            number of output filter maps in the conv_blocks[dst_index - 1] convolution block
        :param dst_size:    the L dimension of the dest tensor
        """
        super().__init__()
        self.src_idx = src_idx
        self.dst_idx = dst_idx
        self.dst_chans = dst_chans
        self.conv = None
        self.pool = None
        self.early = None
        self.transpose = False
        self.unpool = False

        src = [src_chans, src_size]
        dst = [dst_chans, dst_size]

        # just for debugging, need some record
        self.src = src
        self.dst = dst

        src_t = src[::-1]

        if math.dist(src_t, dst) < math.dist(src, dst):
            src = src_t
            self.transpose = True

        if src[0] != dst[0]:
            self.conv = nn.Conv1d(src[0], dst[0], kernel_size=1)

        if src[1] > dst[1]:
            ratio = math.ceil(src[1] / dst[1])
            padding = min(ratio // 2, (dst[1] * ratio - src[1]))
            self.pool = nn.AvgPool1d(ratio, padding=padding)

        elif src[1] < dst[1]:
            self.unpool = True

        self.relu = nn.ReLU(True)
        self.norm = nn.BatchNorm1d(src_chans)
        self.prenorm = nn.BatchNorm1d(dst_chans)
        self.postnorm = nn.BatchNorm1d(dst_chans)

    def start(self, x):
        self.early = x

    def finish(self, current):
        if self.transpose:
            self.early = self.early.view(self.early.size(0), self.early.size(2), -1)

        if self.conv:
            self.early = self.conv(self.early)

        if self.pool:
            self.early = self.pool(self.early)

        if self.unpool:
            self.early = F.interpolate(self.early, self.dst[1])

        if self.early.size() != current.size():
            raise Exception(f"({self.src_idx}, {self.dst_idx}) Tensors "
                            f"{self.early.size()} and {current.size()} have incompatible sizes")

        current = current + self.early
        current = self.prenorm(current)
        current = self.relu(current)
        current = self.postnorm(current)

        return current

    def __str__(self):
        s = f"Residual block ({self.src_idx}, {self.dst_idx}) from {self.src} -> {self.dst}:\n"
        if self.transpose:
            s += "\ttranspose\n"
        if self.conv:
            s += "\tconv: " + self.conv.__str__() + "\n"
        if self.pool:
            s += "\tpool: " + self.pool.__str__() + "\n"
        if self.unpool:
            s += f"\tunpool: ratio={self.dst[1] // self.src[0 if self.transpose else 1]}, dest chans={self.dst_chans}\n"
        return s

    def set_device(self, device):
        self.to(device)

