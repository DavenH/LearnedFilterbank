import copy
import time
import os
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader

from ConvBlock import ConvBlock
from ResidualBlock import ResidualBlock
import Genetics as genetics
import Utils as util
from typing import List, Any, Optional

torchaudio.set_audio_backend("sox_io")


class FilterbankModule(nn.Module):

    def __init__(self, train_data, eval_data, **model_config):

        super().__init__()

        print("Initializing")

        # avoid annoying warnings by initializing here
        self.filterbank = self.loss_fn = self.linear = self.softmax = None
        self.window_size = self.batch_size = self.n_filters = 0
        self.config: dict[str, Any] = {}
        self.skip_batch = None
        self.optimizer = None
        self.conv_blocks: List[ConvBlock] = []
        self.resid_blocks: List[ResidualBlock] = []
        self.device = model_config["device"]
        self.use_cuda = model_config["use_cuda"]
        self.epoch = 0
        self.train_time = 0

        self.init_params(model_config)
        self.softmax = nn.Softmax(dim=1)
        self.loss_fn = nn.MSELoss()

        self.epoch = 0
        if "load_from_checkpoint" in model_config:
            self.load(model_config["model_file"])
        else:
            self.restore_architecture(restore_from_config=self.config["restore_from_config"])
            self.set_device(self.device)

        num_workers = 0 if self.use_cuda or model_config["debug"] else 6
        self.loader = torch.utils.data.DataLoader(train_data,
                                                  batch_size=self.batch_size,
                                                  num_workers=num_workers,
                                                  shuffle=True,
                                                  drop_last=True)

        self.eval_loader = torch.utils.data.DataLoader(eval_data,
                                                       batch_size=self.batch_size,
                                                       num_workers=num_workers,
                                                       shuffle=True,
                                                       drop_last=True)

    def setup(self, config):
        pass

    def reset_config(self, config):
        print("Resetting config")

        self.init_params(config)

        opt_params = ["lr", "momentum", "beta1", "beta2"]
        for param_group in self.optimizer.param_groups:
            for param in opt_params:
                if param in self.config["arch"]:
                    print("Setting ", param, " to ", self.config["arch"][param])
                    param_group[param] = self.config["arch"][param]

        return True

    def init_params(self, config):
        if "config" in config:
            self.config = config["config"]
        else:
            self.config = config

        self.window_size = self.config["window_size"]
        self.batch_size = self.config["batch_size"]
        self.n_filters = self.config["wavelet_kernels"]

    def restore_architecture(self, restore_from_config=True):
        """
        To be called only after setting the self.config field
        :param restore_from_config:
            whether to build convnet architecture from the
            configuration dict, or alternatively randomize
            a new architecture
        """
        if self.config is None:
            raise ValueError("Configuration dict must be set to restore architecture!")

        self.n_filters = self.config["wavelet_kernels"]

        rand = torch.randn(self.n_filters, self.window_size)
        torch.tanh_(rand)   # restrict domain to [-1, 1]
        self.filterbank = torch.nn.Parameter(rand)

        config_to_use = self.config["arch"] if restore_from_config else None
        self.conv_blocks, self.resid_blocks, out_channels, self.config["arch"] = \
            self.make_arch_from_config(config_to_use)

        self.linear = nn.Linear(out_channels, self.n_filters)

        self.restore_optimizer()
        self.set_device(self.device)

    def make_arch_from_config(self, arch_config: Optional[dict]) -> (List, int, int, dict):

        if not arch_config:
            arch_config = genetics.get_decent_arch_config(self.window_size)
        else:
            arch_config = copy.deepcopy(arch_config)

        if arch_config is None:
            raise Exception("Could not find configuration")

        conv_blocks = []
        in_channels = 1
        comp_input_size = self.window_size

        conv_config_layers = arch_config["conv_config"]
        corrected_conv_config = []
        for i, config_layer in enumerate(conv_config_layers):
            out_chan_log2, kernel_size, pool_stride = config_layer

            if i == len(conv_config_layers) - 1:
                # kernel_size = 1  # ensures a maximum of stride length 1
                out_chan_log2 = self.window_size.bit_length() - 1

            block = ConvBlock(in_channels, comp_input_size, out_chan_log2, kernel_size, pool_stride)

            while block.get_output_size() <= 0 and block.pool_stride > 1:
                block.pool_stride -= 1

            comp_input_size = block.get_output_size()

            if i == len(conv_config_layers) - 1:
                block.pool_stride *= max(1, comp_input_size)  # makes output size become 1
                comp_input_size = block.get_output_size()
                if comp_input_size != 1:
                    raise RuntimeError("Conv encoder not producing an output of size 1")

            block.create_layers()
            conv_blocks.append(block)
            corrected_conv_config.append(block.get_compact_conv_config())

            # print(f"{i}\tCi={in_channels.bit_length() - 1}\tCo={out_chan_log2}\tLi={input_size}\tLo={comp_input_size}"
            #       f"\tk={kernel_size}\ts={block.stride}\tps={block.pool_stride}")

            if comp_input_size == 0:
                raise RuntimeError("Computed input size is zero; conv config:", conv_config_layers)
            in_channels = 2 ** out_chan_log2

        arch_config["conv_config"] = corrected_conv_config

        resid_blocks = []
        if "resid_cxns" in arch_config:
            resid_config = arch_config["resid_cxns"]
            for index_pair in resid_config:
                source_idx, dest_idx = index_pair
                if source_idx < 0 or dest_idx - 1 <= source_idx or dest_idx > len(conv_config_layers):
                    print(f"Bad residual connection config: {source_idx}, {dest_idx}")
                    continue

                dest_size = conv_blocks[dest_idx - 1].get_output_size()

                resid_blocks.append(ResidualBlock(src_idx=source_idx,
                                                  src_chans=conv_blocks[source_idx].in_chan,
                                                  src_size=conv_blocks[source_idx].computed_input_size,
                                                  dst_idx=dest_idx,
                                                  dst_chans=conv_blocks[dest_idx - 1].out_chan,
                                                  dst_size=dest_size))

        return conv_blocks, resid_blocks, in_channels, arch_config

    def restore_optimizer(self):
        self.optimizer = torch.optim.Adam(self.parameters(), self.config["arch"]["lr"])
        for block in self.resid_blocks:
            self.optimizer.add_param_group({"params": block.parameters()})

        for block in self.conv_blocks:
            self.optimizer.add_param_group({"params": block.parameters()})

    def forward(self, x: torch.Tensor):
        for i, conv_block in enumerate(self.conv_blocks):

            for resid_cxn in self.resid_blocks:
                if resid_cxn.src_idx == i:
                    resid_cxn.start(x)

            x = conv_block(x)

            for resid_cxn in self.resid_blocks:
                if resid_cxn.dst_idx == i + 1:
                    x = resid_cxn.finish(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)

        filterbank_weights = self.softmax(x)
        x = filterbank_weights.matmul(self.filterbank)

        return x, filterbank_weights

    def step(self):
        n = self.config["epochs_per_step"]
        total_loss = 0
        total_batches = 0
        nn.Module.train(self, mode=True)

        large_wv_penalty = self.config["arch"]["large_penalty"]
        small_wv_penalty = self.config["arch"]["small_penalty"]
        sparsity_reward = self.config["arch"]["sparse_reward"]
        tic = time.perf_counter()

        for i in range(n):
            epoch_loss = 0
            for batch in self.loader:
                if batch.size(0) < 2:
                    break

                batch = batch.to(self.device)
                self.zero_grad()
                output, lincombs = self.forward(batch.unsqueeze(1))  # unsqueeze adds the 'channel' dimension

                # combin_norm = torch.norm(lincombs)  # increase sparsity
                wavelet_norms = torch.norm(self.filterbank, dim=1)
                wavelet_norm_avg = torch.mean(wavelet_norms.pow(2))  # penalize overly large kernels

                # log_norms = torch.log(wavelet_norms)
                # wv_lg_norm_avg = torch.min(log_norms)  # penalize near-zero norms
                # # norm_error = (wavelet_norm_avg * 0.1 - combin_norm * 0.25 - wv_lg_norm_avg) * 0.7e-4
                # - wv_lg_norm_avg * small_wv_penalty - combin_norm * sparsity_reward
                norm_error = wavelet_norm_avg * large_wv_penalty

                error = self.loss_fn.forward(batch, output) + norm_error
                epoch_loss += error.item()

                error.backward()
                self.optimizer.step()
                total_batches += 1

            total_loss += epoch_loss

        torch.cuda.current_stream().synchronize()
        toc = time.perf_counter()
        train_time = toc - tic
        self.train_time += train_time

        # normalize loss by how many batches we've computed
        total_loss /= total_batches

        self.epoch += n

        return 1000 * total_loss, train_time

    def train_model(self) -> (float, float):
        epochs = self.config["epochs"]

        epochs_per_step = self.config["epochs_per_step"]
        max_epochs_without_improvement = self.config["max_epochs_no_imprv"]

        best_loss = 1e10
        best_loss_index = 0
        total_train_time = 0

        # print("\t\ttrain\teval")
        for i in range(epochs // epochs_per_step):
            train_loss, train_time = self.step()
            total_train_time += train_time
            eval_loss = self.evaluate()
            
            if eval_loss < best_loss:
                best_loss = eval_loss
                best_loss_index = i

            if i - best_loss_index >= max_epochs_without_improvement:
                break

            print(f"\t\t{self.epoch}\t{train_loss:0.2f}\t{eval_loss:0.2f}\t{train_time:2.2f}s")

        return best_loss, total_train_time

    def evaluate(self) -> float:
        nn.Module.eval(self)
        n = self.config["epochs_per_step"]
        total_batches = 0
        total_loss = 0
        with torch.no_grad():
            for _ in range(n):
                epoch_loss = 0
                for batch in self.eval_loader:
                    if batch.size(0) < 2:
                        break

                    batch = batch.to(self.device)
                    output, lincombs = self.forward(batch.unsqueeze(1))  # unsqueeze adds the 'channel' dimension
                    error = self.loss_fn.forward(batch, output)
                    epoch_loss += error.item()
                    total_batches += 1
                total_loss += epoch_loss
        total_loss /= total_batches

        return 1000 * total_loss

    def save_checkpoint(self, tmp_checkpoint_dir):
        
        hsh = util.arch_hash(self.config["arch"])
        model_file = f"wavelet-{self.n_filters}-{self.window_size}-{hsh}-{self.epoch}.pt"
        path = os.path.join(tmp_checkpoint_dir, model_file)

        torch.save({
            "train_time": self.train_time,
            "epoch": self.epoch,
            "config": self.config,
            "model": self.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "conv_blocks": [block.state_dict() for block in self.conv_blocks],
            "resid_blocks": [block.state_dict() for block in self.resid_blocks]
        }, path)

        return path

    def load_checkpoint(self, checkpoint):
        self.load(checkpoint)

    def load(self, model_file):
        checkpoint = torch.load(model_file)
        self.epoch = checkpoint.get("epoch", 0)
        self.train_time = checkpoint.get("train_time", 0)
        self.init_params(checkpoint["config"])
        self.restore_architecture(restore_from_config=True)
        self.load_state_dict(checkpoint["model"])

        for i, block_state in enumerate(checkpoint["conv_blocks"]):
            self.conv_blocks[i].load_state_dict(block_state)

        for i, block_state in enumerate(checkpoint["resid_blocks"]):
            self.resid_blocks[i].load_state_dict(block_state)

        self.set_device(self.device)
        self.restore_optimizer()
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        if self.use_cuda:
            # as per https://github.com/pytorch/pytorch/issues/2830
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cuda()

    def set_device(self, device):
        self.to(device)
        for block in self.conv_blocks:
            block.set_device(device)
        for block in self.resid_blocks:
            block.set_device(device)

    def get_num_model_params(self):
        params = 0
        params += util.get_num_model_params(self)

        for block in self.resid_blocks:
            params += util.get_num_model_params(block)
        for block in self.conv_blocks:
            params += util.get_num_model_params(block)

        return params