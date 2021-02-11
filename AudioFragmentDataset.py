import os
import torch
import random as rand
from torch.utils.data import Dataset
import torchaudio
import math


class AudioFragmentDataset(Dataset):
    def __init__(self, root_dir: str, max_samples:int, max_files: int, data_squash_amt=8, **kwargs):
        self.root_dir = root_dir
        self.max_samples = max_samples
        self.max_data_files = max_files
        self.data_squash_amt = data_squash_amt
        self.window_size = kwargs["window_size"]
        self.seed = kwargs["seed"]
        self.rand = rand.Random(self.seed)
        self.child_files = []

        self.cached_wavs = {}

        with torch.no_grad():
            line = torch.linspace(0, math.tau, self.window_size)
            self.nuttall_window = 0.355768 \
                                - 0.487396 * torch.cos(line) \
                                + 0.144232 * torch.cos(line * 2) \
                                - 0.012604 * torch.cos(line * 3)
            self.nuttall_window.clamp_(0, 1)

        for file in os.listdir(self.root_dir):
            if file.endswith((".wav", ".mp3", ".flac")):
                self.child_files.append(os.path.join(self.root_dir, file))

    def __getitem__(self, n: int) -> torch.Tensor:
        self.rand.seed(n * 500000 + self.seed)
        file_name = "unset"

        # max tries to get a non-zero norm'd piece of signal
        tries = 10
        for i in range(tries):
            file_idx = self.rand.randint(0, min(self.max_data_files, len(self.child_files) - 1))
            file_name = self.child_files[file_idx]

            # get from cache if available, if not from disk
            if file_name in self.cached_wavs:
                waveform = self.cached_wavs[file_name]
            else:
                print ("Loading ", file_name)
                waveform, sample_rate = torchaudio.load(file_name)
                self.cached_wavs[file_name] = waveform

            wav_len = waveform.size(1)
            offset = self.rand.randint(0, wav_len - self.window_size)
            chan = self.rand.randint(0, waveform.size(0) - 1)
            section = waveform[chan, offset:offset + self.window_size]

            # test if signal is non-zero. Zero signals cause NaNs
            norm = torch.linalg.norm(section)

            if norm.item() > 0.001:
                # apply nutall window and normalize
                with torch.no_grad():
                    window = torch.mul(section, self.nuttall_window)

                    max_level = max(torch.max(window), -torch.min(window))
                    # don't fully scale from [-1: 1], just squash it to that level most of the time,
                    # but background noise will get amplified a lot by complete normalization
                    multiplicand = math.pow(max_level, -(self.data_squash_amt - 1) / self.data_squash_amt)

                    window.mul_(multiplicand)

                return window

        raise Exception("Waveform " + file_name + " is zeroes, will cause NaNs")

    def __len__(self):
        return self.max_samples

    def make_tensor_dataset(self) -> torch.Tensor:

        big_tensor = torch.Tensor(self.max_samples, self.window_size)
        print("Prefetching data...")

        for i in range(self.max_samples):
            big_tensor[i] = self.__getitem__(i)
            if i % 10000 == 0:
                print(f"{i}/{self.max_samples}")

        print("Done")
        return big_tensor
