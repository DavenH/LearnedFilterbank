import math
import os

import torch
import torchaudio
from matplotlib.backend_bases import Event

import Utils as util
from FilterbankModule import FilterbankModule
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np


def train(model_dir: str, checkpoint_to_load: str = None):
    train_tensor, eval_tensor, _ = util.load_datasets(device, dyn_range_squash_amt=16)
    model = FilterbankModule(train_data=train_tensor, eval_data=eval_tensor, **model_config)

    if checkpoint_to_load:
        model.load_checkpoint(checkpoint_to_load)

    model.train_model()

    path = model.save_checkpoint(model_dir)
    return path


def eval(checkpoint: str,
         show_kernels: bool = False,
         show_spectrogram: bool = False,
         show_reconstruct: bool = False,
         transcode: bool = False):
    train_tensor, eval_tensor, _ = util.load_datasets(device, 1000, 1, 1000, 1, dyn_range_squash_amt=16)
    model = FilterbankModule(train_data=train_tensor,
                             eval_data=eval_tensor,
                             **model_config)

    model.load_checkpoint(checkpoint)
    if show_kernels:
        eval_show_kernels(model)
    if show_spectrogram:
        eval_task_spectrogram(model, "audio/eval/lifes tragedy.ogg", 2000, 20000, 4)
    if show_reconstruct:
        eval_task_reconstruct(model, 12)
    if transcode:
        h = util.arch_hash(model_config["arch"])
        eval_task_transcode(model, "audio/eval/lifes tragedy.ogg", f"transcoded_{h}_{model.epoch}", 8000, 80000)

viz_data_index = 0


def eval_task_reconstruct(model: FilterbankModule, num_to_show: int):
    model.eval()
    model.to(device)
    train_data, eval_data, dataset = util.load_datasets(device)

    # plt.figure(figsize=(10, 8))
    height = round(math.ceil(math.sqrt(num_to_show) * 0.7))
    width = num_to_show // height
    actual_num = width * height
    # rand = random.randint(0, min(dataset.max_samples - actual_num, 1000))

    fig, axs = plt.subplots(nrows=height * 2, ncols=width)

    # register this callback to scroll pages of results by clicking on far left or far right of plot
    # matplotlib should really let you hijack those < > buttons instead, but they don't generate events
    def onclick(event: Event):
        global viz_data_index

        if event.guiEvent is not None:
            for i in range(height):
                for j in range(width):
                    axs[2 * i, j].clear()
                    axs[2 * i + 1, j].clear()

            if event.guiEvent.x < 200:
                viz_data_index = max(0, viz_data_index - actual_num)
            elif event.guiEvent.x > event.canvas.figure.get_figwidth() - 200:
                viz_data_index = min(dataset.max_samples - 1, viz_data_index + actual_num)

        samples = torch.Tensor(actual_num, model.window_size)
        for i in range(actual_num):
            if i < actual_num // 2:
                samples[i] = eval_data[i + viz_data_index]
            else:
                samples[i] = train_data[i + viz_data_index]

        # make sure this tensor is on the same device as the module
        samples = samples.to(device)

        # add 'channel' dim
        padded_samples = samples.unsqueeze(1)
        recons, lincombs = model.forward(padded_samples)

        for i in range(height):
            for j in range(width):
                index = i * width + j

                # squeeze to pop off the batch dim
                sample_np = samples[index].cpu().detach().numpy()
                combin_np = lincombs[index].cpu().detach().numpy()
                recon_np = recons[index].cpu().detach().numpy()

                axis_win = [x for x in range(model.window_size)]
                axis_lincomb = [x for x in range(model.wavelet_kernels)]

                axs[i * 2, j].plot(axis_win, recon_np)
                axs[i * 2, j].plot(axis_win, sample_np)
                axs[i * 2 + 1, j].plot(axis_lincomb, combin_np)
        event.canvas.draw()

    onclick(Event('fake-click', fig.canvas))

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.subplots_adjust(wspace=0.1, hspace=0)
    plt.show()


def eval_show_kernels(model: FilterbankModule):
    indices, kernels_cpu = get_indices_of_approx_kernel_freq_ordering(model)
    sorted_kernels = kernels_cpu[indices]
    plt.figure(figsize=(13, 8))
    cmap = sb.color_palette("coolwarm", as_cmap=True)
    sb.heatmap(sorted_kernels, cmap=cmap)
    plt.show()


def eval_task_spectrogram(model: FilterbankModule, file_name: str, num_frames: int, sample_offset:int, zoom_factor):
    model.eval()
    model.to(device)

    _, _, dataset = util.load_datasets(device)

    waveform, sample_rate = torchaudio.load(file_name)
    frames = torch.Tensor(num_frames, model.window_size)
    squash_amt = dataset.data_squash_amt
    window_scale_amts = torch.zeros(num_frames)

    for i in range(num_frames):
        start_idx = i * model.window_size // 4 // zoom_factor + sample_offset
        window = waveform[0, start_idx:start_idx + model.window_size]
        window = window.mul(dataset.nuttall_window)
        norm = torch.norm(window)

        # dynamic range compression on the input data, during model training,
        # needs to be taken into account. So the magnitudes of kernels it outputs
        # need to be scaled inversely to the dyn range compression of the audio window
        max_level = max(torch.max(window), -torch.min(window))
        if norm < 0.01 or max_level == 0:
            frames[i] = torch.zeros(model.window_size)
            window_scale_amts[i] = 0
        else:
            multiplicand = math.pow(max_level, -(squash_amt - 1) / squash_amt)
            window_scale_amts[i] = 1 / multiplicand
            window.mul_(multiplicand)
            frames[i] = window

    window_scale_amts = window_scale_amts.to(device)
    frames = frames.to(device)
    recons, lincombs = model.forward(frames.unsqueeze(1))
    indices, _ = get_indices_of_approx_kernel_freq_ordering(model)

    scaled_lincombs = window_scale_amts.view(-1, 1) * lincombs
    # scaled_lincombs = lincombs
    scaled_lincombs.mul_(300).log1p_()
    scaled_lincombs = scaled_lincombs[:, indices]
    spect = scaled_lincombs.t().cpu().detach().numpy()

    import seaborn as sb
    import librosa.display
    import librosa
    end_sample = sample_offset + num_frames * model.window_size // 4 // zoom_factor
    stft = librosa.stft(waveform[0, sample_offset:end_sample].numpy())
    stft_db = librosa.amplitude_to_db(abs(stft))
    plt.figure(figsize=(14, 5))
    plt.subplot(2, 1, 1)
    librosa.display.specshow(stft_db, sr=sample_rate, x_axis='time', y_axis='hz', cmap="magma")
    plt.subplot(2, 1, 2)
    cmap = sb.color_palette("magma", as_cmap=True)
    sb.heatmap(spect, cbar=False, xticklabels='', cmap=cmap)
    plt.tight_layout()
    plt.show()


def get_indices_of_approx_kernel_freq_ordering(model: FilterbankModule):
    # get the ordering of wavelet kernels that roughly corresponds with increasing frequency modes
    kernels_cpu = model.kernels.cpu().detach().numpy()
    zero_crosses = np.count_nonzero(np.diff(kernels_cpu > 0, axis=1), axis=1)
    indices = np.flip(np.argsort(zero_crosses)).copy()
    return indices, kernels_cpu


def eval_task_transcode(model: FilterbankModule, file_name: str, dest_name: str,
                        num_frames: int, sample_offset: int, copy_input_in_file=True):

    waveform, sample_rate = torchaudio.load(file_name)
    _, _, dataset = util.load_datasets(device)
    nuttall = dataset.nuttall.to(device)

    with torch.no_grad():
        waveform = waveform.to(device)
        frames = torch.zeros(num_frames, model.window_size, device=device)
        squash_amt = dataset.data_squash_amt
        window_scale_amts = torch.zeros(num_frames, device=device)
        start_idx = 0

        for i in range(num_frames):
            window = waveform[0, start_idx + sample_offset:start_idx + sample_offset + model.window_size]
            window = window.mul(nuttall)
            max_level = max(torch.max(window), -torch.min(window))
            norm = torch.norm(window)

            if norm < 0.001 or max_level == 0:
                frames[i] = torch.zeros(model.window_size)
                window_scale_amts[i] = 0
            else:
                multiplicand = math.pow(max_level, -(squash_amt - 1) / squash_amt)
                window_scale_amts[i] = 1 / multiplicand
                window.mul_(multiplicand)

                frames[i] = window

            start_idx += model.window_size // 4
        sample_size = start_idx + model.window_size

        model.eval()

        recons, filt_weights = model.forward(frames.unsqueeze(1))

        if copy_input_in_file:
            silence_samples = 20000
            start_idx += silence_samples
        else:
            start_idx = 0

        out_frames = torch.zeros(1, start_idx + (num_frames + 4) * model.window_size // 4, device=device)

        if copy_input_in_file:
            out_frames[:, 0:sample_size] += waveform[0, sample_offset:sample_offset + sample_size]

        for i in range(num_frames):
            scaled = recons[i].mul(window_scale_amts[i]*0.7)
            out_frames[:, start_idx:start_idx + model.window_size] += scaled
            start_idx += model.window_size // 4

        save_name = os.path.join(model.config["model_dir"],
                                 os.path.splitext(os.path.basename(file_name))[0] + dest_name + ".wav")
        print(save_name)
        out_frames = out_frames.cpu()
        torchaudio.save(save_name, out_frames, sample_rate)


debug = False
use_cuda = torch.cuda.is_available() and not debug
device = torch.device("cuda:0" if use_cuda else "cpu")

model_config = dict(
    # training
    epochs=20,
    batch_size=64,
    max_epochs_no_imprv=10,
    epochs_per_step=1,
    seed=1,

    # arch
    window_size=256,
    wavelet_kernels=500,
    arch=dict(
        conv_config=[[5, 32, 1], [7, 32, 1], [5, 1, 10], [8, 1, 1]],
        resid_cxns=[(0, 4)],
        lr=0.001,

        large_penalty=0.0000086,
        small_penalty=0.00000142,
        sparse_reward=0.00000145,
    ),

    # flags
    restore_from_config=True,
    use_cuda=use_cuda,
    device=device,
    debug=debug
)

train("~/", model_config, )
# eval("/g/Data/model/wavelet/wavelet-500-256-692404-18.pt", transcode=True)
