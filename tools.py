import os
import re
import math
import yaml
import errno
import torch
import torchaudio

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torchaudio.transforms as T


from torch import nn
# from pykalman import KalmanFilter

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise

def get_file_paths_and_labels(annotations_file):
    df = pd.read_csv(annotations_file)

    paths = df['path'].tolist()
    labels = [torch.tensor(row, dtype=torch.float32) for row in df.iloc[:, -36:].values]

    return paths, labels

def plot_spectrogram(mel_spectrogram):
    mel_spectrogram_db = mel_spectrogram[0].numpy()
    plt.figure(figsize=(10, 6))
    plt.imshow(mel_spectrogram_db, aspect='auto', cmap='inferno', origin='lower')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram (dB)")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.show()

def remove_augmentation_suffix(file_path):
    return re.sub(r"_aug\d+", "", file_path)

def get_adjacent_file_path(file_path):
    file_path = remove_augmentation_suffix(file_path)

    pattern = r'_(\d+)_(\d+).wav'
    match = re.search(pattern, file_path)

    if match:
        num1 = int(match.group(1))
        num2 = int(match.group(2))

        if num1 == 57 and num2 == 60:
            new_num1 = num1 - 1
            new_num2 = num2 - 1
        else:
            new_num1 = num1 + 1
            new_num2 = num2 + 1

        new_file_path = file_path.replace(f'_{num1}_{num2}.wav', f'_{new_num1}_{new_num2}.wav')

        return new_file_path
    else:
        raise ValueError(f"文件名格式不正确: {file_path}")

def mixture_masking(x, y, t0, t, f0, f):
    x_augmented = np.copy(x)
    # print(x_augmented.shape)
    x_augmented[:, :, t0:t0 + t] = 0.5 * (x[:, :, t0:t0 + t] + y[:, :, t0:t0 + t])  # Time masking with mixture
    x_augmented[:, f0:f0 + f, :] = 0.5 * (x[:, f0:f0 + f, :] + y[:, f0:f0 + f, :])  # Frequency masking with mixture
    return x_augmented

def get_mel_spectrogram_db(file_path, train):
    t_mask_prob = 0.1
    f_mask_prob = 0.1

    waveform, sr = torchaudio.load(file_path)
    # print(sr)  # torch.Size([1, 66150])

    resamp = torchaudio.transforms.Resample(orig_freq=22050, new_freq=cfg['sample_rate'])
    mel_spectrogram = T.MelSpectrogram(hop_length=128, n_fft=512, n_mels=128)
    min_max_norm = MinMaxNorm()

    transform = torch.nn.Sequential(resamp,
                                    mel_spectrogram,
                                    T.AmplitudeToDB(),
                                    min_max_norm)

    mel_spectrogram_db = transform(waveform)
    # plot_spectrogram(mel_spectrogram_db)

    if train:
        # print(mel_spectrogram_db.shape)  # torch.Size([1, 128, 376])
        _, mel_bins, time_steps = mel_spectrogram_db.size()

        adjacent_file_path = get_adjacent_file_path(file_path)
        if not os.path.exists(adjacent_file_path):
            return mel_spectrogram_db
        adjacent_waveform, _ = torchaudio.load(adjacent_file_path)
        mel_spectrogram_db_adjacent = transform(adjacent_waveform)

        t = int(t_mask_prob * time_steps)
        t0 = torch.randint(0, time_steps - t, (1,))

        f = int(f_mask_prob * mel_bins)
        f0 = torch.randint(0, mel_bins - f, (1,))

        # Mixture Masking
        mel_spectrogram_db = mixture_masking(mel_spectrogram_db.numpy(), mel_spectrogram_db_adjacent.numpy(), t0, t, f0, f)

        # back to tensor form
        mel_spectrogram_db = torch.tensor(mel_spectrogram_db)

        # plot_spectrogram(mel_spectrogram_db)

    return mel_spectrogram_db

def create_data_frame_AnuraSet(audio_list, audio_labels):
    data = {'filename': [], 'label': []}
    df = pd.DataFrame(data=data)

    df['filename'] = np.array(audio_list)
    df['label'] = audio_labels

    return df

def show_loss(train_loss, validation_loss, save_results_path):
    plt.figure()
    plt.plot(train_loss)
    plt.plot(validation_loss)
    plt.title('Train and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_results_path + '/loss.png')
    plt.show()

def show_f1_score(train_f1score, valid_f1score, save_results_path):
    plt.figure()
    plt.plot(train_f1score)
    plt.plot(valid_f1score)
    plt.title('Train and Validation F1-score')
    plt.ylabel('F1-score')
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_results_path + '/F1-score.png')
    plt.show()

class MinMaxNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x - x.min()) / (x.max() - x.min())

class ZLPRLoss(nn.Module):
    def __init__(self):
        super(ZLPRLoss, self).__init__()

    def forward(self, scores, labels):
        """
        :param scores: Tensor of shape (batch_size, num_classes)
        :param labels: Tensor of shape (batch_size, num_classes) with 0/1
        :return: scalar loss value
        """
        # Avoid log(1 + exp(x)) overflow using softplus
        pos_scores = scores[labels == 1]
        neg_scores = scores[labels == 0]

        if pos_scores.numel() > 0:
            pos_loss = F.softplus(-pos_scores).mean()
        else:
            pos_loss = 0.0

        if neg_scores.numel() > 0:
            neg_loss = F.softplus(neg_scores).mean()
        else:
            neg_loss = 0.0

        total_loss = pos_loss + neg_loss

        return total_loss
