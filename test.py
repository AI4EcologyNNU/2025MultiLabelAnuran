import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import pandas as pd
import random
import re
import matplotlib.pyplot as plt
from torch import nn

class MinMaxNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x - x.min()) / (x.max() - x.min())


# 获取频谱图并进行 Mixture Masking 的函数
# def get_mel_spectrogram_db(file_path, train):
#     t_mask_prob = 0.1
#     f_mask_prob = 0.1
#
#     # 读取音频文件
#     waveform, sr = torchaudio.load(file_path)
#
#     # 预处理
#     resamp = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)
#     mel_spectrogram = T.MelSpectrogram(hop_length=128, n_fft=512, n_mels=128)
#     min_max_norm = MinMaxNorm()
#
#     transform = torch.nn.Sequential(resamp, mel_spectrogram, T.AmplitudeToDB())
#
#     # 计算 Mel 频谱图
#     mel_spectrogram_db = transform(waveform)
#
#     if train:
#         mel_spectrogram_db = min_max_norm(mel_spectrogram_db)
#         _, mel_bins, time_steps = mel_spectrogram_db.size()
#
#         try:
#             adjacent_file_path = get_adjacent_file_path(file_path)
#             if not os.path.exists(adjacent_file_path):
#                 return mel_spectrogram_db  # 如果相邻文件不存在，直接返回
#
#             adjacent_waveform, _ = torchaudio.load(adjacent_file_path)
#             mel_spectrogram_db_adjacent = transform(adjacent_waveform)
#             mel_spectrogram_db_adjacent = min_max_norm(mel_spectrogram_db_adjacent)
#
#             # 获取随机时间和频率范围
#             t = int(t_mask_prob * time_steps)
#             t0 = torch.randint(0, time_steps - t, (1,))
#             f = int(f_mask_prob * mel_bins)
#             f0 = torch.randint(0, mel_bins - f, (1,))
#
#             # Mixture Masking
#             mel_spectrogram_db = mixture_masking(mel_spectrogram_db.numpy(), mel_spectrogram_db_adjacent.numpy(), t0, t,
#                                                  f0, f)
#             mel_spectrogram_db = torch.tensor(mel_spectrogram_db)
#         except Exception as e:
#             print(f"Skipping Mixture Masking due to error: {e}")
#             return mel_spectrogram_db
#
#     return mel_spectrogram_db

# 获取相邻文件路径的函数
def get_adjacent_file_path(file_path):
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
#
#
# # Mixture Masking 函数
# def mixture_masking(x, y, t0, t, f0, f):
#     x_augmented = np.copy(x)
#     print(x_augmented.shape)
#     x_augmented[:, :, t0:t0 + t] = 0.5 * (x[:, :, t0:t0 + t] + y[:, :, t0:t0 + t])  # Time masking with mixture
#     x_augmented[:, f0:f0 + f, :] = 0.5 * (x[:, f0:f0 + f, :] + y[:, f0:f0 + f, :])  # Frequency masking with mixture
#     return x_augmented
#
def plot_spectrogram(mel_spectrogram):
    mel_spectrogram_db = mel_spectrogram[0].numpy()
    plt.figure(figsize=(10, 6))
    plt.imshow(mel_spectrogram_db, aspect='auto', cmap='inferno', origin='lower')
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram (dB)")
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    plt.savefig("next mel spectrogram", dpi=300)

# 统计train和test中物种的个数 --- train和test
# if __name__ == "__main__":
#     import pandas as pd
#
#     # 读取 CSV 文件
#     df = pd.read_csv(r'C:\Users\nnu-xj-group-Tom\Desktop\anuraset\metadata.csv')
#
#     # 只选取 subset 为 train 的数据
#     train_df = df[df['subset'] == 'test']
#
#     # 假设物种列从第9列开始，统计每种物种在 train 中出现的次数（即值为1的个数）
#     species_counts = train_df.iloc[:, 8:].sum().sort_values(ascending=False)
#
#     # 打印每个物种和出现次数
#     for species, count in species_counts.items():
#         print(f"{species}: {int(count)}")


# 绘图，绘制train
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     species_counts = {
#         'SPHSUR': 13258, 'BOABIS': 10888, 'BOAFAB': 6438, 'DENMIN': 6070,
#         'LEPPOD': 6032, 'PHYALB': 5374, 'LEPLAT': 5244, 'PITAZU': 4873,
#         'PHYCUV': 4240, 'DENNAN': 3801, 'SCIPER': 3791, 'BOAALB': 3704,
#         'SCIFUV': 2734, 'BOALUN': 2060, 'BOAALM': 1601, 'PHYSAU': 1479,
#         'BOARAN': 1339, 'LEPLAB': 1329, 'LEPFUS': 1232, 'ELABIC': 1214,
#         'LEPNOT': 1062, 'PHYDIS': 897, 'BOALEP': 846, 'DENCRU': 602,
#         'ADEMAR': 520, 'BOAPRA': 480, 'DENNAH': 467, 'PHYNAT': 410,
#         'ELAMAT': 395, 'ADEDIP': 390, 'RHIICT': 310, 'SCIALT': 232,
#         'PHYMAR': 200, 'DENELE': 149, 'SCIRIZ': 73, 'AMEPIC': 68,
#         'LEPELE': 34, 'RHIORN': 21, 'RHISCI': 11, 'LEPFLA': 7,
#         'SCINAS': 0, 'SCIFUS': 0
#     }
#
#     species = list(species_counts.keys())
#     counts = list(species_counts.values())
#
#     plt.figure(figsize=(20, 8))
#     bars = plt.bar(species, counts, color='#FFA500')  # 橘黄色柱状图
#
#     plt.xticks(rotation=90, fontsize=10, fontweight='bold')
#     plt.xlabel('42 species of anurans', fontsize=12)
#     plt.ylabel("Number of instances (Training set)", fontsize=12)
#
#     # 加入每个柱子上的数字，包括 0 的也显示出来
#     for bar, count in zip(bars, counts):
#         height = bar.get_height()
#         plt.text(
#             bar.get_x() + bar.get_width() / 2.0,
#             height + (50 if count > 0 else 5),  # 为了 0 的显示不被柱子遮挡
#             str(count),
#             ha='center', va='bottom',
#             fontsize=8, color='black'
#         )
#
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.draw()
#     plt.savefig("species_instance_counts_train.png", dpi=300)
#     # plt.show()  # 可选

# 绘图，绘制test
# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#
#     species_counts = {
#         'SPHSUR': 6810, 'BOABIS': 5636, 'BOAFAB': 3355, 'DENMIN': 3254,
#         'LEPPOD': 3120, 'PHYALB': 2695, 'LEPLAT': 2584, 'PITAZU': 2571,
#         'DENNAN': 2007, 'PHYCUV': 1986, 'SCIPER': 1976, 'BOAALB': 1904,
#         'SCIFUV': 1395, 'BOALUN': 1098, 'PHYSAU': 618, 'BOAALM': 593,
#         'BOARAN': 575, 'LEPNOT': 535, 'LEPLAB': 518, 'LEPFUS': 505,
#         'ELABIC': 491, 'BOALEP': 426, 'PHYDIS': 324, 'ELAMAT': 293,
#         'DENCRU': 270, 'DENNAH': 267, 'BOAPRA': 247, 'ADEMAR': 224,
#         'ADEDIP': 223, 'RHIICT': 180, 'PHYMAR': 117, 'SCIFUS': 116,
#         'AMEPIC': 78, 'PHYNAT': 65, 'RHIORN': 43, 'SCINAS': 38,
#         'DENELE': 17, 'SCIALT': 10, 'LEPFLA': 0, 'SCIRIZ': 0,
#         'RHISCI': 0, 'LEPELE': 0
#     }
#
#     species = list(species_counts.keys())
#     counts = list(species_counts.values())
#
#     plt.figure(figsize=(20, 8))
#     bars = plt.bar(species, counts, color='#FFA500')  # 橘黄色柱状图
#
#     plt.xticks(rotation=90, fontsize=10, fontweight='bold')
#     plt.xlabel('42 species of anurans', fontsize=12)
#     plt.ylabel("Number of instances (Test set)", fontsize=12)
#
#     # 加入每个柱子上的数字，包括 0 的也显示出来
#     for bar, count in zip(bars, counts):
#         height = bar.get_height()
#         plt.text(
#             bar.get_x() + bar.get_width() / 2.0,
#             height + (50 if count > 0 else 5),  # 为了 0 的显示不被柱子遮挡
#             str(count),
#             ha='center', va='bottom',
#             fontsize=8, color='black'
#         )
#
#     plt.grid(axis='y', linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     plt.draw()
#     plt.savefig("species_instance_counts_test.png", dpi=300)
#     # plt.show()  # 可选
#
# def plot_3d_spectrogram(spec, title='3D Spectrogram', elev=30, azim=45):
#     """
#     spec: torch.Tensor or np.ndarray, shape [freq, time]
#     """
#     if isinstance(spec, torch.Tensor):
#         spec = spec.cpu().detach().numpy()
#
#     freq_bins, time_steps = spec.shape
#
#     X = np.arange(time_steps)
#     Y = np.arange(freq_bins)
#     X, Y = np.meshgrid(X, Y)
#     Z = spec
#
#     fig = plt.figure(figsize=(10, 6))
#     ax = fig.add_subplot(111, projection='3d')
#
#     surf = ax.plot_surface(X, Y, Z, cmap='viridis')
#
#     ax.set_xlabel('Time')
#     ax.set_ylabel('Frequency')
#     ax.set_zlabel('Amplitude')
#     ax.set_title(title)
#
#     ax.view_init(elev=elev, azim=azim)  # 可调整角度
#     fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
#     plt.tight_layout()
#     plt.savefig("Original spectrogram.png", dpi=300)

# def get_mel_spectrogram_db(file_path, train):
#     t_mask_prob = 0.1
#     f_mask_prob = 0.1
#
#     waveform, sr = torchaudio.load(file_path)
#     # print(sr)  # torch.Size([1, 66150])
#
#     resamp = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)
#     mel_spectrogram = T.MelSpectrogram(hop_length=128, n_fft=512, n_mels=128)
#     min_max_norm = MinMaxNorm()
#
#     transform = torch.nn.Sequential(resamp,
#                                     mel_spectrogram,
#                                     T.AmplitudeToDB(),
#                                     min_max_norm)
#
#     mel_spectrogram_db = transform(waveform)
#     # plot_spectrogram(mel_spectrogram_db)
#     # plot_3d_spectrogram(mel_spectrogram_db.squeeze(), title='Original spectrogram')
#
#     if train:
#         # print(mel_spectrogram_db.shape)  # torch.Size([1, 128, 376])
#         _, mel_bins, time_steps = mel_spectrogram_db.size()
#
#         adjacent_file_path = get_adjacent_file_path(file_path)
#         if not os.path.exists(adjacent_file_path):
#             return mel_spectrogram_db
#         adjacent_waveform, _ = torchaudio.load(adjacent_file_path)
#         mel_spectrogram_db_adjacent = transform(adjacent_waveform)
#
#         t = int(t_mask_prob * time_steps)
#         t0 = torch.randint(0, time_steps - t, (1,))
#
#         f = int(f_mask_prob * mel_bins)
#         f0 = torch.randint(0, mel_bins - f, (1,))
#
#         # Mixture Masking
#         mel_spectrogram_db = mixture_masking(mel_spectrogram_db.numpy(), mel_spectrogram_db_adjacent.numpy(), t0, t, f0, f)
#
#         # back to tensor form
#         mel_spectrogram_db = torch.tensor(mel_spectrogram_db)
#
#         # plot_spectrogram(mel_spectrogram_db)
#         plot_3d_spectrogram(mel_spectrogram_db.squeeze(), title='Spectrogram after MM')
#
#     return mel_spectrogram_db

# 绘制MM后的频谱图，3D版本
# if __name__ == "__main__":
#     file_path = r"C:\Users\nnu-xj-group-Tom\Desktop\anuraset-Tom\augmented_audio_and_metadata_file\augmented_audio\INCT17\INCT17_20200123_211500_34_37.wav"
#
#     get_mel_spectrogram_db(file_path, train=True)

# def add_noise(signal, noise_level=0.005):
#     noise = torch.randn_like(signal) * noise_level
#
#     return signal + noise
#
# def change_speed(signal, target_length=66150):
#     rate = random.uniform(0.9, 1.1)
#
#     resampled_signal = torchaudio.transforms.Resample(orig_freq=22050, new_freq=int(22050 * rate))(signal)
#
#     # Get the new length
#     new_length = resampled_signal.shape[-1]
#
#     # If it's too long, truncate it to the original length
#     if new_length > target_length:
#         resampled_signal = resampled_signal[:, :target_length]
#     # If it's too short, pad it to the original length
#     else:
#         pad_size = target_length - new_length
#         resampled_signal = torch.nn.functional.pad(resampled_signal, (0, pad_size))
#
#     # print(resampled_signal.shape)
#
#     return resampled_signal
#
# def volume_scaling(signal, min_gain=0.7, max_gain=1.3):
#     gain = random.uniform(min_gain, max_gain)
#     return signal * gain
#
# def reverse_audio(signal):
#     return torch.flip(signal, dims=[-1])
#
# def bandpass_filter(signal, sample_rate=22050, lowcut=300, highcut=3000):
#     return torchaudio.functional.bandpass_biquad(signal, sample_rate, lowcut, highcut)
#
# # Randomly select an augmentation operation
# def apply_random_augmentation(signal):
#     augmentations = [add_noise, change_speed, volume_scaling, reverse_audio, bandpass_filter]
#     aug_func = random.choice(augmentations)
#
#     return aug_func(signal)
#
# if __name__ == "__main__":
#     file_path = r'C:\Users\nnu-xj-group-Tom\Desktop\anuraset\audio\INCT20955\INCT20955_20191031_000000_36_39.wav'
#     waveform, sample_rate = torchaudio.load(file_path)
#
#     plt.figure(figsize=(10, 4))
#     plt.plot(waveform[0].numpy(), color='steelblue')
#     # plt.title('Waveform')
#     # plt.xlabel('Sample Index')
#     plt.ylabel('Amplitude')
#     plt.tight_layout()
#     plt.savefig('original waveform.png', dpi=300)

    # waveform = add_noise(waveform)
    # plt.figure(figsize=(10, 4))
    # plt.plot(waveform[0].numpy(), color='steelblue')
    # # plt.title('Waveform')
    # # plt.xlabel('Sample Index')
    # plt.ylabel('Amplitude')
    # plt.tight_layout()
    # plt.savefig('noise added.png', dpi=300)
    #
    # waveform = change_speed(waveform)
    # plt.figure(figsize=(10, 4))
    # plt.plot(waveform[0].numpy(), color='steelblue')
    # # plt.title('Waveform')
    # # plt.xlabel('Sample Index')
    # plt.ylabel('Amplitude')
    # plt.tight_layout()
    # plt.savefig('speed change.png', dpi=300)
    #
    # waveform = volume_scaling(waveform)
    # plt.figure(figsize=(10, 4))
    # plt.plot(waveform[0].numpy(), color='steelblue')
    # # plt.title('Waveform')
    # # plt.xlabel('Sample Index')
    # plt.ylabel('Amplitude')
    # plt.tight_layout()
    # plt.savefig('volume scaling.png', dpi=300)
    #
    # waveform = reverse_audio(waveform)
    # plt.figure(figsize=(10, 4))
    # plt.plot(waveform[0].numpy(), color='steelblue')
    # # plt.title('Waveform')
    # # plt.xlabel('Sample Index')
    # plt.ylabel('Amplitude')
    # plt.tight_layout()
    # plt.savefig('reverse audio.png', dpi=300)
    #
    # waveform = bandpass_filter(waveform)
    # plt.figure(figsize=(10, 4))
    # plt.plot(waveform[0].numpy(), color='steelblue')
    # # plt.title('Waveform')
    # # plt.xlabel('Sample Index')
    # plt.ylabel('Amplitude')
    # plt.tight_layout()
    # plt.savefig('bandpass filter.png', dpi=300)

if __name__ == "__main__":
    file_path = r"C:\Users\nnu-xj-group-Tom\Desktop\anuraset-Tom\augmented_audio_and_metadata_file\augmented_audio\INCT17\INCT17_20191126_040000_46_49.wav"
    adjecent_file_path = get_adjacent_file_path(file_path)

    waveform, _ = torchaudio.load(file_path)
    adjecent_waveform, _ = torchaudio.load(adjecent_file_path)
    # print(sr)  # torch.Size([1, 66150])

    resamp = torchaudio.transforms.Resample(orig_freq=22050, new_freq=16000)
    mel_spectrogram = T.MelSpectrogram(hop_length=128, n_fft=512, n_mels=128)
    min_max_norm = MinMaxNorm()

    transform = torch.nn.Sequential(resamp,
                                    mel_spectrogram,
                                    T.AmplitudeToDB(),
                                    min_max_norm)

    mel_spectrogram_db = transform(waveform)
    adjecent_mel_spectrogram_db = transform(adjecent_waveform)

    # plot_spectrogram(mel_spectrogram_db)
    plot_spectrogram(adjecent_mel_spectrogram_db)
