import os
import torch
import random
import torchaudio

import pandas as pd

from tools import make_sure_path_exists

# Define audio augmentation operations
# Includes adding noise, changing speed, adjusting volume, reversing, and filtering
def add_noise(signal, noise_level=0.005):
    noise = torch.randn_like(signal) * noise_level

    return signal + noise

def change_speed(signal, target_length=66150):
    rate = random.uniform(0.9, 1.1)

    resampled_signal = torchaudio.transforms.Resample(orig_freq=22050, new_freq=int(22050 * rate))(signal)

    # Get the new length
    new_length = resampled_signal.shape[-1]

    # If it's too long, truncate it to the original length
    if new_length > target_length:
        resampled_signal = resampled_signal[:, :target_length]
    # If it's too short, pad it to the original length
    else:
        pad_size = target_length - new_length
        resampled_signal = torch.nn.functional.pad(resampled_signal, (0, pad_size))

    # print(resampled_signal.shape)

    return resampled_signal

def volume_scaling(signal, min_gain=0.7, max_gain=1.3):
    gain = random.uniform(min_gain, max_gain)
    return signal * gain

def reverse_audio(signal):
    return torch.flip(signal, dims=[-1])

def bandpass_filter(signal, sample_rate=22050, lowcut=300, highcut=3000):
    return torchaudio.functional.bandpass_biquad(signal, sample_rate, lowcut, highcut)

# Randomly select an augmentation operation
def apply_random_augmentation(signal):
    augmentations = [add_noise, change_speed, volume_scaling, reverse_audio, bandpass_filter]
    aug_func = random.choice(augmentations)

    return aug_func(signal)

if __name__ == '__main__':
    ANURASET_DIR = r'C:\Users\nnu-xj-group-Tom\Desktop\anuraset'

    AUDIO_DIR = os.path.join(ANURASET_DIR, 'audio')

    ANNOTATIONS_FILE = os.path.join(ANURASET_DIR, 'metadata.csv')

    save_augmented_audio_folder = 'augmented_audio_and_metadata_file'

    AUGMENTED_AUDIO_DIR = os.path.join(save_augmented_audio_folder, 'augmented_audio')

    make_sure_path_exists(AUGMENTED_AUDIO_DIR)

    metadata_augmented = []
    metadata_test = []

    df = pd.read_csv(ANNOTATIONS_FILE)

    df = df.drop(columns=['SCIFUS', 'LEPELE', 'RHISCI', 'SCINAS', 'LEPFLA', 'SCIRIZ'])

    train_df = df[df['subset'] == 'train']
    test_df = df[df['subset'] == 'test']

    species_counts = train_df.iloc[:, 8:].sum()
    species_need_to_be_augmented = species_counts[species_counts < 1000]
    # print(species_need_to_be_augmented)

    for _, row in test_df.iterrows():
        fname = row['fname']
        start_second = row['min_t']
        final_second = row['max_t']
        path = os.path.join(AUDIO_DIR, fname.split('_')[0],
                            f"{fname}_{start_second}_{final_second}.wav")
        label = row.iloc[8:].tolist()

        metadata_test.append([path] + label)

    metadata_test_df = pd.DataFrame(metadata_test,
                                             columns=['path',
                                                      'SPHSUR', 'BOABIS', 'SCIPER', 'DENNAH', 'LEPLAT', 'RHIICT',
                                                      'BOALEP', 'BOAFAB', 'PHYCUV', 'DENMIN', 'ELABIC', 'BOAPRA',
                                                      'DENCRU', 'BOALUN', 'BOAALB', 'PHYMAR', 'PITAZU', 'PHYSAU',
                                                      'LEPFUS', 'DENNAN', 'PHYALB', 'LEPLAB', 'BOARAN', 'SCIFUV',
                                                      'AMEPIC', 'LEPPOD', 'ADEDIP', 'ELAMAT', 'PHYNAT', 'LEPNOT',
                                                      'ADEMAR', 'BOAALM', 'PHYDIS', 'RHIORN', 'DENELE', 'SCIALT'])
    metadata_test_df.to_csv(os.path.join(save_augmented_audio_folder, 'metadata_test.csv'), index=False)

    print('metadata_test.csv has created successfully.')

    for _, row in train_df.iterrows():
        fname = row['fname']
        start_second = row['min_t']
        final_second = row['max_t']
        original_label = row.iloc[8:].tolist()
        original_folder = fname.split('_')[0]
        original_path = os.path.join(AUDIO_DIR, original_folder, f"{fname}_{start_second}_{final_second}.wav")
        augmented_saved_folder = os.path.join(AUGMENTED_AUDIO_DIR, original_folder)
        make_sure_path_exists(augmented_saved_folder)
        augmented_saved_path = os.path.join(augmented_saved_folder, f"{fname}_{start_second}_{final_second}.wav")

        metadata_augmented.append([augmented_saved_path] + original_label)

        signal, sr = torchaudio.load(original_path)
        torchaudio.save(augmented_saved_path, signal, sr)

        augment_count = 0
        for species, count in species_need_to_be_augmented.items():
            if row[species]:
                if count < 200:
                    augment_count = max(augment_count, 300)
                elif count < 800:
                    augment_count = max(augment_count, 5)
                elif count < 1000:
                    augment_count = max(augment_count, 3)

        for i in range(1, augment_count + 1):
            augmented_signal = apply_random_augmentation(signal)
            augmented_fname = f"{fname}_{start_second}_{final_second}_aug{i}.wav"
            augmented_path = os.path.join(augmented_saved_folder, augmented_fname)
            torchaudio.save(augmented_path, augmented_signal, sr)
            metadata_augmented.append([augmented_path] + original_label)

    metadata_augmented_df = pd.DataFrame(metadata_augmented,
                                         columns=['path',
                                                  'SPHSUR', 'BOABIS', 'SCIPER', 'DENNAH', 'LEPLAT', 'RHIICT',
                                                  'BOALEP', 'BOAFAB', 'PHYCUV', 'DENMIN', 'ELABIC', 'BOAPRA',
                                                  'DENCRU', 'BOALUN', 'BOAALB', 'PHYMAR', 'PITAZU', 'PHYSAU',
                                                  'LEPFUS', 'DENNAN', 'PHYALB', 'LEPLAB', 'BOARAN', 'SCIFUV',
                                                  'AMEPIC', 'LEPPOD', 'ADEDIP', 'ELAMAT', 'PHYNAT', 'LEPNOT',
                                                  'ADEMAR', 'BOAALM', 'PHYDIS', 'RHIORN', 'DENELE', 'SCIALT'])
    metadata_augmented_df.to_csv(os.path.join(save_augmented_audio_folder, 'metadata_augmented.csv'), index=False)

    print('metadata_augmented.csv has created successfully.')
