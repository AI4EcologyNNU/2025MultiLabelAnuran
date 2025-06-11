import os
import torch
import torchaudio

from tqdm import tqdm
from tools import get_mel_spectrogram_db
from torch.utils.data import Dataset

class AnuraSet(Dataset):
    def __init__(self,
                 df,
                 train=True,):
        self.df = df
        self.data = []

        for index in tqdm(range(len(df))):
            row = df.iloc[index]
            path = row['filename']

            mel_spectrogram_db = get_mel_spectrogram_db(path, train=train)

            self.data.append(mel_spectrogram_db)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.data[index], self.df.iloc[index]['label'], index
