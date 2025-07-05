import os
import dill
import torch
import yaml

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm import Mamba
from einops import rearrange
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.classification import MultilabelF1Score

from train_and_validate import train, validate
from tools import make_sure_path_exists, show_loss, show_f1_score, ZLPRLoss

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

'''
CBAM adaptively adjusts the channel and spatial weights of the feature map to optimize information expression.
'''
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=3):
        super(CBAM, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(channels, channels // reduction, 1, bias=False)
        self.conv2 = nn.Conv2d(channels // reduction, channels, 1, bias=False)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, H, W = x.size()

        # Channel Attention
        channel_att = self.avg_pool(x)
        channel_att = F.relu(self.conv1(channel_att))
        channel_att = self.sigmoid(self.conv2(channel_att))
        x = x * channel_att

        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.sigmoid(self.spatial_conv(spatial_att))

        x = x * spatial_att

        return x

class CNN(nn.Module):
    def __init__(self, num_classes=36):
        super(CNN, self).__init__()

        # CNN module
        self.conv = nn.Sequential(
            # Input size: [batch_size, 1, 128, 376]
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.1), # [batch_size, 32, 64, 188]

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2), # [batch_size, 64, 32, 94]

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.3), # [batch_size, 128, 16, 47]
        )

        # CBAM module
        self.cbam = CBAM(128, reduction=16, kernel_size=3)

        '''
        The RNN module uses a bi-directional LSTM (bi-LSTM), 
        with an output shape of (batch_size, time_steps, 2 * hidden_size). 
        The advantage of the bidirectional LSTM is that it can combine 
        forward and backward information to make better predictions.
        '''
        self.rnn = nn.LSTM(
            input_size=128 * 16,  # Input feature dimension (128 channels * 16 frequency dimensions)
            hidden_size=256,  # LSTM hidden layer dimension
            num_layers=2,  # Number of LSTM layers
            batch_first=True,  # Input shape is [batch_size, seq_len, feature_dim]
            bidirectional=True,  # Bi-directional LSTM
            dropout=0.3,
        )

        # self.mamba = Mamba(
        #     d_model=256 * 2,
        #     d_state=16,
        #     d_conv=4,
        #     expand=2,
        # )

        self.fc = nn.Linear(256 * 2, num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x)
        x = x.permute(0, 3, 1, 2).reshape(x.size(0), x.size(3), -1)
        # x = self.mamba(x) # before LSTM
        x, _ = self.rnn(x)
        # x = self.mamba(x) # after LSTM
        x = x[:, -1, :]
        x = self.fc(x)

        return x

class CNNDualMamba(nn.Module):
    def __init__(self, num_classes=36):
        super(CNNDualMamba, self).__init__()

        self.conv = nn.Sequential(
            # Input shape: [batch_size, 1, 128, 376]
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.1),  # [batch_size, 32, 64, 188]

            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.2),  # [batch_size, 64, 32, 94]

            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.3),  # [batch_size, 128, 16, 47]
        )

        self.cbam = CBAM(128, reduction=16, kernel_size=3)

        # Dual branch Mamba
        self.mamba_time = Mamba(
            d_model=128 * 16,
            d_state=16,
            d_conv=4,
            expand=2,
        )
        self.mamba_freq = Mamba(
            d_model=128 * 47,
            d_state=16,
            d_conv=4,
            expand=2,
        )

        # Dual branch Multi-layer Mamba
        # self.mamba_time = nn.Sequential(
        #     Mamba(d_model=128 * 16, d_state=16, d_conv=4, expand=2),
        #     Mamba(d_model=2048, d_state=16, d_conv=4, expand=2),
        #     Mamba(d_model=2048, d_state=16, d_conv=4, expand=2),
        # )
        # self.mamba_freq = nn.Sequential(
        #     Mamba(d_model=128 * 47, d_state=16, d_conv=4, expand=2),
        #     Mamba(d_model=6016, d_state=16, d_conv=4, expand=2),
        #     Mamba(d_model=6016, d_state=16, d_conv=4, expand=2),
        # )

        # LayerNorm
        self.norm_time = nn.LayerNorm(128 * 16)
        self.norm_freq = nn.LayerNorm(128 * 47)

        # First perform dimensionality reduction;
        # otherwise, directly reducing from 8024 to 256 parameters is too drastic
        self.reduce = nn.Linear(8064, 1024)

        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        self.fc.apply(self._init_weights)

    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x)

        b, c, h, w = x.shape

        xt = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        xt = self.mamba_time(xt)
        xt = self.norm_time(xt)
        xt = xt.mean(dim=1)

        xf = x.permute(0, 2, 1, 3).reshape(b, h, c * w)
        xf = self.mamba_freq(xf)
        xf = self.norm_freq(xf)
        xf = xf.mean(dim=1)

        x = torch.cat([xt, xf], dim=-1)
        x = self.reduce(x) # 8064 -> 1024
        x = self.fc(x) # 1024 -> num_classes

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

if __name__ == '__main__':
    save_dataloader_folder = f'dataloader_saved_{cfg["batch_size"]}'
    save_results_folder = 'results_saved'

    NFOLD = cfg['n_folds']

    for fold in range(NFOLD):
        save_dataloader_path = os.path.join(save_dataloader_folder, f'fold_{fold + 1}')

        with open(os.path.join(save_dataloader_path, 'training_loader.pkl'), 'rb') as f:
            training_loader = dill.load(f)

        with open(os.path.join(save_dataloader_path, 'validation_loader.pkl'), 'rb') as f:
            validation_loader = dill.load(f)

        best_validation_f1 = 0.0
        training_loss_list, validation_loss_list = [], []
        training_f1_list, validation_f1_list = [], []

        # model = CNN(num_classes=36).to(cfg['device'])
        model = CNNDualMamba(num_classes=36).to(cfg['device'])
        optimizer = AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=1e-4)
        # scheduler = OneCycleLR(optimizer, max_lr=cfg['learning_rate'],
        #                        steps_per_epoch=len(training_loader), epochs=cfg['epochs'])
        loss_fn = ZLPRLoss()
        metric_fn = MultilabelF1Score(num_labels=36).to(cfg['device'])

        save_results_path = os.path.join(save_results_folder, f'fold_{fold + 1}')
        make_sure_path_exists(save_results_path)

        for epoch in range(1, cfg['epochs'] + 1):
            print(f'fold {fold + 1} epoch {epoch}')
            # train
            training_loss, training_f1 = train(model, training_loader, optimizer, loss_fn, metric_fn, cfg['device'])
            # validation
            validation_loss, validation_f1 = validate(model, validation_loader, loss_fn, metric_fn, cfg['device'])

            training_loss_list.append(training_loss)
            validation_loss_list.append(validation_loss)
            training_f1_list.append(training_f1)
            validation_f1_list.append(validation_f1)

            print(f"Epoch {epoch + 1}: Train Loss = {training_loss:.4f}, Train F1 score = {training_f1:.4f}, " 
                  f"Validation Loss = {validation_loss:.4f}, Validation F1 score = {validation_f1:.4f}")

            if validation_f1 > best_validation_f1:
                print(f'Validation F1 score is {validation_f1}. The result has improved')
                torch.save(model.state_dict(), os.path.join(save_results_path, f'model_{fold + 1}.bin'))
                print(f'Model checkpoint saved at ./model_{fold + 1}.bin')
                best_validation_f1 = validation_f1
                
        best_model = CNNDualMamba(num_classes=36).to(cfg['device'])
        best_model.load_state_dict(torch.load(os.path.join(save_results_path, f'model_{fold + 1}.bin')))

        show_loss(np.array(training_loss_list), np.array(validation_loss_list), save_results_path)
        show_f1_score(np.array(training_f1_list), np.array(validation_f1_list), save_results_path)
