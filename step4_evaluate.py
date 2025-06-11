import os
import dill
import yaml
import torch

from torch import nn
from tools import ZLPRLoss
from step3_CNN import CNN, CNNDualMamba
from torchmetrics.classification import MultilabelF1Score

class_mapping = [
        'SPHSUR', 'BOABIS', 'SCIPER', 'DENNAH', 'LEPLAT', 'RHIICT', 'BOALEP',
        'BOAFAB', 'PHYCUV', 'DENMIN', 'ELABIC', 'BOAPRA', 'DENCRU', 'BOALUN',
        'BOAALB', 'PHYMAR', 'PITAZU', 'PHYSAU', 'LEPFUS', 'DENNAN', 'PHYALB',
        'LEPLAB', 'BOARAN', 'SCIFUV', 'AMEPIC', 'LEPPOD', 'ADEDIP',
        'ELAMAT', 'PHYNAT', 'LEPNOT', 'ADEMAR',
        'BOAALM', 'PHYDIS', 'RHIORN', 'DENELE', 'SCIALT'
    ]

freq_species = ['SPHSUR', 'BOABIS', 'BOAFAB', 'LEPPOD']
common_species = ['PITAZU', 'DENMIN', 'PHYCUV', 'LEPLAT', 'PHYALB', 'SCIPER', 'DENNAN', 'BOAALB']
rare_species = ['DENNAH', 'RHIICT', 'BOALEP', 'ELABIC', 'BOAPRA', 'DENCRU', 'BOALUN', 'PHYMAR', 'PHYSAU', 'LEPFUS',
                'LEPLAB', 'BOARAN', 'SCIFUV', 'AMEPIC', 'ADEDIP', 'ELAMAT', 'PHYNAT', 'LEPNOT', 'ADEMAR', 'BOAALM',
                'PHYDIS', 'RHIORN', 'DENELE', 'SCIALT', 'SCINAS', 'SCIRIZ', 'LEPELE', 'RHISCI', 'LEPFLA', 'SCIFUS']

idx_freq = [class_mapping.index(sp) for sp in freq_species if sp in class_mapping]
idx_common = [class_mapping.index(sp) for sp in common_species if sp in class_mapping]
idx_rare = [class_mapping.index(sp) for sp in rare_species if sp in class_mapping]

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

def calculate_metrics(preds, targets, fn_metrics, idx_subset=None):
    f1_scores = fn_metrics(preds, targets.long())

    if idx_subset is not None:
        return f1_scores[idx_subset].mean().item()

    return f1_scores.mean().item()

def evaluate(model, data_loader, loss_fn, metric_fn, device):
    model.eval()

    all_sample_idx, all_predictions = [], []
    loss_total, metric_total = 0.0, 0.0
    sigmoid = nn.Sigmoid()
    num_batches = len(data_loader)

    with torch.no_grad():
        for batch_idx, (input, target, index) in enumerate(data_loader):
            input, target = input.to(device), target.to(device)

            prediction = sigmoid(model(input))

            loss = loss_fn(prediction, target)
            metric = metric_fn(prediction, target).mean()

            loss_total += loss.item()
            metric_total += metric.item()

            all_sample_idx.extend(index.tolist())
            all_predictions.extend(prediction.tolist())

    loss_total /= num_batches
    metric_total /= num_batches

    predictions_tensor = torch.Tensor(all_predictions).to(device)
    targets_tensor = torch.stack([torch.tensor(label) for label in data_loader.dataset.df.iloc[all_sample_idx]['label']]).to(device)

    f1_total = calculate_metrics(predictions_tensor, targets_tensor, metric_fn)
    f1_freq = calculate_metrics(predictions_tensor, targets_tensor, metric_fn, idx_freq)
    f1_common = calculate_metrics(predictions_tensor, targets_tensor, metric_fn, idx_common)
    f1_rare = calculate_metrics(predictions_tensor, targets_tensor, metric_fn, idx_rare)

    return all_sample_idx, all_predictions, f1_total, f1_freq, f1_common, f1_rare

if __name__ == "__main__":
    save_dataloader_folder = f'dataloader_saved_{cfg["batch_size"]}'
    save_results_folder = 'results_saved'

    NFOLD = cfg['n_folds']

    for fold in range(NFOLD):
        save_dataloader_path = os.path.join(save_dataloader_folder, f'fold_{fold + 1}')
        model_checkpoint_path = os.path.join(save_results_folder, f'fold_{fold + 1}', f'model_{fold + 1}.bin')

        with open(os.path.join(save_dataloader_path, 'test_loader.pkl'), 'rb') as f:
            test_loader = dill.load(f)

        model = CNNDualMamba(num_classes=cfg['num_classes'])
        model.load_state_dict(torch.load(model_checkpoint_path))
        model.to(cfg['device'])

        loss_fn = ZLPRLoss()
        metric_fn = MultilabelF1Score(num_labels=cfg['num_classes'], average=None).to(cfg['device'])

        samples, predictions, f1_total, f1_freq, f1_common, f1_rare = evaluate(model, test_loader, loss_fn, metric_fn, cfg['device'])

        print(f"Fold {fold + 1}")
        print(f"Overall F1-score: {f1_total:.4f}")
        print(f"Freq Species F1-score: {f1_freq:.4f}")
        print(f"Common Species F1-score: {f1_common:.4f}")
        print(f"Rare Species F1-score: {f1_rare:.4f}")
