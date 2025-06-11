import yaml
import numpy as np
import torch
import torch.nn as nn

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MultilabelF1Score,
    MultilabelROC,
    MultilabelAveragePrecision,
    MultilabelPrecisionRecallCurve
)
from tqdm import tqdm

with open('config.yaml', 'r') as file:
    cfg = yaml.safe_load(file)

NUM_CLASSES = 36
metric_collection = MetricCollection([
    MultilabelF1Score(num_labels=NUM_CLASSES, average=None, threshold=0.5).to(cfg['device']),
    MultilabelAveragePrecision(num_labels=NUM_CLASSES, average=None, thresholds=None).to(cfg['device']),
    MultilabelROC(num_labels=NUM_CLASSES, thresholds=None).to(cfg['device']),
    MultilabelPrecisionRecallCurve(num_labels=NUM_CLASSES, thresholds=None).to(cfg['device'])
])

def train(model, data_loader, optimizer, loss_fn, metric_fn, device, scheduler=None):
    model.train()

    loss_total, metric_total = 0.0, 0.0
    num_batches = len(data_loader)

    for batch_idx, (input, target, index) in enumerate(tqdm(data_loader)):
        input, target = input.to(device), target.to(device)

        prediction = model(input)

        loss = loss_fn(prediction, target)
        metric = metric_fn(prediction, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        loss_total += loss.item()
        metric_total += metric.item()

    # print(loss_total)
    loss_total /= num_batches
    metric_total /= num_batches

    return loss_total, metric_total

def validate(model, data_loader, loss_fn, metric_fn, device):
    model.eval()

    loss_total, metric_total = 0.0, 0.0
    num_batches = len(data_loader)

    with torch.no_grad():
        for batch_idx, (input, target, index) in enumerate(tqdm(data_loader)):
            input, target = input.to(device), target.to(device)

            prediction = model(input)

            loss = loss_fn(prediction, target)
            metric = metric_fn(prediction, target)

            loss_total += loss.item()
            metric_total += metric.item()

    loss_total /= num_batches
    metric_total /= num_batches

    return loss_total, metric_total
