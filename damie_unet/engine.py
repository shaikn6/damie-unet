"""Shared device selection and full-image evaluation."""

import torch
from torch.utils.data import DataLoader

from damie_unet.data import FolderDataset
from damie_unet.metrics import ConfusionMatrix

DEVICE_CHOICES = ('cpu', 'mps', 'cuda')


def select_device():
    if torch.cuda.is_available():
        return 'cuda'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


@torch.no_grad()
def evaluate(model, dataset, device):
    model.eval()
    metric = ConfusionMatrix()
    batch_size = 1 if isinstance(dataset, FolderDataset) else 8
    for images, masks in DataLoader(dataset, batch_size=batch_size):
        predictions = model(images.to(device).float().div_(255)).argmax(1)
        metric.update(predictions, masks.to(device))
    return metric.compute()
