"""Evaluate a checkpoint on full images, accumulating a dataset confusion matrix."""

import argparse
import json
from pathlib import Path

import torch

from damie_unet.data import val_dataset
from damie_unet.engine import DEVICE_CHOICES, evaluate, select_device
from damie_unet.model import UNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--device', choices=DEVICE_CHOICES, default=None)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    device = args.device or select_device()
    config = checkpoint['config']
    model = UNet(config['width']).to(device)
    model.load_state_dict(checkpoint['model'])
    dataset = val_dataset(config['data'], config['crop'], config['val_size'], config['seed'])
    print(json.dumps(evaluate(model, dataset, device), indent=2))


if __name__ == '__main__':
    main()
