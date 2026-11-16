"""Train a U-Net on synthetic scenes or user-supplied image/mask folders."""

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset, DataLoader

from damie_unet.data import FolderDataset, SyntheticPonds, val_dataset
from damie_unet.engine import DEVICE_CHOICES, evaluate, select_device
from damie_unet.model import UNet


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='synthetic')
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--crop', type=int, default=256)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--repeat', type=int, default=1, help='random crops per image per epoch')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', type=Path, default=Path('runs/synthetic'))
    parser.add_argument('--device', choices=DEVICE_CHOICES, default=None)
    parser.add_argument('--train-size', type=int, default=64, help='synthetic scene count')
    parser.add_argument('--val-size', type=int, default=16, help='synthetic scene count')
    args = parser.parse_args()
    if min(args.epochs, args.crop, args.batch, args.width, args.repeat) < 1 or args.lr <= 0:
        parser.error('epochs, crop, batch, width, repeat and lr must be positive')
    start = time.perf_counter()
    torch.manual_seed(args.seed)
    device = args.device or select_device()
    if args.data == 'synthetic':
        size = args.crop + args.crop // 8
        train = SyntheticPonds(args.train_size, size, args.crop, seed=args.seed)
    else:
        train = FolderDataset(Path(args.data) / 'train', crop=args.crop)
    val = val_dataset(args.data, args.crop, args.val_size, args.seed)
    train = ConcatDataset([train] * args.repeat)
    loader = DataLoader(train, batch_size=args.batch, shuffle=True)
    model = UNet(args.width).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()
    args.out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config['out'] = str(args.out)
    config['device'] = device
    history, best = [], None
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        for images, masks in loader:
            images, masks = images.to(device).float().div_(255), masks.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(images)
        metrics = evaluate(model, val, device)
        record = {'epoch': epoch, 'loss': loss_sum.item() / len(train), **metrics}
        history.append(record)
        if best is None or metrics['miou'] > best['miou']:
            best = record
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'metrics': metrics, 'config': config},
                       args.out / 'best.pt')
        print(f'epoch={epoch} device={device} loss={record["loss"]:.4f} '
              f'val_miou={metrics["miou"]:.4f} val_pond_iou={metrics["iou"][1]:.4f} val_f1={metrics["f1_pond"]:.4f}', flush=True)
    result = {'config': config, 'best': best, 'history': history,
              'wall_seconds': time.perf_counter() - start}
    (args.out / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
