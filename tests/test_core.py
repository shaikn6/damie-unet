import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from damie_unet.data import FolderDataset, SyntheticPonds
from damie_unet.metrics import ConfusionMatrix
from damie_unet.model import UNet

torch.set_num_threads(2)


def write_pair(tmp_path, image, mask, suffix):
    (tmp_path / 'images').mkdir(exist_ok=True)
    (tmp_path / 'masks').mkdir(exist_ok=True)
    Image.fromarray(image).save(tmp_path / 'images' / f'a{suffix}')
    Image.fromarray(mask).save(tmp_path / 'masks/a.png')


@pytest.mark.parametrize('shape', [(1, 3, 64, 64), (1, 3, 65, 79)])
def test_model_shape(shape):
    with torch.no_grad():
        assert UNet(width=8)(torch.rand(shape)).shape == (shape[0], 2, *shape[2:])


@pytest.mark.parametrize('matrix,iou,f1', [
    ([[5, 0], [0, 3]], [1, 1], 1),
    ([[0, 5], [3, 0]], [0, 0], 0),
    ([[3, 1], [2, 4]], [0.5, 4 / 7], 8 / 11),
])
def test_metrics(matrix, iou, f1):
    metric = ConfusionMatrix()
    # Rows are true labels, columns predictions; update in separate chunks.
    for truth in range(2):
        for pred in range(2):
            n = matrix[truth][pred]
            metric.update(torch.full((n,), pred), torch.full((n,), truth))
    result = metric.compute()
    assert result['iou'] == pytest.approx(iou)
    assert result['miou'] == pytest.approx(sum(iou) / 2)
    assert result['f1_pond'] == pytest.approx(f1)


@pytest.mark.parametrize('training', [True, False])
def test_synthetic(training):
    ds = SyntheticPonds(length=3, size=288, crop=256, training=training, seed=7)
    image, mask = ds[0]
    size = 256 if training else 288
    assert image.shape == (3, size, size)
    assert mask.shape == (size, size)
    assert image.dtype == torch.uint8 and mask.dtype == torch.int64
    assert set(mask.unique().tolist()) == {0, 1}
    assert 0 <= image.min() <= image.max() <= 255
    if not training:
        assert torch.equal(image, ds[0][0])


@pytest.mark.parametrize('training', [True, False])
def test_folder(tmp_path, training):
    mask = np.zeros((270, 290), dtype=np.uint8)
    mask[:, 145:] = 255
    write_pair(tmp_path, np.repeat(mask[..., None], 3, axis=2), mask, '.png')
    image, target = FolderDataset(tmp_path, training=training)[0]
    assert image.shape == (3, 256, 256) if training else image.shape == (3, 270, 290)
    assert image.dtype == torch.uint8
    assert target.dtype == torch.long
    assert torch.equal((image[0] > 0).long(), target)


def test_smoke_train(tmp_path):
    start = time.perf_counter()
    subprocess.run([
        sys.executable, 'train.py', '--data', 'synthetic', '--width', '8',
        '--epochs', '2', '--crop', '64', '--batch', '4', '--device', 'cpu',
        '--train-size', '32', '--val-size', '8', '--out', str(tmp_path),
    ], check=True, cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
    result = json.loads((tmp_path / 'metrics.json').read_text())
    assert result['best']['miou'] > 0.5
    assert time.perf_counter() - start < 60
    assert (tmp_path / 'best.pt').exists()


def test_standard_parameter_count():
    model = UNet()
    assert 30_000_000 < sum(p.numel() for p in model.parameters()) < 32_000_000


def test_absent_class_metrics():
    metric = ConfusionMatrix()
    metric.update(torch.zeros(4), torch.zeros(4))
    result = metric.compute()
    assert result['iou'] == [1.0, 0.0]
    assert result['miou'] == 0.5
    assert result['f1_pond'] == 0.0


def test_folder_rejects_missing_mask(tmp_path):
    (tmp_path / 'images').mkdir()
    Image.new('RGB', (16, 16)).save(tmp_path / 'images/a.jpg')
    with pytest.raises(FileNotFoundError):
        FolderDataset(tmp_path)


def test_folder_pads_small_images(tmp_path):
    write_pair(tmp_path, np.full((30, 20, 3), 255, dtype=np.uint8),
               np.ones((30, 20), dtype=np.uint8), '.jpg')
    image, target = FolderDataset(tmp_path)[0]
    assert image.shape == (3, 256, 256)
    assert target.shape == (256, 256)
    assert torch.all(target == 1)


def test_precision_recall_accuracy():
    # truth ice: 6 right, 1 called pond; truth pond: 1 missed, 2 right.
    metric = ConfusionMatrix()
    metric.update(torch.tensor([0] * 6 + [1] + [0] + [1] * 2), torch.tensor([0] * 7 + [1] * 3))
    result = metric.compute()
    assert result['precision_pond'] == pytest.approx(2 / 3)
    assert result['recall_pond'] == pytest.approx(2 / 3)
    assert result['accuracy'] == pytest.approx(0.8)
    assert result['f1_macro'] == pytest.approx((2 / 3 + 12 / 14) / 2)
