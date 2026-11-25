"""Cached synthetic ponds and paired image/mask folders."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


def tensors(image, mask, training, crop, augment=False):
    if training:
        h, w = mask.shape
        if h < crop or w < crop:
            padding = ((0, max(0, crop - h)), (0, max(0, crop - w)))
            image = np.pad(image, (*padding, (0, 0)), mode='edge')
            mask = np.pad(mask, padding, mode='edge')
        h, w = mask.shape
        y = torch.randint(h - crop + 1, ()).item()
        x = torch.randint(w - crop + 1, ()).item()
        image, mask = image[y:y + crop, x:x + crop], mask[y:y + crop, x:x + crop]
        if augment:
            # Aerial scenes have no fixed up direction: random flip and 90-degree rotation.
            turns, flip = torch.randint(4, ()).item(), torch.randint(2, ()).item()
            image, mask = np.rot90(image, turns), np.rot90(mask, turns)
            if flip:
                image, mask = image[:, ::-1], mask[:, ::-1]
    return (torch.from_numpy(np.array(image, dtype=np.uint8).transpose(2, 0, 1).copy()),
            torch.from_numpy(np.array(mask, dtype=np.int64)))


class SyntheticPonds(Dataset):
    """Each index has a fixed seeded scene; training draws fresh random crops."""

    def __init__(self, length=64, size=288, crop=256, training=True, seed=0):
        self.crop, self.training = crop, training
        self.scenes = [self._scene(size, seed + index) for index in range(length)]

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, index):
        image, mask = self.scenes[index]
        return tensors(image, mask, self.training, self.crop)

    @staticmethod
    def _scene(size, seed):
        rng = np.random.default_rng(seed)
        mask_image = Image.new('L', (size, size))
        draw = ImageDraw.Draw(mask_image)
        for _ in range(rng.integers(5, 11)):
            cx, cy = rng.uniform(0, size, 2)
            rx, ry = rng.uniform(size * 0.08, size * 0.24, 2)
            angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
            radii = rng.uniform(0.75, 1.25, len(angles))
            points = list(zip(cx + rx * radii * np.cos(angles),
                              cy + ry * radii * np.sin(angles)))
            draw.polygon(points, fill=1)
        mask = np.asarray(mask_image)
        ice = rng.uniform([205, 215, 220], [240, 245, 250])
        pond = rng.uniform([25, 65, 85], [65, 115, 145])
        image = np.where(mask[..., None] == 1, pond, ice)
        image = np.clip(image + rng.normal(0, 7, image.shape), 0, 255).astype(np.uint8)
        return image, mask


class FolderDataset(Dataset):
    """Match image stems to PNG masks; accept mask values {0,1} or {0,255}."""

    def __init__(self, root, crop=256, training=True, augment=False):
        root = Path(root)
        self.images = sorted(p for p in (root / 'images').glob('*')
                             if p.suffix.lower() in {'.png', '.jpg', '.jpeg'})
        if not self.images:
            raise ValueError(f'No images found in {root / "images"}')
        self.masks = [root / 'masks' / f'{p.stem}.png' for p in self.images]
        if len({p.stem for p in self.images}) != len(self.images):
            raise ValueError('Image stems must be unique')
        for path in self.masks:
            if not path.is_file():
                raise FileNotFoundError(path)
        self.crop, self.training, self.augment = crop, training, augment
        self.cache = {}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        if index not in self.cache:
            with Image.open(self.images[index]) as source:
                image = np.array(source.convert('RGB'))
            with Image.open(self.masks[index]) as source:
                mask = np.array(source)
            if mask.ndim != 2 or image.shape[:2] != mask.shape:
                raise ValueError('Masks must be single-channel and match image dimensions')
            if not set(np.unique(mask)).issubset({0, 1, 255}):
                raise ValueError('Mask values must be 0/1 or 0/255')
            # Decoded once: repeated PNG decoding dominated epoch time on full-size scenes.
            self.cache[index] = (image, (mask > 0).astype(np.uint8))
        image, mask = self.cache[index]
        return tensors(image, mask, self.training, self.crop, self.augment)


def val_dataset(data, crop, val_size, seed, split='val'):
    if data == 'synthetic':
        return SyntheticPonds(val_size, crop + crop // 8, crop,
                              training=False, seed=seed + 100000)
    return FolderDataset(Path(data) / split, crop=crop, training=False)
