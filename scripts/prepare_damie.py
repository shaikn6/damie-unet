"""Convert the DAMIE download (<split>/image + palette PNGs in <split>/labels) to the
FolderDataset layout (<split>/images + 0/1 masks). Images are symlinked, never copied."""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

EXPECTED = {'train': 120, 'val': 30, 'test': 20}
# Palette indices: 1 is pond (darker, bluer pixels), 3 is ice. Index 0 (black) shows up on a few
# thousand stray pixels in six train labels; it is treated as non-pond.
PALETTE_VOID, PALETTE_ICE, PALETTE_POND = 0, 3, 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('raw', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    for split, count in EXPECTED.items():
        images = sorted((args.raw / split / 'image').glob('*.[Pp][Nn][Gg]'))
        if len(images) != count:
            sys.exit(f'{split}: expected {count} images, found {len(images)}')
        (args.out / split / 'images').mkdir(parents=True, exist_ok=True)
        (args.out / split / 'masks').mkdir(parents=True, exist_ok=True)
        stray = 0
        for image in images:
            label = args.raw / split / 'labels' / f'{image.stem}.png'
            if not label.is_file():
                sys.exit(f'{split}: missing label for {image.name}')
            with Image.open(label) as source:
                if source.mode != 'P':
                    sys.exit(f'{label}: expected palette mode P, got {source.mode}')
                indices = np.array(source)
            if not set(np.unique(indices)).issubset({PALETTE_VOID, PALETTE_ICE, PALETTE_POND}):
                sys.exit(f'{label}: unexpected values {np.unique(indices).tolist()}')
            stray += int((indices == PALETTE_VOID).sum())
            Image.fromarray((indices == PALETTE_POND).astype(np.uint8)).save(
                args.out / split / 'masks' / f'{image.stem}.png')
            link = args.out / split / 'images' / f'{image.stem}.png'
            link.unlink(missing_ok=True)
            link.symlink_to(image.resolve())
        print(f'{split}: {count} pairs ok, {stray} stray index-0 pixels')


if __name__ == '__main__':
    main()
