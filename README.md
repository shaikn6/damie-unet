# damie-unet

A plain PyTorch U-Net baseline for the University of Dayton Vision Lab's
DAMIE melt-pond dataset. Request DAMIE from the
[lab](https://sites.google.com/a/udayton.edu/vasari1/research/earth-vision/damie).
No lab data is included or downloaded. Synthetic scenes are generated once per dataset:
bright ice, dark irregular ponds and noise, with paired binary masks.

## Quickstart

Python 3.10+:

```sh
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
python train.py --data synthetic --width 16 --epochs 5
python eval.py --checkpoint runs/synthetic/best.pt
python -m pytest -q
ruff check .
```

Devices are selected in order: CUDA, MPS, CPU; `--device cpu` overrides selection.
The standard four-level U-Net uses width 64 (about 31M parameters). Width 16
is a smaller laptop variant. It uses two 3×3 convolutions per level, max pooling,
transposed convolutions and skip connections, following Ronneberger et al. (2015).
Padded convolutions and automatic input padding preserve arbitrary image sizes.

Training uses random 256×256 crops; evaluation uses full images, batching synthetic scenes.
Run `python train.py --help` for options.
Scene seeds are fixed; crops vary during training. Validation uses separate seeds.
`best.pt` selects the highest validation mIoU; `metrics.json` holds config,
per-epoch metrics, best epoch and total `wall_seconds`.

## Real data later

Arrange approved data locally with separate splits and matching filename stems:

```text
data/damie/
  train/images/example.jpg
  train/masks/example.png
  val/images/example.jpg
  val/masks/example.png
  test/images/example.jpg
  test/masks/example.png
```

Images may be PNG or JPG/JPEG. Masks must be single-channel PNG: 0 = ice,
1 = pond (0/255 is also accepted, with 255 = pond). Confirm the lab's label
convention and arrange its official splits before use. Small training images
are edge-padded to the crop size; evaluation preserves full size.

```sh
python train.py --data data/damie --width 16 --epochs 5 --out runs/damie
python eval.py --checkpoint runs/damie/best.pt
```

No source changes are needed for this layout. Keep lab data and trained weights
local; `data/` and `runs/` are ignored.

## Results

Metrics accumulate one confusion matrix over the entire evaluation set.
mIoU is the mean of ice and pond IoU; F1 is for ponds only. An absent class
scores zero and remains in the mean. Per-class IoU and the matrix are also saved.

| Run | Val mIoU | Pond F1 | Wall time |
| --- | --- | --- | --- |
| Synthetic, width 16, 5 epochs, CPU, seed 0 (best: epoch 5) | 0.9829 | 0.9892 | 31.25 s |
| Paper's U-Net on DAMIE, reported test metrics | 0.885 | 0.938 | — |
| This repo on DAMIE (pending data) | — | — | — |

Synthetic numbers only check the pipeline; they are not comparable to DAMIE.
The measured run used the quickstart training command above.

Paper reference: Sultana et al., *ArcticNet for Semantic Segmentation of Meltpond
Regions in the Arctic Sea Ice*, IEEE JSTARS, 2026, Table III. Reported U-Net
mIoU 0.885 / F1 0.938: **to be confirmed against the paper**, including metric
conventions. This baseline trains on crops rather than the paper's full patches.

MIT License. Copyright (c) 2026 Nagizaaz Shaik.
