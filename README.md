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

## Real DAMIE data

The lab's download has `<split>/image/*.PNG` and palette-mode labels in
`<split>/labels/*.png`, already split 120 / 30 / 20 (train / val / test). Convert it once:

```sh
python scripts/prepare_damie.py data/raw data/damie
python train.py --data data/damie --width 16 --epochs 100 --batch 8 --repeat 15 --val-every 5 --out runs/w16_seed0
python eval.py --checkpoint runs/w16_seed0/best.pt --split test
```

`prepare_damie.py` symlinks the images and writes 0/1 masks into the layout
`FolderDataset` reads (`<split>/images`, `<split>/masks`, matching stems; masks 0 = ice,
1 = pond, 0/255 also accepted). In the labels, palette index 1 is pond and index 3 is ice;
this was checked against mean image colour (pond pixels are darker and bluer). Index 0
appears on 3,072 stray pixels in six label files (none in test) and is treated as non-pond.
The script stops if a split's count, a label's mode or a label value is unexpected.

`--repeat` sets random crops per image per epoch, `--val-every` spaces out the
full-image validation passes, and `--resume` continues from `<out>/last.pt` after a crash.
Keep lab data and trained weights local; `data/` and `runs/` are ignored.

## Results

Metrics accumulate one confusion matrix over the entire evaluation set (pixel-pooled).
Precision, recall, F1 and IoU are for the pond class; mIoU is the mean of ice and pond
IoU. The per-image row averages each scene's pond IoU instead of pooling pixels.

DAMIE test split (20 images), width-16 U-Net (0.5M parameters), Apple MPS. Each seed's
checkpoint was chosen on the validation split and scored on test once.

| Metric | Seed 0 (100 epochs) | Seeds 0–2, mean ± std | Paper's U-Net (Table III) |
| --- | --- | --- | --- |
| Pond IoU | 0.9607 | 0.9555 ± 0.0053 | 0.885 (reported as "mIoU") |
| Pond IoU, per-image mean | 0.9552 | 0.9487 ± 0.0064 | — |
| mIoU, two-class mean | 0.9741 | 0.9707 ± 0.0035 | — |
| Pond F1 | 0.9800 | 0.9772 ± 0.0028 | 0.938 |
| F1, two-class mean | 0.9869 | 0.9851 ± 0.0018 | — |
| Precision | 0.9848 | 0.9853 ± 0.0015 | 0.969 |
| Recall | 0.9752 | 0.9693 ± 0.0053 | 0.914 |
| Accuracy | 99.05% | 98.92% ± 0.13 | 95.75% |

Seeds 1 and 2 ran 30 epochs, because seed 0's validation score was flat after about
epoch 15. Wall time: 64 min for seed 0, 24–28 min for the others.

How to read the comparison:

- **Which paper metric is which.** The paper does not define its metrics. Its U-Net row
  is consistent with pond-class numbers: IoU = F1 / (2 − F1) = 0.938 / 1.062 = 0.883,
  close to the reported 0.885, and 2PR / (P + R) = 0.941, close to the reported F1.
  The paper's "mIoU" is therefore compared with pond IoU here; the two-class readings
  are listed as well.
- **Not a reproduction.** This is a lighter variant: width 16 instead of the standard
  width 64, random 256×256 crops instead of full 1000×1000 patches, Adam at 1e-3 instead
  of 1e-4, batch 8, no augmentation and no learning-rate schedule.
- **Small test set.** 20 images and three seeds; differences of about 0.01 are noise.
  Whether these are the same 20 test images the paper scored is assumed from the
  matching 120 / 30 / 20 split, not confirmed.
- **Splits.** Test shares no flight series with train. Validation shares one series
  (9 of 30 images), so checkpoint selection is slightly optimistic; test is unaffected.
- **Pooling.** The paper may average per image rather than pool pixels; both are given.

The gap to the paper's row comes with protocol differences that cannot be resolved from
the paper, so it should not be read as a finding about the paper's model.

The first attempt at seed 0 stopped at epoch 34 with an MPS GPU error; the reported run
is a clean rerun from scratch with the same seed, which matched the first attempt's
validation numbers exactly up to that point.

Synthetic pipeline check (not comparable to DAMIE): width 16, 5 epochs, CPU, seed 0,
val mIoU 0.9829, pond F1 0.9892, 31 s.

Paper reference: Sultana et al., *ArcticNet for Semantic Segmentation of Meltpond
Regions in the Arctic Sea Ice*, IEEE JSTARS, 2026, Table III.
