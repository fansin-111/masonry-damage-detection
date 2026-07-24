# Masonry Surface Damage Segmentation

Core code for binary point-cloud segmentation of masonry surface damage using
the improved PointNet++ model.

## Requirements

- Python 3.9
- PyTorch with CUDA 11.8
- CUDA-capable GPU

```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```

## Data

Dataset: https://doi.org/10.5281/zenodo.21536673

```text
data/
  train/*.npy
  val/*.npy
  test/*.npy
```

Each file is an `N x 7` array containing `X, Y, Z, R, G, B, label`, where
`label` is 0 for intact points and 1 for damaged points.

## Train

```bash
python train.py --data-root data
```

## Evaluate

```bash
python evaluate.py --data-root data --checkpoint weights/best_model.pth
```
