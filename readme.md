# M2Heat: Heat Conduction Modeling for Hyperspectral and LiDAR Joint Classification

## Configuration

Install the dependencies:

```bash
pip install -r requirement.txt
```

Data are read from `datasets/` by default. Override the location with
`--data-root`.

## Train

Train one dataset and write checkpoints and metrics under `log/`:

```bash
python demo.py --dataset trento --device cuda --gpu-id 0
```

Run all three datasets sequentially:

```bash
python demo.py --dataset all --device cuda --gpu-id 0
```

Useful settings can be overridden, for example:

```bash
python demo.py --dataset trento --epochs 500 --batch-size 64 --patch-size 13 --learning-rate 5e-4
```

## Test

Evaluate a saved checkpoint:

```bash
python demo.py --dataset trento --mode test --checkpoint log/<run>/best.pt --device cuda --gpu-id 0
```

Each run stores `config.json`, `environment.json`, `metrics.csv`,
`metrics.json`, `best.pt`, `last.pt`, and `stdout.log`.
