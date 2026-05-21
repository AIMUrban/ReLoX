# ReLoX

This repository contains the **PyTorch implementation** for the paper:**Where to Go Next: Enhancing Zero-Shot Capability for Cross-City Mobility Prediction**

The current codebase is streamlined to a single **base model** pipeline, with:
- visual encoder (`custom_resnet.py`)
- sequential encoder + fusion (`framework.py`)
- training / within-city test / cross-city zero-shot test entry (`main.py`)

## Project Structure

- `main.py`: unified entrypoint for training, within-city test, and cross-city zero-shot test
- `framework.py`: base model definition
- `traj_dataset.py`: dataset loading, local/relative label construction, optional image cache
- `tools.py`: evaluation and reporting utilities
- `run_city_train_then_test.sh`: one-command run script for cross-city experiments

## Datasets

We provide the **preprocessed datasets** used in this project for reproducibility.
For access to **raw datasets** and official data policy, please refer to:

- YJMob100K: https://sigspatial2025.sigspatial.org/giscup/index.html
- Foursquare-NYC: https://sites.google.com/site/yangdingqi/home/foursquare-dataset

Reference:

```text
Takahiro Yabe, Kota Tsubouchi, Toru Shimizu, Yoshihide Sekimoto, Kaoru Sezaki, Esteban Moro, Alex Pentland.
YJMob100K: City-Scale and Longitudinal Dataset of Anonymized Human Mobility Trajectories.
Scientific Data, 11(1), 397, 2024.
```

```text
Dingqi Yang, Daqing Zhang, Vincent W. Zheng, Zhiyong Yu.
Modeling User Activity Preference by Leveraging User Spatial Temporal Characteristics in LBSNs.
IEEE Transactions on Systems, Man, and Cybernetics: Systems (TSMC), 45(1), 129-142, 2015.
```

## Data Format

Place data under:

```text
dataset/<city_name>/
```

The repository provides the preprocessed city datasets as zip files under
`dataset/`. Unzip them before running:

```bash
unzip dataset/kumamoto.zip -d dataset/
unzip dataset/nyc250m.zip -d dataset/
```

For each city, this code expects files like:

- `train.npz`, `val.npz`, `test.npz`
- `train_traj_frames.npz`, `val_traj_frames.npz`, `test_traj_frames.npz`
- `user_frames.npz`
- `poi_map.npy` (optional, will fallback to zeros if missing)

For some reporting utilities:
- `uid_map.json`, `loc_map.json` (recommended)

## Run

### 1. Run with Script (Recommended)

Use the provided script to run both within-city and cross-city zero-shot evaluation with unified settings:

```bash
./run_city_train_then_test.sh \
  --train kumamoto \
  --tests kumamoto sapporo nyc250m \
  --gpu 0 \
  --seed 42 \
  --epochs 10 \
  --batch_size 256 \
  --lr 1e-3
```

Notes:
- Single GPU: script calls `python main.py ...`
- Multi-GPU (e.g., `--gpu 0,1`): script automatically calls `torchrun --nproc_per_node=2 ...`

### 2. Run `main.py` Directly

#### Cross-city zero-shot only (load existing checkpoint)

```bash
python main.py \
  --data_root dataset/kumamoto \
  --test_data_roots dataset/hiroshima dataset/sapporo \
  --zeroshot_only \
  --zeroshot_ckpt best
```

#### Within-city test only

```bash
python main.py \
  --data_root dataset/kumamoto \
  --test_only \
  --zeroshot_ckpt best
```

## Outputs

By default outputs are saved to:

```text
<data_root>/outputs/
```

Main artifacts:
- `model_ckpt/model_epoch_XXX.pt`
- `model_ckpt/model_best.pt`
- `model_ckpt/model.pt`
- `test_results.txt`
- `zeroshot_results.txt`
