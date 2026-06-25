# EvTurb: Event Camera Guided Turbulence Removal

**[TCI / arXiv 2025]** &nbsp; [![arXiv](https://img.shields.io/badge/arXiv-2508.10582-b31b1b.svg)](https://arxiv.org/abs/2508.10582)

> **EvTurb: Event Camera Guided Turbulence Removal**  
> Yixing Liu\*, Minggui Teng\*, Yifei Xia, Peiqi Duan, Boxin Shi†  
> State Key Laboratory of Multimedia Information Processing, Peking University  
> (\* Equal contribution, † Corresponding author)

## Overview

EvTurb is an event camera-guided framework for atmospheric turbulence removal. It decouples turbulence distortion into blur and geometric tilt effects and addresses them with a two-stage network:

- **D-Net** — A multi-scale U-Net that fuses event integrals with RGB frames via Event-Guided Deformable Convolution (EGDC) to reduce blur.
- **T-Net** — A flow-based network that estimates tilt flow guided by a variance map derived from raw event streams, then warps the coarse D-Net output to correct geometric distortions.

![Pipeline](assets/pipeline.png)

## TurbEvent Dataset

We introduce **TurbEvent**, the first real-captured dataset for event-guided turbulence removal. It contains **6,318 pairs** of turbulent/turbulence-free images at 592×592 resolution with corresponding event streams, captured using a hybrid RGB+event camera system under controlled air turbulence conditions.

Dataset download: [coming soon]

### Dataset structure

```
TurbEvent/
├── frame/          # turbulent RGB input frames (.png)
├── gt/             # ground-truth turbulence-free frames (.png)
├── event/          # stacked event representations (.npz)
├── train.json      # list of training sample IDs
└── test.json       # list of test sample IDs
```

### Preprocessing: computing variance maps

Variance maps can be generated from raw event data using:

```python
from utils import compute_variance_map, event_stack_without_polarity

event = np.load('path/to/events.npy')            # raw events array
stacked = event_stack_without_polarity(event, H, W, T=1024)
var_map = compute_variance_map(stacked)
np.savez('path/to/var.npz', var_map.numpy())
```

## Installation

```bash
pip install -r requirements.txt
```

## Training

**End-to-end training (recommended):**

```bash
python train-end2end.py \
    --data_root /path/to/TurbEvent \
    --checkpoint_dir checkpoints \
    --epochs 150 \
    --batch_size 16
```

Resume from a checkpoint:

```bash
python train-end2end.py \
    --data_root /path/to/TurbEvent \
    --resume checkpoints/best_model.pth
```

## Evaluation

**End-to-end evaluation:**

```bash
python test-end2end.py \
    --data_root /path/to/TurbEvent \
    --checkpoint checkpoints/best_model.pth \
    --save_dir results
```

## Code Structure

```
EvTurb/
├── UNet.py          # D-Net (DeblurNet) — Stage 1 blur removal
├── flow.py          # T-Net (IFNet) — Stage 2 tilt correction
├── Blocks.py        # EGDC, BaselineBlock, EventImageFusion, etc.
├── arch.py          # Basic CNN building blocks
├── warper.py        # Optical flow warping
├── Dataset.py       # TurbEvent dataset loader
├── utils.py         # Event processing and evaluation metrics
├── ssim.py          # Differentiable SSIM loss
├── train-end2end.py # End-to-end training script
└── test-end2end.py  # End-to-end evaluation
```

## Citation

```bibtex
@ARTICLE{11538293,
  author={Liu, Yixing and Teng, Minggui and Xia, Yifei and Duan, Peiqi and Shi, Boxin},
  journal={IEEE Transactions on Computational Imaging}, 
  title={EvTurb: Event Camera Guided Turbulence Removal}, 
  year={2026},
  volume={12},
  number={},
  pages={1060-1072},
  keywords={Modeling;Cameras;Computers;Computer vision;Event detection;Educational institutions;Pattern recognition;Training;Videos;Conferences;Computational photography;event camera;turbulence removal},
  doi={10.1109/TCI.2026.3697623}}
```

## Acknowledgements
- Warp function adapted from [RIFE](https://github.com/megvii-research/ECCV2022-RIFE)
