# [ICLR26 Workshop] RLLay: Reinforce Your Layouts — Online RL Fine-tuning for Layout-to-Image Diffusion Models

[![Paper](https://img.shields.io/badge/OpenReview-Forum-4b44ce.svg)](https://openreview.net/forum?id=awfx7eNf5D&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FWorkshop%2FMM_Intelligence%2FAuthors%23your-submissions))
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=FFD43B)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg?logo=apache&logoColor=white)](http://www.apache.org/licenses/LICENSE-2.0)

**RLLay (Reinforce Your Layouts)** is an online reinforcement learning framework for layout-to-image generation that directly fine-tunes diffusion models to improve consistency between generated images and user-specified layouts. Instead of relying on indirect side guidance, RLLay samples multiple candidates per (prompt, layout), ranks them with an IoU-based layout reward computed from detected boxes (e.g., via GroundingDINO), and forms extreme preference pairs (hard negatives) to strengthen the training signal. We further introduce **ARPO**, a pairwise preference optimization method that uses explicit trajectory log-probabilities to stabilize online learning. Combined with a curriculum from easy to hard layouts, RLLay improves spatial layout fidelity while maintaining semantic alignment and image quality across SD1.5- and SD3-based backbones.

<p align="center">
  <img src="assets/figures/RLLay.png" width="720" alt="RLLay Overview" />
</p>

------

## Quick Start

### Setup

1. **Environment setup**
```bash
conda create -n rllay python=3.10 -y
conda activate rllay
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
2. **Requirements installation**
```bash
pip install -r requirements.txt
```
### Usage example
