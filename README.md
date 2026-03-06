# [ICLR26 Workshop] RLLay: Reinforce Your Layouts — Online Reward-Guided RL for Layout-to-Image Diffusion Fine-Tuning

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg?logo=arxiv&logoColor=red)](https://arxiv.org/abs/XXXX.XXXXX)
[![OpenReview](https://img.shields.io/badge/OpenReview-ICLR%202026-2F80ED.svg)](https://openreview.net/forum?id=YOUR_OPENREVIEW_ID)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB.svg?logo=python&logoColor=FFD43B)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg?logo=creativecommons&logoColor=white)](https://creativecommons.org/licenses/by-nc/4.0/)

**RLLay (Reinforce Your Layouts)** is an online reinforcement learning framework for layout-to-image generation that directly fine-tunes diffusion models to improve spatial consistency between generated images and user-specified layouts. Given a prompt and a set of region descriptions with target boxes, RLLay samples multiple candidate images, ranks them using an IoU-based layout alignment reward (via open-vocabulary detection), and forms hard-negative preference pairs to amplify the training signal. We further introduce **ARPO**, a preference-based optimization method that leverages explicit diffusion trajectory log-probabilities to stabilize online learning. RLLay improves spatial layout fidelity while maintaining image quality and text-image alignment across SD1.5- and SD3-based backbones.

<p align="center">
  <img src="assets/figures/teaser.png" width="720" alt="RLLay Overview" />
</p>

------
