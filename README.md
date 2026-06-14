# DoGKAN: A Domain-Adaptive Kolmogorov-Arnold Network for Cross-Subject EEG-Based Emotion Recognition

<i>
  Official code repository for the manuscript 
  <b>"DoGKAN: A Domain-Adaptive Kolmogorov-Arnold Network for Cross-Subject EEG-Based Emotion Recognition"</b>, submitted to 
  <a href="https://atc-conf.org/">INTERNATIONAL CONFERENCE ON ADVANCED TECHNOLOGIES FOR COMMUNICATIONS</a>.
</i>

> Please press ⭐ button and/or cite papers if you feel helpful.

<p align="center">
<img src="https://img.shields.io/github/stars/DucHuu0104/DoGKAN">
<img src="https://img.shields.io/github/forks/DucHuu0104/DoGKAN">
<img src="https://img.shields.io/github/watchers/DucHuu0104/DoGKAN">
</p>

<div align="center">

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/Torch_2.0.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![cuda](https://img.shields.io/badge/-CUDA_11.8-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit-archive)

</div>

<p align="center">
<img src="https://img.shields.io/badge/Last%20updated%20on-14.06.2026-brightgreen?style=for-the-badge">
<img src="https://img.shields.io/badge/Written%20by-Duc%20Nguyen%20Huu%20Dang-lime?style=for-the-badge"> 
</p>

<div align="center">

[**Abstract**](#Abstract) •
[**Overview**](#Overview) •
[**Dataset**](#Dataset) •
[**Usage**](#usage) •
[**Citation**](#citation) •
[**Contact**](#Contact)

</div>

## Abstract 
> Electroencephalogram (EEG)-based emotion recognition encounters considerable challenges due to inter-subject variability and complex spatio-temporal dependencies, limiting the generalization of existing models in cross-subject contexts. This work presents DoGKAN, a hybrid architecture capable of integrating spatial dependency modeling, temporal sequence learning, and cross-subject domain alignment within a unified framework. Specifically, a Graph Convolutional Network (GCN) is employed to capture spatial interconnections among EEG electrodes, while a Transformer encoder models temporal dynamics of EEG sequences. To enhance nonlinear representation capacity, we replace traditional fully connected layers with Kolmogorov–Arnold Network (KAN), which utilizes learnable spline-based nonlinear mappings to provide greater representational flexibility and built-in sparsity regularization. Furthermore, a Domain-Adversarial Neural Network (DANN) with a Gradient Reversal Layer (GRL) is incorporated to mitigate cross-subject distribution discrepancies. The model's average accuracy under the Leave-One-Subject-Out (LOSO) protocol reaches 77.89\% on SEED-IV and 84.86\% on SEED-V. The results highlight the effectiveness of applying graph-based spatial modeling, Transformer-based temporal learning, and KAN-driven adaptive nonlinear classification for robust cross-subject EEG-based affective computing. 
>
> Index Terms: EEG-based Emotion Recognition, Kolmogorov-Arnold Network, Graph Neural Network, Cross-Subject Adaptation, Brain-Computer Interfaces.

## Overview

<p align="center">
  <img src="assets/DoGKAN.png" width="900">
</p>

Overview of DoGKAN, integrating GCN-based spatial modeling,
Transformer temporal learning, KAN classification, and
DANN-based domain adaptation.

## Requirements

The code has been tested with Python 3.8+ and PyTorch. To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Dataset

This project supports both the **SEED-IV** and **SEED-V** datasets.

Please organize your data in the following structure:

<h4>🔹 SEED-IV </h4>
<ul>
  <li><strong>Description:</strong> EEG emotion recognition dataset with 4 emotion classes collected from 15 subjects across 3 sessions using 62-channel EEG recordings.</li>
  <li><strong>Access:</strong> Requires signing a license agreement.</li>
  <li><a href="https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html" target="_blank">🔗 Download Link</a></li>
</ul>

<h4>🔹 SEED-V</h4>
<ul>
  <li><strong>Description:</strong> EEG emotion recognition dataset with 5 emotion classes collected from 16 subjects across 3 sessions using 62-channel EEG recordings.</li>
  <li><strong>Access:</strong> Requires signing a license agreement.</li>
  <li><a href="https://bcmi.sjtu.edu.cn/home/seed/seed-v.html" target="_blank">🔗 Download Link</a></li>
</ul>

*Note: SEED-IV data files are typically `.mat` format, while SEED-V data files are typically `.npz` format. The dataset loading scripts expect keys mapping to the respective trial data.*

## Dataset Selection

Choose the dataset in `config.py`:

```python
DATASET = "SEED_V"
# DATASET = "SEED_IV"
```

All dataset-specific paths, labels, class counts, and loaders are selected automatically.

## Usage

You can train the model using a Leave-One-Subject-Out (LOSO) cross-validation strategy.

### Basic Run

If you place the dataset in the default `data/` directory, you can run:

```bash
python main.py
```

### Specifying Sessions and Subjects

You can specify which sessions and subjects to evaluate on via command-line arguments:

```bash
python main.py --sessions 1 2 3 --subjects S1_P1 S1_P2
```

## Citation
If you use this code or concept in your research, please consider citing our original manuscript:
```bibtex
Coming soon
```

## Contact
For any information, please contact the corresponding author:

**Duc Nguyen Huu Dang** at AiTA Lab, Faculty of Information Technology, FPT University, Vietnam<br>
**Email:** [huuduc01042006@gmail.com](mailto:huuduc01042006@gmail.com) or [ducdnhse203583@fpt.edu.vn](mailto:ducdnhse203583@fpt.edu.vn) <br>
**GitHub:** <link>https://github.com/DucHuu0104/</link>
