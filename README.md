# DoGKAN: A Domain-Adaptive Kolmogorov-Arnold Network for Cross-Subject EEG-Based Emotion Recognition

## Abstract 
> Electroencephalogram (EEG)-based emotion recognition encounters considerable challenges due to inter-subject variability and complex spatio-temporal dependencies, limiting the generalization of existing models in cross-subject contexts. This work presents DoGKAN, a hybrid architecture capable of integrating spatial dependency modeling, temporal sequence learning, and cross-subject domain alignment within a unified framework. Specifically, a Graph Convolutional Network (GCN) is employed to capture spatial interconnections among EEG electrodes, while a Transformer encoder models temporal dynamics of EEG sequences. To enhance nonlinear representation capacity, we replace traditional fully connected layers with Kolmogorov–Arnold Network (KAN), which utilizes learnable spline-based nonlinear mappings to provide greater representational flexibility and built-in sparsity regularization. Furthermore, a Domain-Adversarial Neural Network (DANN) with a Gradient Reversal Layer (GRL) is incorporated to mitigate cross-subject distribution discrepancies. The model's average accuracy under the Leave-One-Subject-Out (LOSO) protocol reaches 77.89\% on SEED-IV and 84.86\% on SEED-V. The results highlight the effectiveness of applying graph-based spatial modeling, Transformer-based temporal learning, and KAN-driven adaptive nonlinear classification for robust cross-subject EEG-based affective computing. 
>
> Index Terms: EEG-based Emotion Recognition, Kolmogorov-Arnold Network, Graph Neural Network, Cross-Subject Adaptation, Brain-Computer Interfaces.

## Framework Overview

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

## Dataset Preparation

This project supports both the **SEED-IV** and **SEED-V** datasets. You need to extract the differential entropy (DE) features and the channel location file (`channel_62_pos.locs`). 

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

No other file needs to be modified.

## Usage

You can train the model using a Leave-One-Subject-Out (LOSO) cross-validation strategy. By default, the script trains on session 2.

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

## Output

- **Logs**: Training progress, validation accuracy, and domain loss are saved in the `logs/` directory.
- **Plots**: Confusion matrices for the evaluated models are saved in the `logs/plots/` directory.
- **Checkpoints**: The best model weights (based on validation accuracy and domain loss criteria) are saved in the `checkpoints/` directory.

## License

This project is open-sourced under the MIT License.
