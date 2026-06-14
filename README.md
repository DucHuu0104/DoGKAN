# DoGKAN: Domain Generalized Kolmogorov-Arnold Networks for EEG Emotion Recognition

This repository contains the PyTorch implementation of the **DoGKAN** model. DoGKAN combines Graph Convolutional Networks (GCN) for spatial feature extraction, Transformers for temporal dynamics, Kolmogorov-Arnold Networks (KAN) for classification, and Domain Adversarial Neural Networks (DANN) to achieve domain-generalized emotion recognition from EEG signals.

## Requirements

The code has been tested with Python 3.8+ and PyTorch. To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

This project uses the **SEED-V** dataset. You need to extract the differential entropy (DE) features and the channel location file (`channel_62_pos.locs`). 

Please organize your data in the following structure, or provide the paths via command-line arguments.

```
data/
└── SEED_V/
    ├── EEG_DE_features/
    │   ├── 1_123.npz
    │   ├── 2_123.npz
    │   └── ...
    └── channel_62_pos.locs
```

*Note: The script expects `.npz` files for the SEED-V dataset, with keys mapping to the trial data.*

## Usage

You can train the model using a Leave-One-Subject-Out (LOSO) cross-validation strategy. By default, the script trains on session 2. 

### Basic Run

If you place the dataset in the default `data/SEED_V/` directory as shown above, you can simply run:

```bash
python main.py
```

### Specifying Custom Data Paths

If your dataset is located elsewhere, use the `--data_root` and `--locs_path` arguments:

```bash
python main.py --data_root /path/to/EEG_DE_features --locs_path /path/to/channel_62_pos.locs
```

### Specifying Sessions and Subjects

You can also specify which sessions and subjects to evaluate on:

```bash
python main.py --sessions 1 2 3 --subjects S1_P1 S1_P2
```

## Output

- **Logs**: Training progress, validation accuracy, and domain loss are saved in the `logs/` directory.
- **Plots**: Confusion matrices for the evaluated models are saved in the `logs/plots/` directory.
- **Checkpoints**: The best model weights (based on validation accuracy and domain loss criteria) are saved in the `checkpoints/` directory.

## License

This project is open-sourced under the MIT License.
