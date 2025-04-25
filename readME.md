## From Contemporary Datasets to Cultural Heritage: Performance and Energy Profiling of Vision Models Towards Ancient Fabric Recognition

This repository contains a series of reproducible machine learning experiments designed to evaluate and compare the performance and energy efficiency of various deep learning architectures on the highly imbalanced [FABRICS dataset](https://ibug.doc.ic.ac.uk/resources/fabrics/). The dataset consists of images of textile samples from multiple categories with significant class imbalance, posing a challenge for standard supervised learning models.

The experimental setup integrates state-of-the-art convolutional and transformer-based models, energy usage profiling tools, and modern experiment tracking techniques using DVC. The pipeline supports energy-aware benchmarking in realistic settings, with emphasis on both model performance and resource consumption.

---

## Objectives

1. **Evaluate classification performance** of modern deep learning models on an imbalanced image dataset.
2. **Quantify energy consumption and CO₂ emissions** during model training and inference using the CodeCarbon framework.
3. **Compare architectures** in terms of:
   - Predictive performance (accuracy, precision, recall, F1-score, per-class and macro)
   - Energy efficiency (power consumption, total energy used, emissions)
   - Resource footprint (GPU/CPU/RAM utilization)
4. **Enable reproducibility** via DVC pipelines and YAML-configurable scripts.

---

| Model               | Type                   | Architecture Details                                      |
|---------------------|------------------------|-----------------------------------------------------------|
| ResNet50            | CNN                    | Deep residual network (50 layers)                         |
| Vision Transformer  | Transformer            | Patch-based image classification (ViT-B/16)               |
| Swin Transformer    | Hierarchical Transformer| Window-based local self-attention                        |
| ConvNeXt-T          | Hybrid CNN             | ConvNet redesigned with transformer-like components       |
| EfficientNetV2-S    | Scaled CNN             | Progressive compound scaling                              |
| MaxViT              | Multi-Axis Transformer | Multi-scale vision transformer from Google                |

All models are implemented using the `torchvision` library with support for pretrained weights (ImageNet1K), and are fine-tuned on the FABRICS dataset using consistent settings.

---

## Dataset Description

- **Source**: [Imperial College iBUG Group](https://ibug.doc.ic.ac.uk/resources/fabrics/)
- **Classes**: 20 fabric types (e.g., Cotton, Wool, Denim, Suede, etc.)
- **Class Distribution**: Highly imbalanced (some classes with <10 images, others >200)
- **Split**: Stratified split into training and validation subsets
- **Augmentation**: Basic resizing and tensor conversion; no heavy augmentation used to preserve consistency

---

## Metrics Collected

### Performance Metrics

- Accuracy (overall and per-class)
- Precision / Recall / F1-score (macro, micro, weighted)
- Confusion Matrix (visualized)
- Training and Validation Loss Curves
- Training Accuracy Curve

### Energy and Resource Usage

- Energy consumption (in kWh)
- Emissions estimation (in kg CO₂e)
- CPU, GPU, and RAM power usage (tracked via CodeCarbon and NVIDIA SMI)
- GPU power draw logs per experiment

---

## Experiment Tracking & Reproducibility

Experiments are version-controlled and executed using [DVC (Data Version Control)](https://dvc.org/). Each model training script is tracked as a DVC stage with explicit dependencies on data, configuration, and output paths.

### Pipeline Example

```bash
dvc repro train_resnet50
dvc repro train_vit
dvc repro train_swin
```

All experiments can be reproduced by running the corresponding DVC stages. Configuration is controlled via YAML files located in the `configs/` directory.

---

## Folder Structure

```
fabric_models/
├── configs/           # YAML configuration files per model
├── dataset/           # Contains processed train/val splits
├── models/            # Trained model checkpoints (.pt)
├── results/           # Metrics, plots, and CodeCarbon logs
├── scripts/           # Utility scripts
├── training/          # Training scripts per model
├── dvc.yaml           # DVC pipeline definition
├── dvc.lock           # Locked versions of inputs/outputs
└── README.md
```

---

## Technologies Used

- Python 3.10
- PyTorch & torchvision
- scikit-learn
- matplotlib
- CodeCarbon (energy tracking)
- DVC (pipeline & data management)
- PowerShell / Bash (GPU monitoring)
- Windows 11 (local training with RTX 3050)

---

## Energy Profiling with CodeCarbon

Energy consumption and emissions are monitored using [CodeCarbon](https://mlco2.github.io/codecarbon/), which estimates:

- CPU, GPU, RAM energy usage
- Total electricity consumption (kWh)
- CO₂ equivalent emissions
- Geographic energy impact (Greece)

Power draw from the GPU is also logged directly using `nvidia-smi` in 5-second intervals, stored in `gpu_power.log`.

---

## Example Output (ResNet50)

- **Final Accuracy**: 86.5%
- **Macro F1-score**: 0.743
- **Total Energy Used**: 0.0072 kWh
- **Emissions**: 0.00049 kg CO₂e

Visual outputs available:

- `loss_curve.png`
- `accuracy_curve.png`
- `confusion_matrix.png`
- `resnet50_heatmap_grid.png`
  



---

## License

This project is licensed under the MIT License. See `LICENSE` for more information.
