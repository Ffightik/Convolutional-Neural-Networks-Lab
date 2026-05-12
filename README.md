# Deep Learning Laboratory — CNN

**Silesian University of Technology**  
Faculty of Biomedical Engineering  
Department of Medical Informatics and Artificial Intelligence

| | |
|---|---|
| **Course** | Deep Learning Laboratory |
| **Instructor** | dr inż. Agata Sage |
| **Student** | Nataliia Nechyporenko |

---

## Repository Structure

```
├── CNN1_task1.ipynb   # Lab 1, Task 1: CNN training & validation (pneumonia classification)
├── CNN1_task2.ipynb   # Lab 1, Task 2: Common problems + multiclass classification
├── CNN2_ENG.ipynb     # Lab 2: Transfer learning & fine-tuning (skin lesion segmentation)
└── README.md
```

---

## Lab 1 — CNN: Model Validation and Diagnostics

**Goal:** Practical experience in proper CNN validation and diagnostics of common training problems.

### Task 1 — Training & Testing Analysis
- Binary classification of pneumonia in chest X-ray images
- Dataset: **PneumoniaMNIST** (MedMNIST) — 5,856 images (28×28×1)
- Tasks: hyperparameter selection (epochs, optimizer), analysis of learning curves, evaluation metrics, GradCAM visualization

### Task 2 — Common Problems Analysis
- Diagnosing and solving: **overfitting**, **class imbalance**, **data leakage**, suboptimal hyperparameters
- Dataset: PneumoniaMNIST (same as Task 1)

### Task 3* — Multiclass Classification *(additional)*
- Skin lesion classification into 7 classes
- Dataset: **DermaMNIST** — 10,015 images
- Goal: achieve average accuracy ≥ 0.7 on the test set

### Key concepts covered

| Problem | Symptoms | Solutions |
|---|---|---|
| Overfitting | Val loss ↑, Train loss ↓ | Dropout, augmentation, early stopping |
| Underfitting | High loss on both sets | Deeper network, longer training |
| Data leakage | Unrealistically high accuracy | Per-patient split, cross-validation |
| Class imbalance | Poor minority class performance | Oversampling, weighted loss |

---

## Lab 2 — CNN: Transfer Learning and Fine-Tuning

**Goal:** Practical experience with pretrained CNN models, transfer learning and fine-tuning for biomedical image segmentation.

### Task 1 — Transfer Learning
- Complete the **VGG16-UNet** architecture (VGG16 encoder + UNet decoder)
- Split data using `scikit-learn`; set up training callbacks
- Train with frozen encoder weights

### Task 2 — Fine-Tuning
- Unfreeze selected VGG16 encoder blocks (`blocks`: 1–5)
- Experiment with learning rates (`lr`: `1e-4`, `1e-5`, `1e-6`)
- Compare results against transfer learning baseline

### Dataset
**HAM10000** — 10,015 dermatoscopic images with binary skin lesion segmentation masks  
Source: [Kaggle — HAM10000 Lesion Segmentations](https://www.kaggle.com/datasets/tschandl/ham10000-lesion-segmentations)

### Model Architecture
**VGG16-UNet** — encoder-decoder with skip connections:
- **Encoder**: pretrained VGG16 (ImageNet), blocks 1–5
- **Decoder**: UNet-style upsampling with batch normalization
- **Skip connections**: link encoder blocks to decoder layers

---

## Requirements

```bash
conda create -n dl python=3.10
conda activate dl
pip install torch torchvision medmnist grad-cam seaborn scikit-learn
```

---

## Usage

In Lab 2 notebooks, set the `if_train` flag:

```python
if_train = True   # Train from scratch (real results, slower)
if_train = False  # Use pre-saved logs (faster, for testing)
```
