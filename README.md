# Fine-Grained Bird Species Classification (CUB-200-2011)
### EfficientNetB4 Transfer Learning vs. a Custom CNN Trained from Scratch

Coursework project for Newcastle University's MSc Data Science & AI programme (module **CSC8637 Deep Learning**, Task 1). Classifies 200 visually similar bird species using two models: a fine-tuned ImageNet-pretrained **EfficientNetB4**, and a **custom CNN** designed and trained from scratch.

Write-up: [`report/Task1_Report.md`](report/Task1_Report.md) · Notebook: [`notebooks/Fine_Grained_Bird_Classification.ipynb`](notebooks/Fine_Grained_Bird_Classification.ipynb)

---

## 1. Background

Fine-grained classification separates classes that differ only in subtle details — here, bird species that share overall shape and colouring and differ in small cues such as wing edges, bill shape, and plumage patterns. It matters in both academia and industry (biodiversity monitoring, ecological surveys), and it is hard because inter-class differences are small while intra-class variation (pose, lighting, background) is large.

## 2. Problem Statement & Brief

The CSC8637 brief asked for:

1. A network that accurately classifies bird species on the Caltech-UCSD **CUB-200-2011** dataset.
2. **Two final models**: one based on a standard deep-learning architecture (ResNet, VGG, DenseNet, Inception, etc. — transfer learning recommended), and one **of the student's own design**.
3. Training only on the provided training set (optionally split ~9:1 into train/validation); the **test set used only for final evaluation**.
4. Hyper-parameter search using the validation set, and final reporting of **accuracy, precision, recall, and a confusion matrix**.
5. Reproducible code (seeds set), trained model checkpoints, and a short report explaining data processing, hyper-parameters, and the reasons behind them.

## 3. Dataset

**CUB-200-2011** [1]: 11,788 images of 200 bird species, with bounding-box annotations. The official split gives 5,994 training and 5,794 test images; the training portion was split 90/10 (stratified, `random_state=42`) into:

| Split | Images |
|---|---|
| Train | 5,394 |
| Validation | 600 |
| Test | 5,794 |

**Access:** public — [Caltech data repository](https://data.caltech.edu/records/65de6-vp158). The dataset is **not included** in this repo. The notebook expects the extracted folder (`images/`, `bounding_boxes.txt`, `images.txt`, `image_class_labels.txt`, `train_test_split.txt`).

## 4. Solution — Approach

### 4.1 Preprocessing: bounding-box cropping
Early experiments on the original uncropped images gave lower validation accuracy because of background clutter and inconsistent bird scale. Each image is therefore cropped to its bounding box with a **15% margin** (to keep discriminative context such as wing edges and plumage), then **padded to a square** with black pixels to avoid aspect-ratio distortion before resizing. In the project's experiments this improved convergence stability and validation accuracy.

### 4.2 Model 1 — EfficientNetB4 (transfer learning)
- ImageNet-pretrained **EfficientNetB4** backbone, input 380×380, batch size 16. (EfficientNetB0 was tried first and plateaued early; B4's extra capacity helped on fine-grained classes.)
- Head: `GlobalAveragePooling2D → BatchNormalization → Dense(512, ReLU) → Dropout(0.5) → Dense(200, softmax)`.
- **Phase 1 — head only:** backbone frozen, Adam lr 3e-4, up to 20 epochs, early stopping, learning-rate reduction on plateau.
- **Phase 2 — fine-tuning:** top **60 layers** unfrozen (BatchNorm layers kept frozen for stability), Adam lr **1e-5** with `clipnorm=1.0`, label smoothing 0.05, up to 20 epochs.
- Augmentation: rotation ±20°, zoom 0.2, horizontal flip, width/height shift 0.15, brightness 0.8–1.2, shear 0.1.
- Regularisation changes that reduced overfitting (training accuracy was approaching ~95% against ~80% validation): dropout 0.4 → 0.5 and a lower fine-tuning learning rate.

### 4.3 Model 2 — Custom CNN (from scratch)
- Input 224×224, batch size 32. Four progressive convolutional stages (**64 → 128 → 256 → 512 filters**), two 3×3 convolutions per stage, each followed by BatchNorm + ReLU, `he_normal` initialisation, strided convolutions for downsampling, then Global Average Pooling and a dropout-regularised classification head.
- Moderate augmentation; Adam with learning-rate reduction on plateau (initial 3×10⁻⁴); early stopping; up to 50 epochs.
- Built as a controlled baseline to measure what pretraining contributes.

## 5. Implementation

Python, TensorFlow/Keras (`ImageDataGenerator` pipelines, `EfficientNetB4`), scikit-learn (classification report, confusion matrix, stratified split), pandas, Pillow, NumPy, Matplotlib/Seaborn. Seeds (`42`) are set for Python, NumPy, TensorFlow and `PYTHONHASHSEED` at the top of the notebook. Developed and run on Google Colab (GPU).

## 6. Results

Evaluated on the 5,794-image held-out test set:

| Model | Test accuracy | Macro precision | Macro recall | Macro F1 |
|---|---|---|---|---|
| **EfficientNetB4** (fine-tuned) | **0.8421** | 0.8484 | 0.8433 | 0.8416 |
| Custom CNN (from scratch) | 0.2761 | — | — | 0.2675 (weighted 0.2671) |

Validation accuracy: EfficientNetB4 reached ~0.815 after head-only training and 0.840 after fine-tuning. The custom CNN peaked at ~0.27 validation accuracy with ~41% training accuracy. Normalised confusion matrices and training curves for both models are produced in the notebook, along with a bar chart comparing test accuracies.

### What the results mean
- **Pretraining is decisive here.** With only ~5.4k training images spread over 200 classes (~27 per class), a network trained from scratch cannot learn enough fine-grained features; ImageNet features transfer well, giving a ~57-point accuracy gap.
- **Precision, recall and F1 are balanced** for EfficientNetB4 (all ≈0.84), so the accuracy is not coming from a few easy classes.
- **Cropping mattered.** In earlier experiments on uncropped images the custom CNN reached very low test accuracy (~30%) despite high training accuracy — it was memorising backgrounds rather than birds. Cropping made optimisation more stable, but the model still stayed far below the transfer-learning model, consistent with the data-scarcity explanation.

## 7. Limitations & Future Work
- The custom CNN underfits/overfits within a small-data regime; stronger augmentation (MixUp/CutMix), residual connections, or longer schedules might narrow the gap, but are not tested here.
- Final EfficientNetB4 accuracy (84.2%) leaves room: unfreezing more layers, a larger input resolution, test-time augmentation, or ensembling are natural next steps.
- Trained checkpoints (`.keras`) were saved to Google Drive and are **not stored in this repo** (size).

## 8. Repository Structure

```
bird-species-classification/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── Fine_Grained_Bird_Classification.ipynb
└── report/
    └── Task1_Report.md
```

## 9. Setup & Running

The notebook was written for **Google Colab**: it mounts Google Drive, copies `bird_CUB_200_2011.zip` from a personal Drive folder, and saves checkpoints back to Drive. To run it elsewhere:

1. Download CUB-200-2011 (link above) and extract it.
2. Remove the `google.colab` / `drive.mount(...)` / `!cp` / `!unzip` cells, and set `ORIGINAL_BASE` to your extracted folder. Change `save_path_1`, `save_path_1_ft` and `save_path_2` to local paths.
3. `pip install -r requirements.txt`
4. Run cells top to bottom: cropping → split → EfficientNetB4 (phase 1, fine-tune, evaluation) → custom CNN (training, evaluation) → comparison. A GPU is strongly recommended.

## 10. References

[1] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie, *The Caltech-UCSD Birds-200-2011 Dataset*, California Institute of Technology, 2011.
[2] M. Tan and Q. Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks," in *Proc. ICML*, 2019.

---

*Author: Emaduddin Asdaq Syed Mohammed — School of Computing, Newcastle University.*
