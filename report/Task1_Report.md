###I. Introduction

This study investigates fine-grained bird classification on the CUB-200-2011 dataset [1] using two approaches: transfer learning with EfficientNet and a custom convolutional neural network (CNN) trained from scratch. The objective was to evaluate how model capacity and training strategy influence generalisation performance on a dataset containing 200 visually similar bird species.

###II. Dataset Preprocessing

Initial experiments using the original uncropped images resulted in lower validation accuracy, primarily due to background clutter and inconsistent object scale. Since the dataset provides bounding box annotations [1], these were used to crop each bird instance to improve object focus. A 15% margin was added around each bounding box to preserve discriminative contextual cues such as wing edges and plumage patterns, which are essential for distinguishing similar species.

Following cropping, images were padded to square format prior to resizing. This prevented geometric distortion and ensured consistent spatial dimensions across samples. In practice, this preprocessing pipeline improved convergence stability and increased validation accuracy compared to training on the original images.

###III. Transfer Learning with EfficientNet

EfficientNetB0 [2] was initially evaluated; however, validation performance plateaued early. Replacing it with EfficientNetB4 [2] resulted in consistent improvement, indicating that the increased representational capacity better captured subtle inter-class differences across 200 fine-grained categories.

Earlier configurations achieved approximately 80% validation accuracy but exhibited noticeable overfitting, with training accuracy approaching 95%. To improve generalisation, dropout was increased (0.4 → 0.5), the fine-tuning learning rate was reduced (3×10⁻⁴ → 1×10⁻⁵), and the top 60 layers were unfrozen while Batch Normalisation layers remained frozen for stability. These refinements improved performance, achieving a test accuracy of 84.21%, with balanced macro precision (0.8484), recall (0.8433), and F1-score (0.8416).

###IV. Custom CNN (Training from Scratch)

A custom convolutional neural network was implemented as a controlled baseline and trained from scratch using four progressive convolutional stages (64–512 filters) with Batch Normalisation, ReLU activations, strided downsampling, and Global Average Pooling, followed by a dropout-regularised classification head. Moderate augmentation and learning rate scheduling (initial 3×10⁻⁴, reduced on plateau) were applied. In earlier experiments using uncropped images, the model achieved higher training accuracy but very low test performance (~30%), indicating strong memorisation and overfitting due to background bias. After introducing bounding-box cropping with margin and consistent preprocessing, optimisation became more stable; however, the final model reached 27.61% test accuracy (macro F1 ≈ 0.27), with training accuracy ~41% and validation peaking at 27%. Despite architectural tuning and regularisation, the limited dataset size and absence of pretraining constrained feature learning, reinforcing the superiority of the EfficientNetB4 transfer learning approach for this fine-grained task.

###References

[1] C. Wah, S. Branson, P. Welinder, P. Perona, and S. Belongie, The Caltech-UCSD Birds-200-2011 Dataset, California Institute of Technology, 2011.

[2] M. Tan and Q. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” in Proceedings of the 36th International Conference on Machine Learning (ICML), 2019.
