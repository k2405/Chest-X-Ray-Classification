# README for Chest X-Ray Classification Project

## Overview
This project explores the application of convolutional neural networks (CNNs) to classify thoracic diseases using the NIH ChestX-ray14 dataset. It evaluates the performance of multiple pretrained CNN architectures, data augmentation techniques, and optimization strategies to enhance diagnostic accuracy.


---

### Acknowledgments
This project was completed by **Team 24**:
- Jon Andreas Bull Larssen  
- Jon Ingvar Skånøy  
- Isak Killingrød

---

### Key Objectives:
1. Compare popular CNN architectures: DenseNet, ResNet, VGG, and MobileNet.
2. Optimize model performance through hyperparameter tuning.
3. Analyze the effect of input channel configurations and image resolutions.
4. Assess data augmentation and training strategies to reduce overfitting.

---

## Dataset
**NIH ChestX-ray14 Dataset**  
- Total Images: 112,120  
- Number of Patients: 30,805  
- Disease Classes: 14  
- Format: Grayscale images

### Preprocessing:
- **Train/Validation/Test Splits:** Dataset split to ensure no overlap in patients.
- **Channel Handling:** Conversion of single-channel grayscale to three-channel format for compatibility with pretrained models.

---

## Methodology
### Models Evaluated:
- **DenseNet (121, 161, 201)**
- **ResNet (50)**
- **MobileNetV2**
- **VGG16**

### Training Details:
- Framework: PyTorch  
- Optimizer: AdamW  
- Scheduler: CosineAnnealingLR  
- Loss Function: Binary Cross-Entropy with Logits  
- Hyperparameter Tuning: Optuna  

### Augmentation Techniques:
- Random rotations, resized crops, horizontal flips, and color jitter were employed to expand dataset variability.

---

## Results
1. **Best Performing Model:** DenseNet161 exhibited superior performance across most disease classes.  
2. **Input Channels:** Minimal difference between one-channel and three-channel configurations.  
3. **Image Resolution:** Higher resolution (448x448) led to marginal accuracy improvements, particularly for classes with subtle features.  
4. **Optimizers:** Performance was comparable among Adam, AdamW, and NAdam optimizers.

---

## Hardware
- **Shared Server:** Tesla V100 GPU, 32GB VRAM  
- **Local Machines:** NVIDIA RTX 3080, RTX 3070 GPUs  

---

## Key Findings
- Data augmentation significantly reduces overfitting.
- DenseNet161 provides the best trade-off between computational cost and diagnostic accuracy.
- Larger input image sizes improve model performance on smaller pathologies.



This README was written by ChatGPT.
