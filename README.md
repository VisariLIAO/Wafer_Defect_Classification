# Project Overview

This project focuses on classifying wafer defect patterns using convolutional neural networks (CNN). Accurate detection and classification of wafer defects are crucial for semiconductor manufacturing quality control.

# Dataset

We use the Mixed-type Wafer Defect Dataset (MixedWM38), which contains 38,015 wafer maps of size 52×52 pixels. The dataset includes:

- arr_0: Wafer map data where  
  - 0 = blank spot  
  - 1 = normal die (passed electrical test)  
  - 2 = broken die (failed electrical test)  

- arr_1: One-hot encoded labels representing 8 basic defect types (single defects).

This is a multi-label classification task, as one wafer map can exhibit multiple defect types simultaneously.

# Problem Definition

- **Task**: Multi-label classification of wafer defects.  
- **Output layer**: 8 neurons with sigmoid activation function.  
- **Loss function**: Binary cross-entropy (`binary_crossentropy`).

# Model Architecture

Convolutional Neural Network (CNN) with multiple convolutional and pooling layers.

Final dense layer outputs 8 sigmoid-activated values representing defect probabilities.
