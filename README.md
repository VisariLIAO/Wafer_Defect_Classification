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

The core model is a Convolutional Neural Network (CNN) composed of multiple convolutional and pooling layers designed to effectively extract spatial features from the wafer maps. The network concludes with a fully connected (dense) layer containing 8 neurons with sigmoid activation functions, each outputting the probability of a specific defect type, suitable for the multi-label classification task.

To optimize performance, various CNN architectures such as VGGNet, ResNet, and DenseNet were explored to identify the most suitable model for this dataset. Additionally, several techniques were applied to enhance model generalization and reduce overfitting, including:

Data augmentation: Random transformations such as rotations, flips, and shifts were applied to the training data to artificially increase dataset diversity and improve robustness.

Regularization methods: Techniques such as dropout and L2 weight decay were incorporated to prevent the model from overfitting to the training data.

Early stopping: Training was monitored on a validation set, and stopped when performance ceased to improve to avoid overfitting and reduce unnecessary computation.

Hyperparameter tuning: Parameters like learning rate, batch size, and network depth were systematically adjusted to find the optimal training setup.

Through this comprehensive approach, the model aims to accurately classify multiple defect types on wafer maps while maintaining strong generalization on unseen data.


## Training Details

## Evaluation Metrics

This project addresses a multi-label classification problem where each wafer map may contain multiple defect types. Therefore, a single accuracy metric is insufficient to fully evaluate model performance. We use the following evaluation metrics:

- **Precision**: The ratio of correctly predicted positive observations to the total predicted positives. Indicates the reliability of positive predictions and helps reduce false positives.

- **Recall**: The ratio of correctly predicted positive observations to all actual positives. Reflects the model’s ability to detect positive instances, reducing false negatives.

- **F1-score**: The harmonic mean of Precision and Recall, providing a balance between the two, commonly used in multi-label classification.

- **Sub-accuracy**: The average accuracy calculated per label, measuring the model’s stable performance across each defect type.

All metrics are computed per defect category and averaged (macro or micro averaging) to fairly assess the multi-label classification results.
