## 🛠️ Challenges & Solutions


Throughout the project, several challenges were encountered and systematically resolved to improve model performance and robustness.

| **Challenge**                               | **Solution**                                                                                                                                                    |
|---------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Imbalanced defect types                     | Applied data augmentation (rotation, zoom, shift, horizontal flip) to generate more varied samples and mitigate class imbalance.                              |
| Overfitting in early models (model1~model5) | Introduced L2 regularization and Dropout layers in both CNN and MLP branches. Also adopted EarlyStopping and learning rate reduction strategy.               |
| Dying ReLU neurons                          | Switched from ReLU to LeakyReLU activation, which retains small gradients for negative inputs and improves training stability.                                |
| Slow convergence                           | Used Adam optimizer with `ReduceLROnPlateau` scheduler to automatically reduce learning rate when validation loss plateaus.                                    |
| Hard-to-detect defects (e.g., Donut vs. Loc) | Enhanced spatial pattern learning through deeper CNN layers and included MLP branch to learn non-spatial patterns directly from raw data.                      |
| Evaluation inconsistency                    | Adopted multiple metrics (Macro Precision, Recall, F1-score, Subset Accuracy) for more comprehensive performance analysis.                                   |
| Multi-label classification challenge       | Set the prediction threshold to 0.9 (instead of the default 0.5) to ensure higher precision, reflecting the strict quality requirements in wafer defect detection. A higher threshold reduces false positives, which is critical in maintaining manufacturing standards and minimizing unnecessary rework. |
