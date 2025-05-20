![Repo Header](./assets/images/repo_header.png)

# Pneumonia-Detection

This project explores various traditional machine learning and deep learning approaches for detecting pneumonia in chest X-rays. It achieved up to **92.7% accuracy** and **94.4% F1-score** with DenseNet. For model interpretability, EigenCAM is employed to highlight the key regions on the X-rays contributing to the models' predictions.

> An sklearn-like classifier for each pretrained deep learning model is also developed to provide a more scalable, cohesive and easier-to-use interface for model fine-tuning and inferencing.

## Results

Deep learning models are the top performers in the project, easily out-performing traditional machine learning models by at least 10%. Convolution-based deep learning models, in particular, top the entire list.

| Classifier             | Accuracy | F1    | Precision | Recall | ROC-AUC |
| :--------------------- | :------: | :---: | :-------: | :----: | :-----: |
| DenseNet               | 0.927    | 0.944 | 0.909     | 0.982  | 0.954   |
| ResNet                 | 0.916    | 0.935 | 0.900     | 0.974  | 0.968   |
| EfficientNet           | 0.889    | 0.916 | 0.867     | 0.971  | 0.955   |
| ConvNeXt               | 0.881    | 0.912 | 0.848     | 0.987  | 0.965   |
| Vision Transformer     | 0.868    | 0.896 | 0.885     | 0.907  | 0.934   |
| Swin Transformer       | 0.847    | 0.889 | 0.815     | 0.976  | 0.951   |
| Multilayer Perceptron  | 0.756    | 0.832 | 0.730     | 0.966  | 0.888   |
| K-Nearest Neighbor     | 0.746    | 0.824 | 0.728     | 0.948  | 0.861   |
| Support Vector Machine | 0.743    | 0.825 | 0.717     | 0.971  | 0.845   |
| XGBoost                | 0.741    | 0.824 | 0.716     | 0.971  | 0.872   |
| Logistic Regression    | 0.741    | 0.804 | 0.764     | 0.848  | 0.818   |
| Random Forest          | 0.727    | 0.816 | 0.705     | 0.969  | 0.853   |
| AdaBoost               | 0.724    | 0.810 | 0.710     | 0.943  | 0.813   |
| Decision Tree          | 0.711    | 0.797 | 0.710     | 0.907  | 0.691   |
| Naive Bayes            | 0.564    | 0.498 | 0.888     | 0.346  | 0.750   |

> [!NOTE]
> Deep learning models include DenseNet, ResNet, EfficientNet, ConvNeXt, Vision Transformer, and Swin Transformer. The first four being convolution-based, while the later two being transformer-based.

### Model Interpretation with EigenCAM

All visualizations showed that the deep learning models are capable of identifying key patterns surrounding the lung area.
![EigenCAM Sample](./assets/images/eigencam_by_model.png)

## Overview

More than 2 million children under the age of 5 die from pneumonia each year, accounting for nearly one in five deaths among children under 5 worldwide. However, patients who receive treatment within 3 days of admission have a significantly higher survival rate than those treated later. This project aims to provide a stable and accurate model that helps reduce human errors in diagnosis and increase child survival rates.

## Dateset

- Source: [Kaggle - Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- Information:
  - Provider: Guangzhou Women and Children’s Medical Center
  - Age: 1 ~ 5 years old
  - Size: 5,863 images
  - Distribution:
    ![Dataset Distribution](./assets/images/dataset_distribution.png)

## Preprocessing

Images will first be checked if they are in grayscale and stored in a single channel, then uniformly resized to 224 x 224 px and standardized before any model-specific preprocessing.

### Feature Extraction

For traditional machine learning models, training on images of size 224 x 224 px may take an impractical amount of time. Therefore, HOG and LBP features extracted from the original image will be used to train the models instead.

- HOG: Captures edge orientations and gradient structures, which can highlight lung boundary irregularities and consolidation patterns

- LBP: Captures local texture patterns, highlighting distinctive texture changes found in infected areas.

![Feature Extraction: HOG, LBP](./assets/images/feature_extraction.png)

### Image Transformation

To further improve the performance of deep learning models, stability plays a crucial role. Random transformations are applied to the images during the training phase to reduce model's vulnerability to factors such as patient posture, body size, X-ray quality, and others.

```python
transforms.Compose([
  transforms.RandomHorizontalFlip(),
  transforms.RandomResizedCrop(size=(224, 224), scale=(0.95, 1)),
  transforms.RandomRotation(5),
  transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
  transforms.RandomPerspective(distortion_scale=0.05),
])
```
## Modeling

### Traditional Model Hyperparameter Tuning

Hyperparameters of a model can impact its overall structure, training efficiency, and performance. This project leverages scikit-learn's `GridSearchCV` to tune hyperparameters of traditional models.

- Multilayer Perceptron: `hidden_layer_sizes`, `activation`, `solver`, `alpha`, `learning_rate`, `learning_rate_init`, `batch_size`, `early_stopping`, `validation_fraction`
- K-Nearest Neighbor: `n_neighbors`, `weights`, `algorithm`, `leaf_size`, `p`
- Support Vector Machine: `C`, `kernel`, `gamma`, `degree`, `coef0` 
- XGBoost: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `reg_alpha`, `reg_lambda`
- Logistic Regression: `penalty`, `C`, `solver`, `max_iter`
- Random Forest: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`, `bootstrap`
- AdaBoost: `n_estimators`, `learning_rate`
- Decision Tree: `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`
- Naive Bayes: `var_smoothing`

![Traditional Model Score](./assets/images/score_by_model(traditional).png)

### Deep Learning Model Fine-Tuning

Learning rate plays an important role in adjusting model parameters at each step of its optimization algorithm. However, defining a uniform learning rate that does not cause the model to diverge or get stuck in local minima remains a serious challenge. This is were the following technique can be helpful:

- **Dynamic Scheduling Learning Rate**: Fine-tuning the learning rate can help improve convergence and provide better generalization. For example, when the model's performance plateaus, the learning rate can be decreased to prevent overshooting.

Additionaly, deep learning models can easily overfit the training data, thus multiple techniques have been employed to address this issue:

1. **Transfer Learning**: A total of 5,863 images is far from enough to train deep learning models from scratch. This is where transfer learning comes into play. The weights from earlier layers, learned from a large dataset, will be frozen to capture important patterns in the images, and only the weights of later layers will be trained to fit our task.

2. **Loss Weighting**: Pneumonia X-ray images make up more than half of the training set. Therefore, the loss will be weighted to prevent the model from deviating toward predicting pneumonia instead of normal cases.

3. **Early Stopping**: A validation set is further split from the training set. During each epoch of the training process, this set will not be used to calculate loss gradients for backpropogation. Instead, it will only be used to evaluate whether the model continues to improve on unseen data.

![Deep Learning Model Score](./assets/images/score_by_model(deep).png)

## Evaluation
### Metrics

- F1-score: harmonic mean of precision and recall ($2 \times \frac{precision \times recall}{precision + recall}$)
- Precision: the proportion of all the model's positive classifications that are actually positive ($\frac{TP}{TP\ +\ FP}$)
- Recall:  the proportion of all actual positives that were classified correctly as positives ($\frac{TP}{TP +\ FN}$)
- Accuracy: the proportion of all classifications that were correct ($\frac{TP +\ TN}{TP +\ TN+\ FP+\ FN}$)

...[read more](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall)

### Confusion Matrix Analysis
Confusion matrix is a simple table used to measure how well a classification model is performing. It compares the predictions made by the model with the actual results and shows where the model was right or wrong. This helps understand where the model is making mistakes as further improvement guidance... [read more](https://www.geeksforgeeks.org/confusion-matrix-machine-learning/)

### ROC Curve Analysis
The ROC curve is drawn by calculating the true positive rate (TPR) and false positive rate (FPR) at every possible threshold, then graphing TPR over FPR. A perfect model, which at some threshold has a TPR of 1.0 and a FPR of 0.0, can be represented by either a point at (0, 1) if all other thresholds are ignored. [read more](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)

![Model ROC Curve](./assets/images/auc_by_model.png)

### Precision-Recall Curve Analysis
The precision-recall curve shows the tradeoff between precision and recall for different thresholds. A high area under the curve represents both high recall and high precision. [read more](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html)

## Visualization
Deep learning models are often viewed as black boxes because their internal decision making processes are difficult to understand. Tools like GradCAM and Eigen-CAM are able to solve this problem by creating a heatmap that can be overlaid on top of the input image to highlight which parts of the image induced the greatest magnitude of activation from the layers. [read more](https://www.datature.io/blog/understanding-your-yolov8-model-with-eigen-cam)

## Future Work

- Dataset Expansion: Increasing the dataset size may further improve overall model performance, as certain variations may not appear frequently in a small dataset.
- Bias & Fairness Analysis: Currently, the model is trained on X-rays of children aged 1 to 5 in Guangzhou. Further investigations into the model's performance across different age groups and ethnicities could be conducted.