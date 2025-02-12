# Wine Quality Analysis Using SVM and KNN

## Overview
This project implements machine learning models to predict wine quality based on physicochemical properties. The analysis uses a binary classification approach where wines are classified as either high quality (rating >= 7) or standard quality (rating < 7).

## Features
- Binary classification of wine quality
- Comparison of SVM and KNN models
- Implementation of preprocessing pipeline
- Hyperparameter tuning using GridSearchCV
- Model evaluation with various metrics

## Dataset
The dataset used is the Wine Quality Dataset containing various chemical properties of red wines:
- Fixed acidity
- Volatile acidity
- Citric acid
- Residual sugar
- Chlorides
- Free sulfur dioxide
- Total sulfur dioxide
- Density
- pH
- Sulphates
- Alcohol
- Quality (target variable)

## Technical Implementation

### Dependencies
```python
numpy
pandas
matplotlib
seaborn
scikit-learn
```

### Data Preprocessing
- Data cleaning and handling missing values
- Feature standardization using StandardScaler
- Pipeline implementation for preprocessing steps

### Machine Learning Models
1. Support Vector Machine (SVM)
   - Kernel: RBF
   - Hyperparameter tuning for:
     - C: [0.1, 1, 10]
     - gamma: [0.1, 0.1, 1]
   - Probability estimation enabled

2. K-Nearest Neighbors (KNN)
   - Hyperparameter tuning for:
     - n_neighbors: [3, 5, 7]
     - weights: ['uniform', 'distance']

### Model Evaluation Features
- Classification reports showing:
  - Precision
  - Recall
  - F1-score
  - Support
- Confusion matrix visualization using seaborn heatmaps
- Cross-validation during hyperparameter tuning

## Results
Both models showed strong performance with:
- SVM achieving 92% accuracy
  - High precision (0.93) for standard quality wines
  - Good precision (0.88) for high-quality wines
- KNN also achieving 92% accuracy
  - High precision (0.94) for standard quality wines
  - Moderate precision (0.76) for high-quality wines

## Implementation Details

### Data Pipeline
```python
numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features)
])
```

### Model Pipelines
```python
svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', SVC(kernel='rbf', probability=True))
])

knn_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])
```

## Visualization Features
- Confusion matrix heatmaps
  - Blues colormap for SVM results
  - Greens colormap for KNN results
- Annotated values in confusion matrices

## Future Improvements
- Feature importance analysis
- Implementation of additional models
- Hyperparameter tuning with broader parameter ranges
- Advanced visualization of decision boundaries
- Cross-validation analysis visualization
