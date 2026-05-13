# PCA Performance Optimizer

**Project-Based Learning - Machine Learning (10300401)**

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green.svg)

A web-based tool for understanding and visualizing how Principal Component Analysis (PCA) affects machine learning model performance.

---

## Overview

PCA Performance Optimizer helps you:
- Upload and preprocess datasets
- Apply PCA with adjustable components
- Train 7 different ML models
- Compare performance before and after PCA
- Visualize results with interactive charts

## Features

- **Auto Data Cleaning** - Handles text targets, missing values, too many classes
- **7 ML Models** - Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM, KNN, Naive Bayes
- **Comprehensive Metrics** - Accuracy, Precision, Recall, F1, Training Time, Inference Time, Memory
- **Interactive Visualizations** - Bar charts, confusion matrices, variance plots
- **Performance Comparison** - Side-by-side before/after PCA analysis
- **Recommendations** - Optimal PCA components based on variance explained

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run the App

```bash
streamlit run app.py
```

Open browser: `http://localhost:8501`

## Usage Guide

### Step 1: Upload Dataset
Click "Browse files" and select a CSV file with features and a target column.

### Step 2: Configure Target
Select the target column you want to predict. The app will auto-clean text columns and bin too-many-classes columns.

### Step 3: Preprocess Data
- Adjust test size (default 20%)
- Set random state for reproducibility
- Click "Preprocess & Split Data"

### Step 4: Apply PCA
- Set number of components (slider)
- Keep "Standardize Features" checked
- Click "Apply PCA"

### Step 5: Train Models
- Select models in sidebar (default: all)
- Click "Train All Models"

### Step 6: View Results
5 tabs available:
1. **Accuracy Comparison** - Bar chart comparing models
2. **Training & Inference Time** - Time performance metrics
3. **Memory Usage** - Memory comparison
4. **Confusion Matrices** - Per-model confusion matrices
5. **Variance Analysis** - PCA variance visualization

## Tech Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| Frontend | Streamlit | Web UI |
| Charts | Plotly, Matplotlib | Visualizations |
| ML | scikit-learn | Models & PCA |
| Data | pandas, numpy | Data handling |

## Project Structure

```
PCA/
├── app.py                    # Main Streamlit application
├── core/
│   ├── preprocessor.py       # Data loading & cleaning
│   ├── pca_analyzer.py       # PCA transformation
│   ├── model_trainer.py      # Model training & metrics
│   └── performance_comparator.py  # Results comparison
├── ui/
│   └── components.py         # UI components & charts
├── data/
│   ├── sample_classification.csv  # Sample dataset
│   └── sample_generator.py   # Dataset generator
├── docs/
│   ├── PCA_Documentation.docx  # Full documentation
│   └── DOCUMENTATION.md      # Markdown docs
└──requirements.txt
```

## Sample Datasets

A built-in sample dataset is included for testing:
```
data/sample_classification.csv
```

Also generates additional datasets using scikit-learn:
```python
from data.sample_generator import generate_classification_data
df = generate_classification_data(n_samples=500, n_features=15, n_classes=3)
df.to_csv('my_data.csv', index=False)
```

## Supported Models

| Model | Type | Best For | Speed |
|-------|------|----------|-------|
| Logistic Regression | Linear | Binary classification | Fast |
| Decision Tree | Tree | Feature importance | Fast |
| Random Forest | Ensemble | Complex data | Medium |
| Gradient Boosting | Ensemble | High accuracy | Slow |
| SVM | Kernel | High-dimensional | Slow |
| KNN | Instance | Simple baseline | Slow |
| Naive Bayes | Probabilistic | Text classification | Very Fast |

## Requirements

```
streamlit>=1.30.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.18.0
joblib>=1.3.0
psutil>=5.9.0
matplotlib>=3.7.0
```

## Dataset Requirements

- CSV file format
- Numeric or categorical features
- Target column with 2-50 unique values
- Less than 100,000 rows
- Fewer than 100 features

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Contains text data" | Auto-encoded by app |
| "Too many classes" | Auto-binned into ~10 categories |
| "Contains NaN" | Auto-filled with mean |
| Slider error | Dataset has only 1 feature |

## Documentation

Full documentation including screenshots, settings explained, and model details:
- [DOCUMENTATION.md](docs/DOCUMENTATION.md) - Markdown version
- [PCA_Documentation.docx](docs/PCA_Documentation.docx) - Word document

## License

MIT License

## Author

Project-Based Learning - Machine Learning (10300401)
