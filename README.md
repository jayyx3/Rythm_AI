# Rhythm_AI - Arrhythmia Classification using Machine Learning

<div align="center">
  <img src="Image/arrhythmia.jpg" alt="Arrhythmia Detection" width="600"/>
  
  [![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
  [![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-green.svg)](https://scikit-learn.org/)
  [![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
</div>

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Machine Learning Models](#machine-learning-models)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🔍 Overview

**Rhythm_AI** is a comprehensive machine learning project designed to classify cardiac arrhythmias using advanced data preprocessing, exploratory data analysis (EDA), and multiple classification algorithms. The project leverages Principal Component Analysis (PCA) for dimensionality reduction and implements various techniques to handle class imbalance through oversampling methods.

### Key Objectives:
- Predict whether a person is suffering from arrhythmia
- Classify arrhythmia into one of 12 distinct categories
- Handle high-dimensional medical data (279 features)
- Address class imbalance in the dataset
- Achieve optimal model performance through hyperparameter tuning

## 📊 Dataset

The project uses the **Arrhythmia Dataset** from the UCI Machine Learning Repository.

**Dataset Characteristics:**
- **Source:** [UCI Machine Learning Repository - Arrhythmia Dataset](https://archive.ics.uci.edu/ml/datasets/Arrhythmia)
- **Instances:** 452 examples
- **Features:** 279 attributes including:
  - Patient demographics (age, sex, weight, height)
  - ECG measurements and cardiac features
  - Various physiological parameters
- **Classes:** 16 classes total
  - 1 Normal class (245 instances)
  - 12 Arrhythmia types
  - 3 Additional categories

**Most Representative Arrhythmia Types:**
- Coronary Artery Disease
- Right Bundle Branch Block

**Challenges:**
- High feature-to-sample ratio (279 features vs 452 samples)
- Significant class imbalance
- Missing values in the dataset

## 📁 Project Structure

```
Rhythm_AI/
│
├── Data/
│   └── arrhythmia.csv                    # Original dataset
│
├── Preprocessing and EDA/
│   ├── Data preprocessing.ipynb          # Data cleaning and preprocessing
│   └── EDA.ipynb                         # Exploratory data analysis
│
├── Model/
│   ├── general and pca.ipynb             # General models with PCA
│   └── oversampled and pca.ipynb         # Models with oversampled data
│
├── Image/
│   ├── arrhythmia.jpg                    # Project banner image
│   ├── age.png                           # Age distribution visualization
│   ├── sex.png                           # Gender distribution
│   ├── weight.png                        # Weight distribution
│   ├── height.png                        # Height distribution
│   ├── QRS duration.png                  # QRS duration analysis
│   ├── missingvalue.png                  # Missing values visualization
│   ├── Histogram class distribution.png  # Class distribution
│   ├── class wise visualization.png      # Class-wise analysis
│   └── pairwiseGraph.png                 # Feature pairwise relationships
│
├── final with pca.ipynb                  # Complete end-to-end pipeline
├── new data with target class.csv        # Processed dataset
├── requirements.txt                      # Python dependencies
└── README.md                             # Project documentation

```

## ✨ Features

### Data Preprocessing
- Missing value detection and imputation
- Handling '?' placeholders with appropriate strategies
- Feature scaling and normalization
- Outlier detection and treatment

### Exploratory Data Analysis (EDA)
- Comprehensive statistical analysis
- Distribution visualization of key features
- Correlation analysis
- Class imbalance visualization
- Missing data pattern analysis

### Dimensionality Reduction
- Principal Component Analysis (PCA) implementation
- Variance explained analysis
- Optimal component selection
- Feature importance evaluation

### Model Implementation
Multiple classification algorithms:
- **Logistic Regression**
- **Decision Tree Classifier**
- **Random Forest Classifier**
- **Support Vector Machine (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Gradient Boosting Classifier**
- **XGBoost Classifier**

### Class Imbalance Handling
- SMOTE (Synthetic Minority Over-sampling Technique)
- Random Oversampling
- Class weight adjustment

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook or JupyterLab

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jayyx3/Rythm_AI.git
   cd Rythm_AI
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   ```

## 💻 Usage

### Running the Complete Pipeline

1. **Data Preprocessing:**
   ```bash
   # Open and run
   Preprocessing and EDA/Data preprocessing.ipynb
   ```

2. **Exploratory Data Analysis:**
   ```bash
   # Open and run
   Preprocessing and EDA/EDA.ipynb
   ```

3. **Model Training and Evaluation:**
   ```bash
   # For general models
   Model/general and pca.ipynb
   
   # For oversampled models
   Model/oversampled and pca.ipynb
   ```

4. **Complete End-to-End Workflow:**
   ```bash
   # Run the comprehensive pipeline
   final with pca.ipynb
   ```

### Quick Start Example

```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('Data/arrhythmia.csv', header=None)

# Preprocessing
# ... (refer to notebooks for complete code)

# Train model
# ... (refer to notebooks for complete code)
```

## 🤖 Machine Learning Models

### Model Performance Comparison

The project evaluates multiple classification algorithms:

| Model | Description | Key Parameters |
|-------|-------------|----------------|
| **Logistic Regression** | Linear classification model | Regularization (L1/L2), Solver |
| **Decision Tree** | Tree-based classifier | Max depth, Min samples split |
| **Random Forest** | Ensemble of decision trees | Number of estimators, Max features |
| **SVM** | Support Vector Machine | Kernel type, C parameter, Gamma |
| **KNN** | K-Nearest Neighbors | Number of neighbors, Distance metric |
| **Gradient Boosting** | Sequential ensemble method | Learning rate, Number of estimators |
| **XGBoost** | Optimized gradient boosting | Learning rate, Max depth, Subsample |

### Model Evaluation Metrics
- **Accuracy:** Overall prediction correctness
- **Precision:** True positive rate per class
- **Recall:** Sensitivity per class
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Detailed class-wise performance
- **ROC-AUC Score:** Area under the ROC curve

## 📈 Results

The project implements comprehensive model evaluation with:
- Cross-validation for robust performance estimation
- Hyperparameter tuning using Grid Search/Random Search
- Feature importance analysis
- Model comparison visualizations
- Confusion matrix heatmaps
- ROC curves and precision-recall curves

*Note: Detailed results and performance metrics are available in the respective notebook files.*

## 🛠️ Technologies Used

### Core Libraries
- **Python 3.8+:** Primary programming language
- **Pandas:** Data manipulation and analysis
- **NumPy:** Numerical computing
- **SciPy:** Scientific computing

### Machine Learning
- **scikit-learn:** ML algorithms and utilities
- **imbalanced-learn:** Handling imbalanced datasets
- **scikit-plot:** ML visualization

### Data Visualization
- **Matplotlib:** Basic plotting
- **Seaborn:** Statistical visualizations

### Development Environment
- **Jupyter Notebook:** Interactive development
- **Git:** Version control

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes:**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch:**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Contribution Guidelines
- Follow PEP 8 style guidelines for Python code
- Add comments and docstrings to your code
- Update documentation for new features
- Test your changes thoroughly

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👤 Contact

**Project Maintainer:** jayyx3

- GitHub: [@jayyx3](https://github.com/jayyx3)
- Project Link: [https://github.com/jayyx3/Rythm_AI](https://github.com/jayyx3/Rythm_AI)

## 🙏 Acknowledgments

- UCI Machine Learning Repository for providing the Arrhythmia dataset
- scikit-learn community for excellent documentation and tools
- All contributors and supporters of this project

## 📚 References

1. [UCI Arrhythmia Dataset](https://archive.ics.uci.edu/ml/datasets/Arrhythmia)
2. [scikit-learn Documentation](https://scikit-learn.org/stable/)
3. [Imbalanced-learn Documentation](https://imbalanced-learn.org/)
4. [PCA for Dimensionality Reduction](https://scikit-learn.org/stable/modules/decomposition.html#pca)

---

<div align="center">
  Made with ❤️ for advancing healthcare through AI
  
  ⭐ Star this repository if you found it helpful!
</div>
