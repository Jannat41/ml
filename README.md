# 🧬 Breast Cancer Classification Using Logistic Regression

> ⚠️ **Disclaimer:** This is a classroom/educational model and must **NOT** be used for real medical decisions.

---

## 📋 Project Overview

A beginner machine learning project by **Team Dataflux** (CSE 12th Batch) that classifies breast cancer tumors as **Malignant (M)** or **Benign (B)** using Logistic Regression.

The full ML workflow is covered end-to-end:

```
Data Collection → Inspection → Preprocessing → EDA → Train-Test Split → Model Training → Evaluation → Interpretation
```

---

## 📊 Dataset

**Breast Cancer Wisconsin (Diagnostic)** — UCI ML Repository  
🔗 https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

| Property         | Value                          |
|------------------|-------------------------------|
| Rows             | 569                            |
| Input Features   | 30 (all `float64`)             |
| Target Column    | `Diagnosis` (M = Malignant, B = Benign) |
| Missing Values   | None                           |
| Duplicate Rows   | None                           |

**Class Distribution:**

| Class     | Count | Percentage |
|-----------|-------|------------|
| Benign (B)    | 357   | 62.74%     |
| Malignant (M) | 212   | 37.26%     |

---

## 🔧 Tech Stack

| Tool           | Version  |
|----------------|----------|
| Python         | 3.13.18  |
| pandas         | 2.2.3    |
| numpy          | 2.4.6    |
| scikit-learn   | 1.6.1    |
| matplotlib     | latest   |
| seaborn        | latest   |

---

## 🗂️ Project Workflow

### 1. Setup & Imports
Installs `ucimlrepo`, imports all required libraries, and sets global configuration (random seed, plot style, output directories).

### 2. Load the Dataset
Fetches the dataset from the UCI ML Repository via `ucimlrepo`. Falls back to scikit-learn's built-in copy if the download fails.

### 3. Data Inspection
- Shape, data types, missing values, duplicates
- Class distribution (count + percentage)
- Descriptive statistics (`df.describe()`)

### 4. Exploratory Data Analysis (EDA)
- **Fig. 1:** Class distribution bar chart
- **Fig. 2:** Feature distributions by diagnosis (histograms / box plots)
- Correlation analysis and heatmap
- Feature importance insights

### 5. Preprocessing
- Label encoding: `M → 1`, `B → 0`
- Feature scaling with `StandardScaler`
- Train-test split (stratified, 80/20)

### 6. Model Training
- `sklearn.linear_model.LogisticRegression` wrapped in a `Pipeline`
- Hyperparameter tuning with `GridSearchCV` + `StratifiedKFold`

### 7. Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix
- ROC Curve & AUC Score
- Full `classification_report`

### 8. Interpretation
- Model coefficients analysis
- Most influential features for prediction

---

## 📁 Repository Structure

```
📦 ml/
 ┣ 📓 Breast_Cancer_Logistic_Regression.ipynb   ← Main notebook
 ┗ 📄 README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install ucimlrepo scikit-learn pandas numpy matplotlib seaborn
```

### Run the Notebook
1. Clone this repository:
   ```bash
   git clone https://github.com/Jannat41/ml.git
   cd ml
   ```
2. Open the notebook:
   ```bash
   jupyter notebook Breast_Cancer_Logistic_Regression.ipynb
   ```
3. Run all cells in order (Kernel → Restart & Run All).

---

## 📌 Notes

- Figures are saved to a `figures/` directory when running locally.
- The notebook was developed and tested on **Google Colab**.
- The dataset is loaded automatically from the UCI repository; no manual download needed.
