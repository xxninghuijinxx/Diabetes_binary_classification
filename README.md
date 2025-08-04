# 🧬 Diabetes Binary Classification Project

> Predicting diabetes in patients using XGBoost and a custom Neural Network  
> 🗂️ Includes preprocessing, visualization, modeling, and evaluation.

---

## 📌 Project Description

This project applies machine learning techniques to perform **binary classification** on the **Pima Indians Diabetes Database**. We explored both traditional tree-based models and deep learning approaches to predict whether a patient is likely to have diabetes, based on medical diagnostic features.

The workflow includes:

- 🧹 Data cleaning & preprocessing  
- 📊 Exploratory data analysis (EDA) & visualization  
- 🧠 Dimensionality reduction with PCA  
- 🌲 Tree-based models (XGBoost, Random Forest, Decision Tree)  
- 🔥 PyTorch-based neural network  
- 📈 Model evaluation on test data

---

## 🎯 Objective

To accurately classify whether a patient has diabetes using machine learning models based on clinical data. The ultimate goal is to support early diagnosis and intervention.

---

## 📂 Dataset Information

- **Name**: *Pima Indians Diabetes Database*  
- **Source**: National Institute of Diabetes and Digestive and Kidney Diseases  
- **Download**: [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)  
- **Type**: Supervised learning, binary classification  
- **Features**: Medical attributes like glucose level, BMI, age, etc.  
- **Target**: `Outcome` — 0 (non-diabetic) or 1 (diabetic)

---

## 🛠️ Tech Stack

- **Data handling**: `numpy`, `pandas`  
- **Visualization**: `matplotlib`, `seaborn`  
- **Preprocessing & modeling**: `scikit-learn`, `xgboost`  
- **Deep learning**: `torch` (PyTorch)

---

## 🚀 How to Run

> 💡 This project is implemented in **Jupyter Notebook**

### 🧰 Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
Install dependencies:


pip install -r requirements.txt
Launch Jupyter Notebook and open the .ipynb file:


jupyter notebook
📊 Results
✅ Achieved ~80% accuracy on the test set

📌 Evaluated using:

Accuracy

Precision / Recall / F1-score

Confusion Matrix

📉 PCA and EDA revealed strong feature relationships and class imbalances

🙏 Acknowledgements
Special thanks to the National Institute of Diabetes and Digestive and Kidney Diseases for providing the dataset.

📄 License
This project is licensed under the terms defined in the LICENSE file.
