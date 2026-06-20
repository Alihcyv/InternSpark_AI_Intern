# Credit Risk Classification

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)
![CatBoost](https/img.shields.io/badge/CatBoost-Latest-yellow.svg)
![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blueviolet.svg)

## Project Overview
This project focuses on building a high-performance machine learning pipeline to predict **Credit Risk**. The primary objective is to analyze customer financial data and predict the likelihood of a loan default (Binary Classification). By identifying high-risk applicants, financial institutions can minimize losses and optimize their lending strategies.

## Tech Stack & Tools
- **Language:** Python
- **Data Manipulation:** `Pandas`, `NumPy`
- **Visualization:** `Matplotlib`, `Seaborn`
- **ML Frameworks:** 
  - `Scikit-Learn` (Preprocessing, Baseline Models)
  - `XGBoost`, `CatBoost`, `LightGBM` (Gradient Boosting)
- **Environment:** Jupyter Notebook

## Machine Learning Pipeline

### 1. Exploratory Data Analysis (EDA) & Preprocessing
To ensure data quality and model robustness, the following preprocessing steps were implemented:
*   **Data Cleaning:** Conducted a thorough analysis of missing values and feature distributions.
*   **Advanced Imputation:** Used **`KNNImputer`** ($k=5$) to handle missing values. This approach preserves the data distribution better than simple mean/median imputation by leveraging the similarity between samples.

  $$\hat{x}_i = \frac{1}{k} \sum_{j \in \mathcal{N}_k(i)} x_j$$

*   **Feature Scaling:** Applied **`StandardScaler`** to normalize numerical features, ensuring that distance-based models (like SVC or KNN) perform optimally.
*   **Categorical Encoding:** Utilized **`OneHotEncoder`** to transform categorical variables into a machine-readable format.

### 2. Modeling Strategy
I employed a multi-stage modeling approach, moving from simple baselines to complex ensembles to maximize predictive power.

#### A. Baseline Models
Evaluated several classifiers to establish a performance benchmark:
- Logistic Regression
- Random Forest
- Gradient Boosting
- LightGBM & XGBoost
- Support Vector Classifier (SVC)

#### B. Ensemble Learning (Voting)
To reduce variance and improve stability, I implemented a **`VotingClassifier`** with **`voting='soft'`**. This method aggregates the predicted probabilities of multiple models, which typically leads to better generalization than a single model.

$$\hat{y} = \text{argmax} \sum_{i=1}^{n} w_i P(y=c | x, \mathcal{M}_i)$$

#### C. Advanced Stacking (Meta-Learning)
The final and most powerful model was a **`StackingClassifier`**, which uses terms of a two-layer architecture:
- **Base Learners:** `ExtraTreesClassifier`, `XGBClassifier`, and `CatBoostClassifier`.
- **Meta-Learner:** `RandomForestClassifier`.
- **Mechanism:** The meta-model learns how to optimally combine the predictions of the base learners using **Out-of-Fold (OOF)** predictions via 5-fold cross-validation.

### 3. Evaluation Metrics
Since credit risk is a highly imbalanced problem where a **False Negative** (predicting a default as a good loan) is much more costly than a False Positive, the following metrics were prioritized:
- **ROC-AUC Score**: To evaluate terms of the model's ability to distinguish between classes.
- **F1-Score**: To balance Precision and Recall.
- **Recall**: Specifically monitored to ensure a high detection rate of potential defaulters.
- **Confusion Matrix**: To visualize the trade-off between Type I and Type II errors.

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/credit-risk-classification.git
