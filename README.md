
# American Express Credit Card Payment Risk Forecasting Project

This repository contains code and resources for the **American Express Credit Card Payment Risk Forecasting Project**. The goal of this project is to predict credit card payment risk for American Express customers using machine learning models and real-world data.

## Project Objective

The objective of this project is to forecast the risk of credit card payment defaults by analyzing various customer features and transaction histories. 

The project involves the following key tasks:

- Data preprocessing (handling missing values, encoding categorical data)
- Feature correlation analysis
- Building and evaluating machine learning models for risk prediction
- Final model submission based on test data predictions

## Project Setup

### Prerequisites

To run this project, you'll need to have the following software installed:

- **Python 3.8+**
- **Anaconda/Miniconda** (optional but recommended)
- **Jupyter Notebook** (for running the notebook in the development phase)
- Libraries such as `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, etc.

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/credit-card-risk-forecasting.git
   cd credit-card-risk-forecasting
   ```

2. Set up a virtual environment (optional but recommended):
   ```bash
   conda create -n amex-risk python=3.8
   conda activate amex-risk
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Steps

### 1. Data Loading and Initial Exploration

We load the dataset containing customer credit card transaction data. This includes features like transaction histories, balances, and credit scores.

### 2. Data Preprocessing

- **Handling missing values**: We use the `IterativeImputer` from the `scikit-learn` library to fill missing values in the dataset. This method allows us to impute missing data by predicting them based on other features.
- **Encoding categorical features**: Categorical variables are either encoded as numerical values using one-hot encoding or excluded for correlation analysis.

```python
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Handling missing values
imputer = IterativeImputer()
train = imputer.fit_transform(train)
```

### 3. Feature Correlation Analysis

Since the dataset contains both numerical and categorical data, correlation analysis is performed only on numerical features to identify important relationships with the target variable (credit card payment risk).

```python
# Select only numeric columns for correlation analysis
train_numeric = train.select_dtypes(include=['number'])
corr = train_numeric.corr()
```

### 4. Building Machine Learning Models

- We utilize models such as Random Forest, Decision Trees, and Logistic Regression to forecast credit card payment risk.
- The target variable represents the likelihood of a customer defaulting on a payment.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(train_numeric, target, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier()
model.fit(X_train, y_train)
```

### 5. Model Evaluation

Evaluate the model performance using metrics such as **AUC-ROC** and **Accuracy**. This helps determine how well the model is predicting credit card payment risk.

```python
from sklearn.metrics import roc_auc_score, accuracy_score

# Make predictions and evaluate the model
y_pred = model.predict(X_test)
roc_score = roc_auc_score(y_test, y_pred)
print(f"ROC-AUC Score: {roc_score}")
```

### 6. Final Submission

Once the model is trained and evaluated, predictions are made on the test dataset, and the results are prepared for submission.

```python
# Make predictions on the test set
y_pred_test = model.predict_proba(test)[:, 1]

# Create a submission file
submission_df = pd.DataFrame({'prediction': y_pred_test})
submission_df.to_csv('submission.csv', index=False)
```

## Project Files

- `American_Express_Project_with_Improved_Comments_and_Code.py`: Python script with code and detailed comments.
- `American_Express_Project_Cleaned.py`: Cleaned version of the Python script.
- `submission.csv`: Final submission file with predictions.
- `requirements.txt`: List of dependencies.

## Future Work

1. **Feature Engineering**: Enhance features to improve model accuracy.
2. **Hyperparameter Tuning**: Optimize model performance by tuning hyperparameters.
3. **Deployment**: Deploy the trained model to a real-time environment using Flask or FastAPI.

## Conclusion

This project provides a full workflow for forecasting credit card payment risk using machine learning techniques. Through careful preprocessing, feature selection, and model evaluation, we can build an accurate predictive model to assist American Express in minimizing credit card payment risks.
