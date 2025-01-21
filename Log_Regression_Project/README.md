# Logistic Regression Project: Telco Customer Churn Prediction

## Overview
This project involves building a **logistic regression model** to predict customer churn using the **Telco Customer Churn** dataset. The project explores how various customer attributes influence the likelihood of churn and evaluates the model's performance using standard classification metrics.

## Project Objectives
- Predict customer churn based on input features.
- Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.
- Provide actionable insights into key factors affecting churn.

## Dataset
- **Source**: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Repository**: Kaggle
- **Description**: 
  - The dataset contains **7,043 rows** and **21 columns**, including customer demographics, account information, and subscription details.
  - The target variable, `Churn`, indicates whether the customer has left the company (`Yes`) or not (`No`).

## Project Contents
The repository includes the following:
- `notebooks/Regression.ipynb`: A Jupyter Notebook implementing the logistic regression model with step-by-step code, analysis, and visualizations.
- `requirements.txt`: A file listing Python dependencies required to run the project.
- `.gitignore`: Configuration to exclude unnecessary files from the repository.

## Key Features
1. **Data Preprocessing**:
   - Handled missing values and outliers.
   - Converted categorical variables into numerical format using encoding techniques.
2. **Model Training**:
   - Used `LogisticRegression` from `scikit-learn` to train the model.
   - Split the data into training and testing sets.
3. **Model Evaluation**:
   - Generated confusion matrix and classification report.
   - Calculated metrics: accuracy, precision, recall, and F1-score.
4. **Visualizations**:
   - Visualized churn distribution and feature importance.

## Results
1. Model 1
    - **Accuracy**: 79%
    - **Precision**: 0.78 (weighted average)
    - **Recall**: 0.79 
    - **F1-score**: 0.78 (Weighted Average)

1. Model 2
    - **Accuracy**: 80%
    - **Precision**: 0.79 (Weighted Average)
    - **Recall**: 0.80 
    - **F1-score**: 0.79 (Weighted Average )

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/telco-churn-logistic-regression.git

2. Navigate to the project directory:
    cd telco-churn-logistic-regression

3. Install dependecies 
    pip install -r requirements.txt

4. Open the notebook in jupyter 
    pip install -r requirements.txt

## Future Work 
1. Test the logistic regression model against other machine learning algorithms (e.g., Random Forest, Gradient Boosting).
2. Perform hyperparameter tuning to improve performance.
3. Explore feature engineering to enhance model accuracy.

** This Project is LIcensed under the MIT License ** 

## Author 
- Name : Busari Abdulmuiz Olakunle 
- username : ittztint 
