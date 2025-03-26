#Description: This script contains functions to evaluate the models and check their robustness.
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc

def evaluate_models(logit_model, probit_model, df_filtered: pd.DataFrame) -> None:
    """
    Print summaries and evaluate model performance with ROC.

    Args:
        logit_model: Trained logistic model
        probit_model: Trained probit model
        df_filtered (pd.DataFrame): Data used for evaluation
    """
    # Prepare the explanatory variables (independent variables)
    X = df_filtered[['Location', 'Gender', 'Age']].copy()
    # Encode gender as binary (1 = Male, 0 = Female) for regression
    X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})
    # Add constant term for intercept in the regression model
    X = sm.add_constant(X)

    # Regression results
    # ------------------------------
    print("Logit regression summary :")
    print(logit_model.summary())

    print("Probit regression summary :")
    print(probit_model.summary())

    # Statistically significant variables
    # ------------------------------
    print("Statistically significant variables (p < 0.05) - Logit:")
    print(logit_model.pvalues[logit_model.pvalues < 0.05])

    print("Statistically significant variables (p < 0.05) - Probit:")
    print(probit_model.pvalues[probit_model.pvalues < 0.05])

    # ROC curve
    # ------------------------------

    # Define the binary target variable: 1 = Success (Pass rate > 50%), 0 = Failure (Pass rate <= 50%)
    df_filtered['Success'] = (df_filtered['Passes'] / df_filtered['Conducted'] > 0.5).astype(int)
    # Generate predictions from the logit model
    logit_preds = logit_model.predict(X)
    # Calculate true positive rate and false positive rate for ROC curve
    fpr, tpr, _ = roc_curve(df_filtered['Success'], logit_preds)
    roc_auc = auc(fpr, tpr)

    # Plot the ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve Logit (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC Curve - Logit model')
    plt.legend()
    plt.show()

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def predict_probabilities(logit_model) -> None:
    """
    Predict pass probabilities for different candidate profiles.
    """
    print("**Probability predictions**")
    
    # Define test candidate scenarios
    # Each row represents a profile with:
    # - const: intercept term
    # - Location: 1 = Bletchley, 0 = Wood Green
    # - Gender: 1 = Male, 0 = Female
    # - Age: fixed at 22 for this example
    # ----------------------------------------
    test_cases = pd.DataFrame({
        'const': [1, 1, 1, 1],  # Intercept term
        'Location': [1, 0, 1, 0],  # Test center
        'Gender': [1, 1, 0, 0],    # Gender
        'Age': [22, 22, 22, 22]    # Age of candidate
    })
    # Generate predictions using the model
    predictions = logit_model.predict(test_cases)
    # Add predicted probabilities to the test cases
    test_cases['Predicted_Prob'] = predictions
    # Display results in a readable format
    for _, row in test_cases.iterrows():
        location = 'Bletchley' if row['Location'] == 1 else 'Wood Green'
        gender = 'Male' if row['Gender'] == 1 else 'Female'
        print(f"Probability of passing for a {row['Age']}-year-old {gender} at {location}: {row['Predicted_Prob']:.2%}")

def robustness_check(df_filtered: pd.DataFrame, logit_model) -> None:
    """
    Run logistic regression on 80% subsample and compare coefficients.
    """
    print("**Robustness test**")

    # Step 1: draw a random 80% subsample of the dataset
    df_sample = df_filtered.sample(frac=0.8, random_state=42)

    # Step 2: Prepare the predictors (X) and response (y)
    # I ensure Gender is correctly encoded as 0/1 if still in string format
    X_sample = df_sample[['Location', 'Gender', 'Age']].copy()
    if X_sample['Gender'].dtype == 'object':
        X_sample['Gender'] = X_sample['Gender'].map({'Male': 1, 'Female': 0})

    # Constant term (intercept)
    X_sample = sm.add_constant(X_sample)

    # Convert all data to numeric (just in case)
    X_sample = X_sample.apply(pd.to_numeric, errors='coerce')
    y_sample = df_sample[['Passes', 'Failures']].apply(pd.to_numeric, errors='coerce')

    # Step 3: drop rows with missing values
    valid_rows = X_sample.notna().all(axis=1) & y_sample.notna().all(axis=1)
    X_sample = X_sample[valid_rows]
    y_sample = y_sample[valid_rows]

    # Step 4: fit logistic regression model on subsample
    logit_model_sample = sm.GLM(y_sample, X_sample, family=sm.families.Binomial()).fit()

    # Step 5: compare coefficients from original and re-estimated model
    print("Coefficient comparison (original vs re-estimated):")
    comparison = pd.DataFrame({
        "Original": logit_model.params,
        "Re-estimated": logit_model_sample.params
    })

    print(comparison)