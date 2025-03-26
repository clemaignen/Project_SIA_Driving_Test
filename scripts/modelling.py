# Description: This script contains the function to train the Logit and Probit regression models.
import statsmodels.api as sm
import pandas as pd

def train_models(df_filtered):
    """
    Trains Logit and Probit regression models using Location, Gender, and Age as predictors.

    Args:
        df_filtered (pd.DataFrame): Preprocessed dataset with columns Location, Gender, Age, Passes, Failures.

    Returns:
        tuple: Fitted Logit and Probit models.
    """
    # 1. Define explanatory variables (X) and target (y)
    X = df_filtered[['Location', 'Gender', 'Age']].copy()
    X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})  # Encode gender as binary

    y = df_filtered[['Passes', 'Failures']].copy()

    # 2. Ensure numeric types and drop incomplete rows
    X = X.apply(pd.to_numeric, errors='coerce')
    y = y.apply(pd.to_numeric, errors='coerce')

    df_model = pd.concat([X, y], axis=1).dropna()
    X_clean = sm.add_constant(df_model[['Location', 'Gender', 'Age']])
    y_clean = df_model[['Passes', 'Failures']]

    # 3. Fit Logit and Probit models
    logit_model = sm.GLM(y_clean, X_clean, family=sm.families.Binomial()).fit()
    probit_model = sm.GLM(y_clean, X_clean, family=sm.families.Binomial(link=sm.families.links.Probit())).fit()

    return logit_model, probit_model