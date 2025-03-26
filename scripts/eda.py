# Description: This script contains the exploratory data analysis (EDA) function for the project.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def exploratory_data_analysis(df: pd.DataFrame) -> None:
    """
    Generate descriptive stats and visualizations for pass rate trends.

    Args:
        df (pd.DataFrame): Cleaned dataframe
    """
    print("**Descriptive statistics**")
    print(df.describe())

    # Histogram of total pass rates
    plt.figure(figsize=(6, 4))
    sns.histplot(df['PassRate_Total'], bins=15, kde=True, color='skyblue')
    plt.title("Distribution of total pass rates")
    plt.xlabel("Pass Rate (%)")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    # Boxplot: Pass rate by year
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Year', y='PassRate_Total', data=df, hue='Year', palette='coolwarm', legend=False)
    plt.title("Pass rate by year")
    plt.xlabel("Year")
    plt.ylabel("Pass rate (%)")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    # Boxplot: Pass rate by age
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Age', y='PassRate_Total', data=df, hue='Age', palette='coolwarm', legend=False)
    plt.title("Pass rate by age")
    plt.xlabel("Age")
    plt.ylabel("Pass rate (%)")
    plt.grid(True)
    plt.show()


    # Boxplot: Pass rate by gender (reshape the data)
    df_long = pd.melt(df, id_vars=['Location', 'Age', 'Year'],
                      value_vars=['PassRate_Male', 'PassRate_Female'],
                      var_name='Gender', value_name='Value')

    df_long['Gender'] = df_long['Gender'].replace({'PassRate_Male': 'Male', 'PassRate_Female': 'Female'})

    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Gender', y='Value', data=df_long, hue='Gender', palette='pastel', legend=False)
    plt.title("Pass rate by gender")
    plt.xlabel("Gender")
    plt.ylabel("Pass rate (%)")
    plt.grid(True)
    plt.show()

    # Boxplot: Pass rate by test center location
    plt.figure(figsize=(6, 4))
    sns.boxplot(x='Location', y='PassRate_Total', data=df, hue='Location', palette='Set2', legend=False)
    plt.title("Pass Rate by test center location")
    plt.xlabel("Location (0 = Wood Green, 1 = Bletchley)")
    plt.ylabel("Pass rate (%)")
    plt.grid(True)
    plt.show()