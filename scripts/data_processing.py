# Description: This script contains functions to load and preprocess the data for logistic regression analysis.
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and clean the Excel data.

    Args:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: Cleaned and formatted dataset.
    """
    # Define file path
    xls = pd.ExcelFile(file_path)

    sheets = [sheet for sheet in xls.sheet_names if sheet != "Notes"]
    # Initialize an empty data frame in which we will put all the extracted data we need from 'DVSA1203'
    data_frames = []

    # Load data from all available years, read the sheet within the dataset. Each sheet represents a year
    # Remove unnecessary rows at the top
    for year in range(len(sheets)):
        sheet_data = pd.read_excel(file_path, sheet_name=sheets[year], skiprows=6)
        sheet_data.columns = ["Location", "Age", "Conducted_Male", "Passes_Male", "PassRate_Male",
                              "Conducted_Female", "Passes_Female", "PassRate_Female", "Conducted_Total",
                              "Passes_Total", "PassRate_Total"]
        sheet_data['Location'] = sheet_data['Location'].ffill()
        sheet_data = sheet_data.dropna(how='all', subset=sheet_data.columns.difference(['Location']))
        sheet_data = sheet_data[sheet_data['Age'] != 'Total']
        sheet_data = sheet_data[sheet_data['Location'].isin(["Bletchley", "Wood Green (London)"])]
        sheet_data['Year'] = sheets[year]
        # Append the filtered data to the main data frame of interest
        data_frames.append(sheet_data)
    
    df = pd.concat(data_frames, ignore_index=True)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess dataset into clean long-format structure for modelling.

    Args:
        df (pd.DataFrame): Raw dataframe.

    Returns:
        pd.DataFrame: Processed dataframe in long format.
    """
    # Convert columns to numeric where needed
    to_num = ["Age", "Conducted_Male", "Passes_Male", "PassRate_Male",
              "Conducted_Female", "Passes_Female", "PassRate_Female",
              "Conducted_Total", "Passes_Total", "PassRate_Total"]
    df[to_num] = df[to_num].apply(pd.to_numeric, errors='coerce')

    # Encode location as binary (1 = Bletchley, 0 = Wood Green)
    pd.set_option('future.no_silent_downcasting', True)
    df['Location'] = df['Location'].replace({
        "Bletchley": 1,
        "Wood Green (London)": 0
    }).astype(int)

    # Ensure year is a string
    df['Year'] = df['Year'].astype(str)

    # Pivot to long format
    df_long = pd.melt(
        df,
        id_vars=['Location', 'Age', 'Year'],
        value_vars=['Conducted_Male', 'Passes_Male', 'Conducted_Female', 'Passes_Female'],
        var_name='Variable',
        value_name='Value'
    )

    # Extract Gender from the column names
    df_long['Gender'] = df_long['Variable'].apply(lambda x: 'Male' if 'Male' in x else 'Female')
    df_long['Measure'] = df_long['Variable'].apply(lambda x: 'Conducted' if 'Conducted' in x else 'Passes')

    # Pivot again: geting conducted / passes in columns
    df_pivot = df_long.pivot_table(
        index=['Location', 'Age', 'Year', 'Gender'],
        columns='Measure',
        values='Value',
        aggfunc='sum'
    ).reset_index()

    # Calculate passrate and failures
    df_pivot['PassRate'] = 100 * df_pivot['Passes'] / df_pivot['Conducted']
    df_pivot['Failures'] = df_pivot['Conducted'] - df_pivot['Passes']

    # Keep only post-2021 data
    df_filtered = df_pivot[df_pivot['Year'].isin(['2021-22', '2022-23', '2023-24'])]

    return df_filtered
