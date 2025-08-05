import pandas as pd
import numpy as np

def clean_hr_data(file_path):
    """
    Cleans and prepares raw HR employee data by handling missing values,
    correcting data types, and encoding categorical columns.

    Args:
        file_path (str): The path to the raw HR data CSV file.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    print(f"--- HR Data Cleaning for Analysis: {file_path} ---")

    # 1. Data Input Handling (Reading CSV)
    try:
        df = pd.read_csv(file_path)
        print("\nOriginal DataFrame Info:")
        df.info()
        print("\nMissing values before cleaning:")
        print(df.isnull().sum())
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

    # Make a copy to avoid SettingWithCopyWarning - this is good practice
    cleaned_df = df.copy()

    # 2. Data Cleaning & Preprocessing: Handle Missing Values
    # Strategy:
    # - For numerical columns (Age, Salary, Experience), fill with median.
    # - For categorical columns (Department, Gender, Attrition, Education), fill with mode.
    print("\n--- Handling Missing Values ---")
    for column in ['Age', 'Salary', 'Experience']: # Assuming these are numeric
        if cleaned_df[column].isnull().any():
            median_val = cleaned_df[column].median()
            # Fix: Assign the result back to the column instead of using inplace=True
            cleaned_df[column] = cleaned_df[column].fillna(median_val)
            print(f"Filled missing values in '{column}' with median: {median_val}")

    for column in ['Department', 'Gender', 'Attrition', 'Education']: # Assuming these are categorical
        if cleaned_df[column].isnull().any():
            mode_val = cleaned_df[column].mode()[0]
            # Fix: Assign the result back to the column instead of using inplace=True
            cleaned_df[column] = cleaned_df[column].fillna(mode_val)
            print(f"Filled missing values in '{column}' with mode: {mode_val}")

    # Verify no missing values (Expected Outcome: No missing values)
    print("\nMissing values after filling:")
    print(cleaned_df.isnull().sum())
    if cleaned_df.isnull().sum().sum() == 0:
        print("Successfully handled all missing values.")
    else:
        print("Warning: Some missing values still exist.")


    # 3. Data Cleaning & Preprocessing: Correct Data Types
    # (Expected Outcome: Correct data types)
    print("\n--- Correcting Data Types ---")
    # Convert 'Age', 'Salary', 'Experience' to integer if they are floats after filling
    for col in ['Age', 'Salary', 'Experience']:
        if cleaned_df[col].dtype == 'float64':
            cleaned_df[col] = cleaned_df[col].astype(int)
            print(f"Converted '{col}' to integer type.")

    # Convert 'EmployeeID' to string/object if it's not needed for numerical operations
    if cleaned_df['EmployeeID'].dtype != 'object':
         cleaned_df['EmployeeID'] = cleaned_df['EmployeeID'].astype(str)
         print(f"Converted 'EmployeeID' to string type.")

    print("\nDataFrame Info after type correction:")
    cleaned_df.info()

    # 4. Data Cleaning & Preprocessing: Encoded Categorical Columns
    # (Expected Outcome: Encoded categorical columns)
    print("\n--- Encoding Categorical Columns ---")
    categorical_cols = ['Department', 'Attrition', 'Education', 'Gender']
    # Use one-hot encoding for simplicity and to avoid ordinality issues
    cleaned_df = pd.get_dummies(cleaned_df, columns=categorical_cols, drop_first=True)
    print(f"Encoded categorical columns: {categorical_cols}")

    print("\nDataFrame Info after encoding:")
    cleaned_df.info()
    print("\nFirst 5 rows of the cleaned DataFrame:")
    print(cleaned_df.head())

    print("\n--- HR Data Cleaning Complete ---")
    return cleaned_df

# Example Usage (You would replace 'hr_data.csv' with your actual file)
if __name__ == "__main__":
    # Create a dummy CSV for demonstration purposes
    dummy_data = {
        'EmployeeID': [1, 2, 3, 4, 5, 6],
        'Age': [30, 25, np.nan, 40, 35, 28],
        'Department': ['HR', 'Sales', 'IT', 'HR', np.nan, 'IT'],
        'Attrition': ['No', 'Yes', 'No', 'No', 'Yes', np.nan],
        'Salary': [50000, 60000, 75000, np.nan, 55000, 70000],
        'Experience': [5, 2, 10, 15, np.nan, 3],
        'Education': ['Bachelors', 'Masters', 'PhD', 'Bachelors', 'Masters', 'High School'],
        'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', np.nan]
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_file_path = 'dummy_hr_data.csv'
    dummy_df.to_csv(dummy_file_path, index=False)
    print(f"Created dummy CSV: {dummy_file_path}")

    cleaned_hr_dataframe = clean_hr_data(dummy_file_path)

    if cleaned_hr_dataframe is not None:
        print("\nCleaned HR Data Head:")
        print(cleaned_hr_dataframe.head())
        print("\nCleaned HR Data Info:")
        cleaned_hr_dataframe.info()
        print("\nFinal Missing values check (should be all 0):")
        print(cleaned_hr_dataframe.isnull().sum())
