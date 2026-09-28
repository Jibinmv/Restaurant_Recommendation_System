

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load the restaurant dataset from a CSV file.

    Parameters
    ----------
    filepath : str
        Path to the restaurants.csv file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe loaded from the CSV.
    """
    # Try different encodings since some CSVs have special characters
    for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            print(f"[INFO] Dataset loaded successfully with encoding: {encoding}")
            print(f"[INFO] Dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
            return df
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not read the file: {filepath}")


def display_basic_info(df: pd.DataFrame) -> None:
    """
    Display basic information about the dataset:
    shape, columns, data types, missing values, and duplicates.

    Parameters
    ----------
    df : pd.DataFrame
        The restaurant dataframe.
    """
    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    # Shape
    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")

    # Columns and data types
    print("\nColumns and Data Types:")
    print("-" * 40)
    for col in df.columns:
        print(f"  {col:30s} -> {df[col].dtype}")

    # Missing values
    missing = df.isnull().sum()
    missing_cols = missing[missing > 0]
    if len(missing_cols) > 0:
        print("\nMissing Values:")
        print("-" * 40)
        for col, count in missing_cols.items():
            pct = count / len(df) * 100
            print(f"  {col:30s} -> {count:5d} ({pct:.2f}%)")
    else:
        print("\n[INFO] No missing values found.")

    # Duplicates
    dup_count = df.duplicated().sum()
    print(f"\nDuplicate Rows: {dup_count}")


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean column names by stripping whitespace and removing BOM characters.

    Parameters
    ----------
    df : pd.DataFrame
        The restaurant dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with cleaned column names.
    """
    # Remove BOM (byte-order-mark) characters that sometimes appear
    df.columns = df.columns.str.replace('\ufeff', '', regex=False)
    # Strip any leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()
    return df


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.

    Strategy:
    - Cuisines: Fill with "Unknown" (text field)
    - Other categorical columns: Fill with "Unknown" or the mode
    - Numerical columns: Fill with the median

    Parameters
    ----------
    df : pd.DataFrame
        The restaurant dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with missing values handled.
    """
    df = df.copy()

    # --- Handle Cuisines ---
    if 'Cuisines' in df.columns:
        missing_cuisines = df['Cuisines'].isnull().sum()
        df['Cuisines'] = df['Cuisines'].fillna('Unknown')
        print(f"[INFO] Filled {missing_cuisines} missing 'Cuisines' values with 'Unknown'.")

    # --- Handle other categorical columns ---
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        if col == 'Cuisines':
            continue  # Already handled
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col] = df[col].fillna('Unknown')
            print(f"[INFO] Filled {missing_count} missing '{col}' values with 'Unknown'.")

    # --- Handle numerical columns ---
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numerical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"[INFO] Filled {missing_count} missing '{col}' values with median ({median_val}).")

    return df


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize text columns by stripping whitespace and converting to
    consistent casing where appropriate.

    Parameters
    ----------
    df : pd.DataFrame
        The restaurant dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with normalized text fields.
    """
    df = df.copy()

    # Strip whitespace from key text columns
    text_cols = ['City', 'Cuisines', 'Restaurant Name',
                 'Has Online delivery', 'Has Table booking']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Normalize City to title case for consistency
    if 'City' in df.columns:
        df['City'] = df['City'].str.title()

    # Normalize Yes/No columns
    for col in ['Has Online delivery', 'Has Table booking']:
        if col in df.columns:
            df[col] = df[col].str.strip().str.title()

    print("[INFO] Text columns normalized (whitespace stripped, casing standardized).")
    return df


def preprocess_dataset(filepath: str) -> pd.DataFrame:
    """
    Full preprocessing pipeline: load, inspect, clean, and return the
    processed dataset ready for the recommendation engine.

    Parameters
    ----------
    filepath : str
        Path to the restaurants.csv file.

    Returns
    -------
    pd.DataFrame
        Fully preprocessed dataframe.
    """
    print("\n" + "#" * 60)
    print("  STEP 1: DATA PREPROCESSING")
    print("#" * 60)

    # 1. Load the data
    df = load_dataset(filepath)

    # 2. Clean column names (remove BOM, whitespace)
    df = clean_column_names(df)

    # 3. Display basic info BEFORE cleaning
    display_basic_info(df)

    # 4. Handle missing values
    print("\n--- Handling Missing Values ---")
    df = handle_missing_values(df)

    # 5. Normalize text
    print("\n--- Normalizing Text ---")
    df = normalize_text(df)

    # 6. Final check
    remaining_missing = df.isnull().sum().sum()
    print(f"\n[INFO] Remaining missing values after preprocessing: {remaining_missing}")
    print(f"[INFO] Final dataset shape: {df.shape[0]} rows x {df.shape[1]} columns")
    print("[INFO] Preprocessing complete!\n")

    return df


# ---------------------------------------------------------------------------
# If run directly, preprocess and display summary
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Determine the path relative to this file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'restaurants.csv')
    df = preprocess_dataset(data_path)
    print(df.head())
