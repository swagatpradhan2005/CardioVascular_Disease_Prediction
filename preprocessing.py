"""
Data preprocessing module for CVD prediction project.

Handles loading, cleaning, feature engineering, scaling, and balancing
of cardiovascular disease dataset.
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_data(filepath):
    """
    Load CSV data from filepath with semicolon or comma separator.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file
    
    Returns:
    --------
    df : pandas.DataFrame
        Loaded dataframe
    """
    try:
        df = pd.read_csv(filepath, sep=';')
        if df.shape[1] == 1:  # Fallback to comma if semicolon didn't work
            df = pd.read_csv(filepath, sep=',')
        print(f"✓ Data loaded successfully. Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        raise


def clean_data(df):
    """
    Clean dataset by handling missing values, duplicates, and outliers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    
    Returns:
    --------
    df_clean : pandas.DataFrame
        Cleaned dataframe
    """
    print("\n" + "="*60)
    print("DATA CLEANING")
    print("="*60)
    
    initial_rows = len(df)
    
    # Check missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n✓ Missing values found:\n{missing[missing > 0]}")
        imputer = SimpleImputer(strategy='mean')
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        print("✓ Missing values imputed using mean strategy")
    else:
        print("\n✓ No missing values found")
    
    # Remove duplicates
    duplicates_before = len(df)
    df = df.drop_duplicates()
    duplicates_removed = duplicates_before - len(df)
    if duplicates_removed > 0:
        print(f"✓ Removed {duplicates_removed} duplicate rows")
    
    # Remove outliers in blood pressure using IQR method (1.5 * IQR rule)
    for col in ['ap_hi', 'ap_lo']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_before = len(df)
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        outliers_removed = outliers_before - len(df)
        if outliers_removed > 0:
            print(f"✓ Removed {outliers_removed} outliers in '{col}' (IQR method)")
    
    total_removed = initial_rows - len(df)
    print(f"\n✓ Total rows removed: {total_removed}")
    print(f"✓ Remaining rows: {len(df)}")
    
    return df


def feature_engineering(df):
    """
    Create new features from existing ones.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe with original features
    
    Returns:
    --------
    df_engineered : pandas.DataFrame
        Dataframe with new engineered features
    """
    print("\n" + "="*60)
    print("FEATURE ENGINEERING")
    print("="*60)
    
    df = df.copy()
    
    # Convert age from days to years
    df['age'] = df['age'] / 365.25
    print("✓ Age converted from days to years")
    
    # Calculate BMI
    df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
    print("✓ BMI calculated")
    
    # Calculate pulse pressure
    df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']
    print("✓ Pulse pressure calculated")
    
    # Calculate mean arterial pressure
    df['mean_arterial_pressure'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3
    print("✓ Mean arterial pressure calculated")
    
    # Lifestyle risk score
    df['lifestyle_risk_score'] = df['smoke'] + df['alco'] + (1 - df['active'])
    print("✓ Lifestyle risk score calculated")
    
    # Health risk score
    df['health_risk_score'] = df['cholesterol'] + df['gluc']
    print("✓ Health risk score calculated")
    
    # Age-BMI interaction
    df['age_bmi_interaction'] = df['age'] * df['bmi']
    print("✓ Age-BMI interaction created")
    
    # Blood pressure category
    def categorize_bp(row):
        if row['ap_hi'] < 120 and row['ap_lo'] < 80:
            return 0  # Normal
        elif row['ap_hi'] >= 120 and row['ap_hi'] <= 129 and row['ap_lo'] < 80:
            return 1  # Elevated
        elif (row['ap_hi'] >= 130 and row['ap_hi'] <= 139) or (row['ap_lo'] >= 80 and row['ap_lo'] <= 89):
            return 2  # High Stage 1
        else:
            return 3  # High Stage 2
    
    df['bp_category'] = df.apply(categorize_bp, axis=1)
    print("✓ Blood pressure category created")
    
    print(f"\n✓ Total engineered features: 8 new features added")
    return df


def scale_features(X_train, X_test):
    """
    Standardize features using StandardScaler.
    Fit only on training data, transform both train and test.
    
    Parameters:
    -----------
    X_train : np.ndarray or pd.DataFrame
        Training features
    X_test : np.ndarray or pd.DataFrame
        Test features
    
    Returns:
    --------
    X_train_scaled : np.ndarray
        Scaled training features
    X_test_scaled : np.ndarray
        Scaled test features
    scaler : StandardScaler
        Fitted scaler object
    """
    print("\n" + "="*60)
    print("FEATURE SCALING")
    print("="*60)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("✓ Features scaled using StandardScaler")
    print(f"  - Scaler fit on training data ({X_train_scaled.shape[0]} samples)")
    print(f"  - Transformation applied to test data ({X_test_scaled.shape[0]} samples)")
    
    return X_train_scaled, X_test_scaled, scaler


def handle_imbalance(X_train, y_train):
    """
    Handle class imbalance using SMOTE.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    
    Returns:
    --------
    X_train_balanced : np.ndarray
        Balanced training features
    y_train_balanced : np.ndarray
        Balanced training labels
    """
    print("\n" + "="*60)
    print("HANDLING CLASS IMBALANCE")
    print("="*60)
    
    # Print before
    unique, counts = np.unique(y_train, return_counts=True)
    print("\nBefore SMOTE:")
    for u, c in zip(unique, counts):
        label = "Disease" if u == 1 else "No Disease"
        print(f"  Class {u} ({label}): {c} samples ({100*c/len(y_train):.1f}%)")
    
    # Apply SMOTE
    smote = SMOTE(k_neighbors=5, random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    # Print after
    unique, counts = np.unique(y_train_balanced, return_counts=True)
    print("\nAfter SMOTE:")
    for u, c in zip(unique, counts):
        label = "Disease" if u == 1 else "No Disease"
        print(f"  Class {u} ({label}): {c} samples ({100*c/len(y_train_balanced):.1f}%)")
    
    return X_train_balanced, y_train_balanced


def prepare_data_pipeline(filepath, test_size=0.2, random_state=42, apply_smote=True):
    """
    Execute complete data preprocessing pipeline.
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file
    test_size : float
        Proportion of test set (default: 0.2)
    random_state : int
        Random seed for reproducibility (default: 42)
    apply_smote : bool
        Whether to apply SMOTE balancing (default: True)
    
    Returns:
    --------
    X_train : np.ndarray
        Scaled training features
    X_test : np.ndarray
        Scaled test features
    y_train : np.ndarray
        Training labels (balanced if apply_smote=True)
    y_test : np.ndarray
        Test labels
    feature_names : list
        Names of all features
    """
    print("\n" + "█"*60)
    print("█ CVD PREDICTION - DATA PREPROCESSING PIPELINE")
    print("█"*60)
    
    # Step 1: Load data
    df = load_data(filepath)
    
    # Step 2: Clean data
    df = clean_data(df)
    
    # Step 3: Feature engineering
    df = feature_engineering(df)
    
    # Step 4: Prepare X and y
    target_col = 'cardio'
    X = df.drop(columns=[target_col, 'age'] if 'id' not in df.columns else [target_col])
    y = df[target_col]
    
    # Ensure we only have numeric columns in X
    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns.tolist()
    
    print("\n" + "="*60)
    print("TRAIN-TEST SPLIT")
    print("="*60)
    
    # Step 5: Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"✓ Stratified split completed")
    print(f"  - Training set: {X_train.shape[0]} samples")
    print(f"  - Test set: {X_test.shape[0]} samples")
    print(f"  - Number of features: {X_train.shape[1]}")
    
    # Step 6: Scale features (before SMOTE to fit on original training data)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
    
    # Step 7: Handle imbalance
    if apply_smote:
        X_train_balanced, y_train_balanced = handle_imbalance(X_train_scaled, y_train)
        print("\n✓ SMOTE applied successfully")
    else:
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
        print("\n✓ SMOTE skipped (apply_smote=False)")
    
    print("\n" + "█"*60)
    print("█ PREPROCESSING COMPLETE")
    print("█"*60 + "\n")
    
    return X_train_balanced, X_test_scaled, y_train_balanced, y_test, feature_names
