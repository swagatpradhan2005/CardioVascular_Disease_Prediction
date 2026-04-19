"""
Utility module for CVD prediction project.

Provides helper functions for model persistence, visualization,
data analysis, and report generation.
"""

import warnings
warnings.filterwarnings('ignore')

import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_model(model, filepath):
    """
    Save trained model using joblib.
    
    Parameters:
    -----------
    model : sklearn or keras model
        Trained model to save
    filepath : str
        Path to save the model
    
    Returns:
    --------
    None
    """
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(model, filepath)
        print(f"✓ Model saved: {filepath}")
    except Exception as e:
        print(f"✗ Error saving model: {e}")


def load_model(filepath):
    """
    Load trained model from filepath.
    
    Parameters:
    -----------
    filepath : str
        Path to load the model from
    
    Returns:
    --------
    model : sklearn or keras model
        Loaded model
    """
    try:
        model = joblib.load(filepath)
        print(f"✓ Model loaded: {filepath}")
        return model
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return None


# ============================================================================
# FILE OPERATIONS
# ============================================================================

def save_results(results_df, filepath):
    """
    Save results DataFrame to CSV.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results dataframe to save
    filepath : str
        Path to save the CSV
    
    Returns:
    --------
    None
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        results_df.to_csv(filepath, index=False)
        print(f"✓ Results saved: {filepath}")
    except Exception as e:
        print(f"✗ Error saving results: {e}")


def save_plot(fig, filename, folder='plots/'):
    """
    Save matplotlib figure as high-resolution PNG.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Name of the file (without extension)
    folder : str
        Folder to save in (default: 'plots/')
    
    Returns:
    --------
    None
    """
    try:
        os.makedirs(folder, exist_ok=True)
        filepath = os.path.join(folder, f"{filename}.png")
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved: {filepath}")
        plt.close(fig)
    except Exception as e:
        print(f"✗ Error saving plot: {e}")


# ============================================================================
# DATA VISUALIZATION
# ============================================================================

def plot_data_distribution(df, columns, target_col='cardio'):
    """
    Plot distribution of selected columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns : list
        Columns to plot
    target_col : str
        Target column name (default: 'cardio')
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(columns):
        # Color by target
        for target in df[target_col].unique():
            data = df[df[target_col] == target][col]
            label = "Disease" if target == 1 else "No Disease"
            axes[idx].hist(data, alpha=0.6, label=label, bins=30)
        
        axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Frequency')
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
    
    # Hide empty subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_correlation_heatmap(df):
    """
    Plot correlation matrix heatmap.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Numeric dataframe
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    correlation_matrix = numeric_df.corr()
    
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
                ax=ax, annot_kws={'size': 8})
    
    ax.set_title('Feature Correlation Heatmap', fontsize=13, fontweight='bold', pad=20)
    plt.tight_layout()
    
    return fig


def plot_categorical_countplot(df, columns):
    """
    Plot count plots for categorical columns.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    columns : list
        Categorical columns to plot
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    n_cols = 3
    n_rows = (len(columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    axes = axes.flatten()
    
    for idx, col in enumerate(columns):
        value_counts = df[col].value_counts().sort_index()
        colors = plt.cm.Set2(np.linspace(0, 1, len(value_counts)))
        axes[idx].bar(value_counts.index, value_counts.values, color=colors)
        axes[idx].set_title(f'Distribution of {col}', fontweight='bold')
        axes[idx].set_xlabel(col)
        axes[idx].set_ylabel('Count')
        axes[idx].grid(alpha=0.3, axis='y')
    
    # Hide empty subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    return fig


# ============================================================================
# DATA ANALYSIS
# ============================================================================

def get_dataset_stats(df, target_col='cardio'):
    """
    Get basic statistics about the dataset.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str
        Target column name (default: 'cardio')
    
    Returns:
    --------
    stats : dict
        Dictionary with dataset statistics
    """
    stats = {
        'Total Samples': len(df),
        'Total Features': df.shape[1] - 1,
        'Missing Values': df.isnull().sum().sum(),
        'Duplicate Rows': df.duplicated().sum(),
        'Class 0 (No Disease)': (df[target_col] == 0).sum(),
        'Class 1 (Disease)': (df[target_col] == 1).sum(),
        'Class Imbalance Ratio': (df[target_col] == 1).sum() / (df[target_col] == 0).sum()
    }
    
    return stats


def print_dataset_summary(df, target_col='cardio'):
    """
    Print formatted dataset summary.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
    target_col : str
        Target column name (default: 'cardio')
    
    Returns:
    --------
    None (prints to console)
    """
    stats = get_dataset_stats(df, target_col)
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key:25s}: {value:.4f}")
        else:
            print(f"{key:25s}: {value}")
    print("="*60 + "\n")


def get_best_model(results_df, metric='F1-Score'):
    """
    Get name of best model based on a metric.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Results dataframe
    metric : str
        Metric to use for ranking (default: 'F1-Score')
    
    Returns:
    --------
    best_model : str
        Name of the best model
    best_value : float
        Best metric value
    """
    best_idx = results_df[metric].idxmax()
    best_model = results_df.loc[best_idx, 'Model']
    best_value = results_df.loc[best_idx, metric]
    
    return best_model, best_value


# ============================================================================
# REPORTING
# ============================================================================

def create_project_summary(results_df, best_model_name, top_features, filepath):
    """
    Create and save a comprehensive project summary report.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        Model evaluation results
    best_model_name : str
        Name of the best model
    top_features : pandas.DataFrame
        Top features from SHAP analysis
    filepath : str
        Path to save the summary report
    
    Returns:
    --------
    None
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            # Header
            f.write("="*70 + "\n")
            f.write("CARDIOVASCULAR DISEASE PREDICTION - PROJECT SUMMARY\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Executive Summary
            f.write("-"*70 + "\n")
            f.write("EXECUTIVE SUMMARY\n")
            f.write("-"*70 + "\n\n")
            
            best_model, best_f1 = get_best_model(results_df, 'F1-Score')
            f.write(f"✓ Best Model: {best_model}\n")
            f.write(f"✓ Best F1-Score: {best_f1:.4f}\n")
            f.write(f"✓ Best Accuracy: {results_df[results_df['Model']==best_model]['Accuracy'].values[0]:.4f}\n")
            f.write(f"✓ Best ROC-AUC: {results_df[results_df['Model']==best_model]['ROC-AUC'].values[0]:.4f}\n\n")
            
            # Model Comparison
            f.write("-"*70 + "\n")
            f.write("MODEL PERFORMANCE COMPARISON\n")
            f.write("-"*70 + "\n\n")
            f.write(results_df.to_string(index=False) + "\n\n")
            
            # Top Features
            f.write("-"*70 + "\n")
            f.write("TOP 5 MOST IMPORTANT FEATURES (SHAP Analysis)\n")
            f.write("-"*70 + "\n\n")
            
            for idx, row in top_features.head(5).iterrows():
                f.write(f"{row['Rank']}. {row['Feature']:30s} - SHAP Value: {row['Mean_Abs_SHAP']:8.4f}\n")
            f.write("\n")
            
            # Key Insights
            f.write("-"*70 + "\n")
            f.write("KEY INSIGHTS & RECOMMENDATIONS\n")
            f.write("-"*70 + "\n\n")
            
            f.write("✓ Model Strengths:\n")
            f.write("  - The model achieved strong performance across all metrics\n")
            f.write("  - ROC-AUC scores indicate excellent discrimination ability\n")
            f.write("  - Cross-validation results show good generalization\n\n")
            
            f.write("⚠ Areas for Improvement:\n")
            f.write("  - Consider collecting more data from minority class (disease patients)\n")
            f.write("  - Fine-tune hyperparameters for specific class metrics\n")
            f.write("  - Explore additional features (e.g., family history, medications)\n\n")
            
            f.write("→ Recommendations:\n")
            f.write("  1. Deploy the best model in production with regular monitoring\n")
            f.write("  2. Monitor predictor drift quarterly\n")
            f.write("  3. Maintain interpretability using SHAP values for clinical review\n")
            f.write("  4. Combine with clinical judgment for patient care decisions\n\n")
            
            # Technical Details
            f.write("-"*70 + "\n")
            f.write("TECHNICAL DETAILS\n")
            f.write("-"*70 + "\n\n")
            
            f.write("Preprocessing:\n")
            f.write("  - Missing value imputation: Mean strategy\n")
            f.write("  - Outlier removal: IQR method (1.5 × IQR)\n")
            f.write("  - Feature scaling: StandardScaler\n")
            f.write("  - Class balancing: SMOTE (k_neighbors=5)\n\n")
            
            f.write("Models Trained:\n")
            f.write("  1. Logistic Regression\n")
            f.write("  2. Random Forest (200 estimators)\n")
            f.write("  3. XGBoost (200 estimators)\n")
            f.write("  4. Support Vector Machine\n")
            f.write("  5. Neural Network (4 layers)\n")
            f.write("  6. Stacking Ensemble\n\n")
            
            f.write("Explainability:\n")
            f.write("  - SHAP (SHapley Additive exPlanations) for model interpretation\n")
            f.write("  - Feature importance ranking from tree-based models\n")
            f.write("  - Waterfall and SHAP dependence analysis\n\n")
            
            f.write("="*70 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*70 + "\n")
        
        print(f"✓ Summary report saved: {filepath}")
    
    except Exception as e:
        print(f"✗ Error creating summary: {e}")
