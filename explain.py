"""
SHAP explainability module for CVD prediction project.

Provides comprehensive explainability analysis using SHAP values,
showing global and local feature importance.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


def explain_with_shap(model, X_train, X_test, feature_names, model_name):
    """
    Create SHAP explainer and compute SHAP values.
    
    Parameters:
    -----------
    model : sklearn or xgb model
        Trained model
    X_train : np.ndarray
        Training features (for background)
    X_test : np.ndarray
        Test features (for explanation)
    feature_names : list
        Names of features
    model_name : str
        Name of the model
    
    Returns:
    --------
    shap_values : np.ndarray
        SHAP values for test set
    explainer : shap.Explainer
        Fitted SHAP explainer
    """
    print(f"Creating SHAP explainer for {model_name}...", end=" ")
    
    try:
        # Use TreeExplainer for tree-based models
        if model_name in ['Random Forest', 'XGBoost']:
            explainer = shap.TreeExplainer(model)
        # Use LinearExplainer for linear models
        elif model_name == 'Logistic Regression':
            explainer = shap.LinearExplainer(model, X_train)
        # Use KernelExplainer for other models (SVM, Neural Network)
        else:
            explainer = shap.KernelExplainer(
                model.predict,
                shap.sample(X_train, min(100, X_train.shape[0]))
            )
        
        # Compute SHAP values
        if model_name in ['Random Forest', 'XGBoost']:
            shap_values = explainer.shap_values(X_test)
            # For binary classification, take positive class
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
        else:
            shap_values = explainer.shap_values(X_test)
        
        print("✓")
        return shap_values, explainer
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return None, None


def plot_shap_summary(shap_values, X_test, feature_names):
    """
    Create SHAP summary plot (beeswarm).
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values for test set
    X_test : np.ndarray
        Test features
    feature_names : list
        Names of features
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create SHAP values object for plotting
    shap_df = shap.Explanation(
        values=shap_values,
        base_values=shap_values.mean(axis=0),
        data=X_test,
        feature_names=feature_names
    )
    
    shap.summary_plot(shap_df, plot_type="beeswarm", show=False)
    plt.title('SHAP Summary Plot - Feature Impact on Model Output', 
              fontsize=12, fontweight='bold', pad=20)
    
    return plt.gcf()


def plot_shap_bar(shap_values, feature_names):
    """
    Create SHAP bar plot showing mean absolute SHAP values.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values for test set
    feature_names : list
        Names of features
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Calculate mean absolute SHAP values
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # Create dataframe for plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=True)
    
    # Plot
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(importance_df)))
    ax.barh(importance_df['Feature'], importance_df['Mean_Abs_SHAP'], color=colors)
    ax.set_xlabel('Mean |SHAP value|', fontweight='bold')
    ax.set_title('SHAP Feature Importance Ranking', fontsize=12, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for i, v in enumerate(importance_df['Mean_Abs_SHAP']):
        ax.text(v + 0.01, i, f'{v:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_shap_waterfall(shap_values, X_test, feature_names, patient_index=0):
    """
    Create SHAP waterfall plot for a single patient.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values for test set
    X_test : np.ndarray
        Test features
    feature_names : list
        Names of features
    patient_index : int
        Index of patient to explain (default: 0)
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create SHAP values object
    shap_df = shap.Explanation(
        values=shap_values,
        base_values=shap_values.mean(axis=0),
        data=X_test,
        feature_names=feature_names
    )
    
    shap.waterfall_plot(shap_df[patient_index], show=False)
    plt.title(f'SHAP Waterfall Plot - Patient {patient_index} Prediction Explanation',
              fontsize=12, fontweight='bold', pad=20)
    
    return plt.gcf()


def plot_shap_dependence(shap_values, X_test, feature_names, feature):
    """
    Create SHAP dependence plot for a specific feature.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values for test set
    X_test : np.ndarray
        Test features
    feature_names : list
        Names of features
    feature : str
        Name of feature to analyze
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    feature_idx = feature_names.index(feature)
    
    # Create SHAP values object
    shap_df = shap.Explanation(
        values=shap_values,
        base_values=shap_values.mean(axis=0),
        data=X_test,
        feature_names=feature_names
    )
    
    shap.dependence_plot(feature, shap_values, X_test, 
                        feature_names=feature_names, show=False)
    plt.title(f'SHAP Dependence Plot - {feature}',
              fontsize=12, fontweight='bold')
    
    return plt.gcf()


def get_top_features(shap_values, feature_names, top_n=5):
    """
    Get top N features by mean absolute SHAP value.
    
    Parameters:
    -----------
    shap_values : np.ndarray
        SHAP values for test set
    feature_names : list
        Names of features
    top_n : int
        Number of top features to return (default: 5)
    
    Returns:
    --------
    top_features : pandas.DataFrame
        DataFrame with top features ranked by importance
    """
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    top_features = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False).head(top_n)
    
    top_features['Rank'] = range(1, len(top_features) + 1)
    
    return top_features
