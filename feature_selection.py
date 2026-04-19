"""
Feature selection module for CVD prediction project.

Provides methods for selecting and comparing the most important features
using statistical and model-based approaches.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, chi2


def selectkbest_features(X, y, feature_names, k=10):
    """
    Select k best features using chi-square test.
    
    Parameters:
    -----------
    X : np.ndarray
        Feature matrix (must be non-negative)
    y : np.ndarray
        Target variable
    feature_names : list
        Names of features
    k : int
        Number of features to select (default: 10)
    
    Returns:
    --------
    importance_df : pandas.DataFrame
        DataFrame with features ranked by chi2 score
    """
    # Ensure features are non-negative by min-max scaling to [0, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_normalized = (X - X_min) / (X_max - X_min + 1e-10)
    
    selector = SelectKBest(score_func=chi2, k=k)
    selector.fit(X_normalized, y)
    
    scores = selector.scores_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Chi2_Score': scores
    }).sort_values('Chi2_Score', ascending=False)
    
    importance_df['Rank'] = range(1, len(importance_df) + 1)
    
    return importance_df


def plot_feature_importance_comparison(rf_importance, xgb_importance, 
                                       shap_importance, feature_names):
    """
    Create side-by-side bar charts comparing feature importance from
    Random Forest, XGBoost, and SHAP methods.
    
    Parameters:
    -----------
    rf_importance : np.ndarray or list
        Feature importance from Random Forest
    xgb_importance : np.ndarray or list
        Feature importance from XGBoost
    shap_importance : np.ndarray or list
        Mean absolute SHAP values
    feature_names : list
        Names of features
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    # Normalize importances to [0, 1] for comparison
    rf_norm = rf_importance / rf_importance.sum()
    xgb_norm = xgb_importance / xgb_importance.sum()
    shap_norm = shap_importance / shap_importance.sum()
    
    # Create dataframe for easier plotting
    comparison_df = pd.DataFrame({
        'Feature': feature_names,
        'Random Forest': rf_norm,
        'XGBoost': xgb_norm,
        'SHAP': shap_norm
    })
    
    # Sort by average importance
    comparison_df['Average'] = comparison_df[['Random Forest', 'XGBoost', 'SHAP']].mean(axis=1)
    comparison_df = comparison_df.sort_values('Average', ascending=True).tail(10)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('Feature Importance Comparison: Three Methods', 
                 fontsize=14, fontweight='bold')
    
    # Random Forest
    comparison_df.set_index('Feature')['Random Forest'].plot(
        kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title('Random Forest', fontweight='bold')
    axes[0].set_xlabel('Normalized Importance')
    axes[0].grid(axis='x', alpha=0.3)
    
    # XGBoost
    comparison_df.set_index('Feature')['XGBoost'].plot(
        kind='barh', ax=axes[1], color='darkorange')
    axes[1].set_title('XGBoost', fontweight='bold')
    axes[1].set_xlabel('Normalized Importance')
    axes[1].grid(axis='x', alpha=0.3)
    
    # SHAP
    comparison_df.set_index('Feature')['SHAP'].plot(
        kind='barh', ax=axes[2], color='seagreen')
    axes[2].set_title('SHAP Values', fontweight='bold')
    axes[2].set_xlabel('Mean Absolute SHAP')
    axes[2].grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return fig


def get_consensus_top_features(rf_imp, xgb_imp, shap_imp, feature_names, top_n=8):
    """
    Find features that appear in top N across multiple importance methods.
    
    Parameters:
    -----------
    rf_imp : np.ndarray
        Random Forest feature importances
    xgb_imp : np.ndarray
        XGBoost feature importances
    shap_imp : np.ndarray
        Mean absolute SHAP values
    feature_names : list
        Names of features
    top_n : int
        Number of top features to consider in consensus (default: 8)
    
    Returns:
    --------
    consensus_df : pandas.DataFrame
        DataFrame with consensus ranking and feature counts
    """
    # Create ranking dataframes
    rf_ranked = pd.DataFrame({
        'Feature': feature_names,
        'RF_Importance': rf_imp,
        'RF_Rank': range(1, len(feature_names) + 1)
    }).sort_values('RF_Importance', ascending=False).reset_index(drop=True)
    rf_ranked['RF_Rank'] = range(1, len(rf_ranked) + 1)
    
    xgb_ranked = pd.DataFrame({
        'Feature': feature_names,
        'XGB_Importance': xgb_imp,
        'XGB_Rank': range(1, len(feature_names) + 1)
    }).sort_values('XGB_Importance', ascending=False).reset_index(drop=True)
    xgb_ranked['XGB_Rank'] = range(1, len(xgb_ranked) + 1)
    
    shap_ranked = pd.DataFrame({
        'Feature': feature_names,
        'SHAP_Value': shap_imp,
        'SHAP_Rank': range(1, len(feature_names) + 1)
    }).sort_values('SHAP_Value', ascending=False).reset_index(drop=True)
    shap_ranked['SHAP_Rank'] = range(1, len(shap_ranked) + 1)
    
    # Merge rankings
    consensus = rf_ranked[['Feature', 'RF_Rank']].merge(
        xgb_ranked[['Feature', 'XGB_Rank']], on='Feature'
    ).merge(
        shap_ranked[['Feature', 'SHAP_Rank']], on='Feature'
    )
    
    # Count how many times each feature appears in top N
    consensus['Count_in_Top_N'] = 0
    
    top_rf = set(rf_ranked.head(top_n)['Feature'])
    top_xgb = set(xgb_ranked.head(top_n)['Feature'])
    top_shap = set(shap_ranked.head(top_n)['Feature'])
    
    consensus['Count_in_Top_N'] = consensus['Feature'].apply(
        lambda f: sum([f in top_rf, f in top_xgb, f in top_shap])
    )
    
    # Average rank
    consensus['Avg_Rank'] = (consensus['RF_Rank'] + 
                             consensus['XGB_Rank'] + 
                             consensus['SHAP_Rank']) / 3
    
    consensus = consensus.sort_values(['Count_in_Top_N', 'Avg_Rank'], 
                                      ascending=[False, True])
    
    return consensus
