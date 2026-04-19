"""
Ultra-fast CVD prediction pipeline.

Uses only LR, RF, and XGB for quick results.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd

# Import modules
from preprocessing import prepare_data_pipeline
from evaluate import (
    evaluate_all_models, plot_all_confusion_matrices,
    plot_all_roc_curves_combined, plot_model_comparison
)
from explain import (
    explain_with_shap, plot_shap_summary, plot_shap_bar,
    plot_shap_waterfall, get_top_features
)
from utils import (
    save_model, save_results, save_plot, plot_correlation_heatmap,
    print_dataset_summary, get_best_model, create_project_summary
)

# Import ML libraries
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import shap


def main():
    """Execute ultra-fast CVD prediction pipeline."""
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  CVD PREDICTION - ULTRA-FAST (3 Models)  ".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    
    # Create directories
    for dir_name in ['data', 'models', 'reports', 'plots']:
        os.makedirs(dir_name, exist_ok=True)
    
    print("✓ Project directories ready\n")
    
    # Load and preprocess data
    dataset_path = 'data/cardio_train.csv'
    if not os.path.exists(dataset_path):
        print(f"✗ Error: Dataset not found at {dataset_path}")
        return
    
    print("Loading and preprocessing data...\n")
    X_train, X_test, y_train, y_test, feature_names = prepare_data_pipeline(
        filepath=dataset_path, test_size=0.2, random_state=42, apply_smote=True
    )
    
    # EDA plots
    print("\nCreating EDA plots...\n")
    df_original = pd.read_csv(dataset_path, sep=';')
    if df_original.shape[1] == 1:
        df_original = pd.read_csv(dataset_path, sep=',')
    
    print_dataset_summary(df_original)
    numeric_df = df_original.select_dtypes(include=['number'])
    fig_corr = plot_correlation_heatmap(numeric_df)
    save_plot(fig_corr, 'correlation_heatmap', 'plots/')
    
    # Train only 3 fast models
    print("Training 3 core models...\n")
    
    models = {}
    
    print("Logistic Regression...", end=" ")
    models['Logistic Regression'] = LogisticRegression(
        C=1, max_iter=500, random_state=42, n_jobs=4)
    models['Logistic Regression'].fit(X_train, y_train)
    print("✓")
    
    print("Random Forest...", end=" ")
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=60, max_depth=10, random_state=42, n_jobs=4)
    models['Random Forest'].fit(X_train, y_train)
    print("✓")
    
    print("XGBoost...", end=" ")
    models['XGBoost'] = xgb.XGBClassifier(
        max_depth=8, learning_rate=0.1, n_estimators=60,
        objective='binary:logistic', random_state=42, n_jobs=4,
        use_label_encoder=False, eval_metric='logloss', verbosity=0)
    models['XGBoost'].fit(X_train, y_train)
    print("✓\n")
    
    # Evaluate models
    print("Evaluating models...\n")
    results_df, predictions = evaluate_all_models(models, X_test, y_test)
    
    # Visualizations
    print("\nCreating visualizations...\n")
    fig_cm = plot_all_confusion_matrices(predictions, y_test, list(models.keys()))
    save_plot(fig_cm, 'confusion_matrices', 'plots/')
    
    fig_roc = plot_all_roc_curves_combined(models, X_test, y_test)
    save_plot(fig_roc, 'roc_curves_combined', 'plots/')
    
    fig_comp = plot_model_comparison(results_df)
    save_plot(fig_comp, 'model_comparison', 'plots/')
    
    # Best model
    best_model_name, best_f1 = get_best_model(results_df, 'F1-Score')
    print(f"✓ Best Model: {best_model_name} (F1={best_f1:.4f})\n")
    
    # SHAP for best model (if it's RF or XGB)
    if best_model_name in ['Random Forest', 'XGBoost']:
        print("Computing SHAP values...\n")
        best_model = models[best_model_name]
        
        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # SHAP plots
        fig_summary = plot_shap_summary(shap_values, X_test, feature_names)
        save_plot(fig_summary, 'shap_summary', 'plots/')
        
        fig_bar = plot_shap_bar(shap_values, feature_names)
        save_plot(fig_bar, 'shap_importance', 'plots/')
        
        top_features = get_top_features(shap_values, feature_names, top_n=10)
    else:
        # Use RF for SHAP if best model is LR
        print("Computing SHAP values (using Random Forest)...\n")
        explainer = shap.TreeExplainer(models['Random Forest'])
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        fig_summary = plot_shap_summary(shap_values, X_test, feature_names)
        save_plot(fig_summary, 'shap_summary', 'plots/')
        
        fig_bar = plot_shap_bar(shap_values, feature_names)
        save_plot(fig_bar, 'shap_importance', 'plots/')
        
        top_features = get_top_features(shap_values, feature_names, top_n=10)
    
    # Save models and results
    print("\nSaving models and results...\n")
    for name, model in models.items():
        save_model(model, f'models/{name.lower().replace(" ", "_")}.pkl')
    
    save_results(results_df, 'reports/model_results.csv')
    create_project_summary(results_df, best_model_name, top_features, 'reports/project_summary.txt')
    
    # Final summary
    print("\n" + "█"*70)
    print("█" + "  COMPLETE!  ".center(70) + "█")
    print("█"*70 + "\n")
    
    print("✓ All models trained and evaluated")
    print(f"✓ Best Model: {best_model_name} (F1={best_f1:.4f})")
    print("\n📁 Output saved to:")
    print("  - plots/ (PNG images)")
    print("  - models/ (trained models)")
    print("  - reports/model_results.csv")
    print("  - reports/project_summary.txt\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
