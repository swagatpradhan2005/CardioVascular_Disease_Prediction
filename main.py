"""
Main execution pipeline for CVD prediction project.

Orchestrates the complete workflow from data loading through model
training, evaluation, and interpretation.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
from pathlib import Path

# Import all modules
from preprocessing import prepare_data_pipeline
from train import train_models, cross_validate_models, get_feature_importance
from evaluate import (
    evaluate_all_models, plot_all_confusion_matrices,
    plot_all_roc_curves_combined, plot_model_comparison,
    plot_neural_network_history, print_classification_report
)
from explain import (
    explain_with_shap, plot_shap_summary, plot_shap_bar,
    plot_shap_waterfall, get_top_features
)
from feature_selection import (
    selectkbest_features, plot_feature_importance_comparison,
    get_consensus_top_features
)
from utils import (
    save_model, save_results, save_plot, plot_data_distribution,
    plot_correlation_heatmap, print_dataset_summary, get_best_model,
    create_project_summary, load_model
)


def main():
    """Execute complete CVD prediction pipeline."""
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  CARDIOVASCULAR DISEASE PREDICTION - MACHINE LEARNING PROJECT  ".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    
    # ========================================================================
    # STEP 1: CREATE DIRECTORIES
    # ========================================================================
    
    print("Step 1: Creating project directories...\n")
    
    dirs = ['data', 'models', 'reports', 'plots']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
    
    print("✓ Project directories ready\n")
    
    # ========================================================================
    # STEP 2: LOAD & PREPROCESS DATA
    # ========================================================================
    
    print("Step 2: Loading and preprocessing data...\n")
    
    dataset_path = 'data/cardio_train.csv'
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"✗ Error: Dataset not found at {dataset_path}")
        print("Please place 'cardio_train.csv' in the 'data/' folder")
        return
    
    # Preprocess data
    X_train, X_test, y_train, y_test, feature_names = prepare_data_pipeline(
        filepath=dataset_path,
        test_size=0.2,
        random_state=42,
        apply_smote=True
    )
    
    print(f"\n✓ Data preprocessing complete")
    print(f"  - Training features shape: {X_train.shape}")
    print(f"  - Test features shape: {X_test.shape}")
    print(f"  - Number of features: {len(feature_names)}\n")
    
    # ========================================================================
    # STEP 3: EXPLORATORY DATA ANALYSIS
    # ========================================================================
    
    print("Step 3: Creating exploratory data analysis plots...\n")
    
    # Load original data for EDA
    import pandas as pd
    df_original = pd.read_csv(dataset_path, sep=';')
    if df_original.shape[1] == 1:
        df_original = pd.read_csv(dataset_path, sep=',')
    
    print_dataset_summary(df_original)
    
    # Plot correlation heatmap
    numeric_df = df_original.select_dtypes(include=['number'])
    fig_corr = plot_correlation_heatmap(numeric_df)
    save_plot(fig_corr, 'correlation_heatmap', 'plots/')
    
    # Plot feature distributions
    numeric_cols = [col for col in numeric_df.columns if col != 'cardio']
    fig_dist = plot_data_distribution(df_original, numeric_cols[:9], 'cardio')
    save_plot(fig_dist, 'feature_distributions', 'plots/')
    
    print("\n✓ exploratory data analysis complete\n")
    
    # ========================================================================
    # STEP 4: TRAIN ALL MODELS
    # ========================================================================
    
    print("Step 4: Training all models...\n")
    
    models, histories = train_models(
        X_train, y_train,
        X_val=None,
        y_val=None,
        hyperparameter_tuning=False
    )
    
    print("✓ All models trained successfully\n")
    
    # ========================================================================
    # STEP 5: CROSS-VALIDATION
    # ========================================================================
    
    print("Step 5: Performing cross-validation...\n")
    
    cv_results = cross_validate_models(models, X_train, y_train, cv=5)
    
    # ========================================================================
    # STEP 6: EVALUATE ALL MODELS
    # ========================================================================
    
    print("Step 6: Evaluating all models on test set...\n")
    
    results_df, predictions = evaluate_all_models(models, X_test, y_test)
    
    # ========================================================================
    # STEP 7: VISUALIZE MODEL PERFORMANCE
    # ========================================================================
    
    print("\nStep 7: Creating performance visualization plots...\n")
    
    models_list = list(models.keys())
    
    # Confusion matrices
    fig_cm = plot_all_confusion_matrices(predictions, y_test, models_list)
    save_plot(fig_cm, 'confusion_matrices', 'plots/')
    
    # Combined ROC curves
    fig_roc = plot_all_roc_curves_combined(models, X_test, y_test)
    save_plot(fig_roc, 'roc_curves_combined', 'plots/')
    
    # Model comparison
    fig_comp = plot_model_comparison(results_df)
    save_plot(fig_comp, 'model_comparison', 'plots/')
    
    # Neural network training history
    if 'Neural Network' in histories:
        fig_nn = plot_neural_network_history(histories['Neural Network'])
        save_plot(fig_nn, 'nn_training_history', 'plots/')
    
    print("✓ Performance visualizations created and saved\n")
    
    # ========================================================================
    # STEP 8: GET BEST MODEL
    # ========================================================================
    
    print("Step 8: Identifying best model...\n")
    
    best_model_name, best_f1 = get_best_model(results_df, 'F1-Score')
    best_model = models[best_model_name]
    
    print(f"✓ Best Model: {best_model_name}")
    print(f"✓ F1-Score: {best_f1:.4f}\n")
    
    # ========================================================================
    # STEP 9: SHAP EXPLAINABILITY ANALYSIS
    # ========================================================================
    
    print("Step 9: SHAP explainability analysis on best model...\n")
    
    # Get SHAP values for best model (skip Neural Network due to complexity)
    if best_model_name != 'Neural Network':
        shap_values, _ = explain_with_shap(
            best_model, X_train, X_test,
            feature_names, best_model_name
        )
        
        if shap_values is not None:
            # SHAP summary plot
            fig_shap_summary = plot_shap_summary(shap_values, X_test, feature_names)
            save_plot(fig_shap_summary, 'shap_summary', 'plots/')
            
            # SHAP bar plot
            fig_shap_bar = plot_shap_bar(shap_values, feature_names)
            save_plot(fig_shap_bar, 'shap_importance', 'plots/')
            
            # SHAP waterfall plot (for first patient)
            fig_shap_waterfall = plot_shap_waterfall(
                shap_values, X_test, feature_names, patient_index=0
            )
            save_plot(fig_shap_waterfall, 'shap_waterfall_patient0', 'plots/')
            
            # Get top features
            top_shap_features = get_top_features(shap_values, feature_names, top_n=10)
            print("\nTop 10 Most Important Features (SHAP):\n")
            print(top_shap_features[['Rank', 'Feature', 'Mean_Abs_SHAP']])
            print()
    else:
        print("⚠ SHAP analysis skipped for Neural Network (use tree-based models for detailed SHAP)")
        top_shap_features = pd.DataFrame()  # Empty dataframe
        shap_values = None
    
    # ========================================================================
    # STEP 10: FEATURE IMPORTANCE COMPARISON
    # ========================================================================
    
    print("\nStep 10: Feature importance comparison across methods...\n")
    
    # Random Forest feature importance
    rf_importance = models['Random Forest'].feature_importances_
    
    # XGBoost feature importance
    xgb_importance = models['XGBoost'].feature_importances_
    
    # SHAP importance (if available)
    if shap_values is not None:
        shap_importance = np.abs(shap_values).mean(axis=0)
    else:
        shap_importance = np.ones(len(feature_names)) / len(feature_names)
    
    # Plot comparison
    fig_imp_comp = plot_feature_importance_comparison(
        rf_importance, xgb_importance, shap_importance, feature_names
    )
    save_plot(fig_imp_comp, 'feature_importance_comparison', 'plots/')
    
    # Get consensus features
    consensus_features = get_consensus_top_features(
        rf_importance, xgb_importance, shap_importance,
        feature_names, top_n=8
    )
    
    print("Consensus Top Features (appearing in top 8 across all methods):\n")
    print(consensus_features[consensus_features['Count_in_Top_N'] > 0].to_string())
    print()
    
    # ========================================================================
    # STEP 11: SAVE MODELS
    # ========================================================================
    
    print("Step 11: Saving trained models...\n")
    
    for model_name, model in models.items():
        filepath = f'models/{model_name.lower().replace(" ", "_")}.pkl'
        save_model(model, filepath)
    
    # ========================================================================
    # STEP 12: SAVE RESULTS
    # ========================================================================
    
    print("\nStep 12: Saving evaluation results...\n")
    
    results_path = 'reports/model_results.csv'
    save_results(results_df, results_path)
    
    # ========================================================================
    # STEP 13: GENERATE PROJECT SUMMARY
    # ========================================================================
    
    print("Step 13: Generating comprehensive project summary...\n")
    
    summary_path = 'reports/project_summary.txt'
    
    # Use SHAP features if available, otherwise use RF features
    if not top_shap_features.empty:
        top_features = top_shap_features
    else:
        top_features = get_feature_importance(
            models['Random Forest'], feature_names, 'Random Forest'
        )
    
    create_project_summary(results_df, best_model_name, top_features, summary_path)
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  PIPELINE EXECUTION COMPLETE  ".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")
    
    print("📊 PROJECT SUMMARY")
    print("="*70)
    print(f"\n✓ Best Model: {best_model_name}")
    print(f"  - F1-Score: {results_df[results_df['Model']==best_model_name]['F1-Score'].values[0]:.4f}")
    print(f"  - Accuracy: {results_df[results_df['Model']==best_model_name]['Accuracy'].values[0]:.4f}")
    print(f"  - ROC-AUC: {results_df[results_df['Model']==best_model_name]['ROC-AUC'].values[0]:.4f}\n")
    
    print("📁 OUTPUT FILES SAVED:")
    print(f"  - Models: models/")
    print(f"  - Plots: plots/")
    print(f"  - Results: reports/model_results.csv")
    print(f"  - Summary: reports/project_summary.txt\n")
    
    print("🎯 NEXT STEPS:")
    print("  1. View plots in 'plots/' folder for visual analysis")
    print("  2. Check 'reports/project_summary.txt' for detailed insights")
    print("  3. Review 'reports/model_results.csv' for metrics comparison")
    print("  4. Open 'CVD_project.ipynb' for interactive analysis\n")
    
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
