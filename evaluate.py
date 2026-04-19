"""
Model evaluation module for CVD prediction project.

Provides comprehensive evaluation metrics, visualizations, and comparisons
of trained models.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)


def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a single model on test data.
    
    Parameters:
    -----------
    model : sklearn or keras model
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    model_name : str
        Name of the model
    
    Returns:
    --------
    metrics : dict
        Dictionary with evaluation metrics
    y_pred : np.ndarray
        Predicted labels
    """
    # Handle neural network models
    if model_name == 'Neural Network':
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.decision_function(X_test)
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }
    
    return metrics, y_pred


def evaluate_all_models(models, X_test, y_test):
    """
    Evaluate all models and return comparison DataFrame.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    
    Returns:
    --------
    results_df : pandas.DataFrame
        DataFrame with evaluation results for all models
    predictions : dict
        Dictionary with predictions for each model
    """
    print("\n" + "="*60)
    print("MODEL EVALUATION")
    print("="*60 + "\n")
    
    results = []
    predictions = {}
    
    for model_name, model in models.items():
        metrics, y_pred = evaluate_model(model, X_test, y_test, model_name)
        results.append(metrics)
        predictions[model_name] = y_pred
    
    results_df = pd.DataFrame(results)
    
    # Print formatted results table
    print(results_df.to_string(index=False))
    print("\n")
    
    return results_df, predictions


def plot_confusion_matrix(y_test, y_pred, model_name, ax):
    """
    Plot confusion matrix for a single model.
    
    Parameters:
    -----------
    y_test : np.ndarray
        True labels
    y_pred : np.ndarray
        Predicted labels
    model_name : str
        Name of the model
    ax : matplotlib.axes.Axes
        Axes object to plot on
    
    Returns:
    --------
    None (modifies ax in place)
    """
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
                xticklabels=['No Disease', 'Disease'],
                yticklabels=['No Disease', 'Disease'])
    ax.set_title(f'{model_name}', fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')


def plot_all_confusion_matrices(predictions, y_test, models_list):
    """
    Plot confusion matrices for all models in a grid.
    
    Parameters:
    -----------
    predictions : dict
        Dictionary with predictions for each model
    y_test : np.ndarray
        True test labels
    models_list : list
        List of model names
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Confusion Matrices - All Models', fontsize=14, fontweight='bold')
    axes = axes.flatten()
    
    for idx, model_name in enumerate(models_list):
        if model_name in predictions:
            plot_confusion_matrix(y_test, predictions[model_name], model_name, axes[idx])
    
    # Hide empty subplot if odd number of models
    if len(models_list) < 6:
        axes[-1].set_visible(False)
    
    plt.tight_layout()
    return fig


def plot_roc_curve(model, X_test, y_test, model_name, ax):
    """
    Plot ROC curve for a single model.
    
    Parameters:
    -----------
    model : sklearn or keras model
        Trained model
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    model_name : str
        Name of the model
    ax : matplotlib.axes.Axes
        Axes object to plot on
    
    Returns:
    --------
    None (modifies ax in place)
    """
    # Get prediction probabilities
    if model_name == 'Neural Network':
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
    else:
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.decision_function(X_test)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    # Plot
    ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC={roc_auc:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)


def plot_all_roc_curves_combined(models, X_test, y_test):
    """
    Plot all ROC curves on a single combined plot.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_test : np.ndarray
        Test features
    y_test : np.ndarray
        Test labels
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for idx, (model_name, model) in enumerate(models.items()):
        # Get prediction probabilities
        if model_name == 'Neural Network':
            y_pred_proba = model.predict(X_test, verbose=0).flatten()
        else:
            if hasattr(model, 'predict_proba'):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = model.decision_function(X_test)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Plot
        ax.plot(fpr, tpr, lw=2.5, label=f'{model_name} (AUC={roc_auc:.3f})',
                color=colors[idx % len(colors)])
    
    # Plot random classifier
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title('Combined ROC Curves - All Models', fontsize=13, fontweight='bold')
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_model_comparison(results_df):
    """
    Create bar chart comparing all metrics across all models.
    
    Parameters:
    -----------
    results_df : pandas.DataFrame
        DataFrame with evaluation results
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    fig.suptitle('Model Performance Comparison - All Metrics', 
                 fontsize=14, fontweight='bold')
    
    for idx, metric in enumerate(metrics):
        data = results_df[['Model', metric]].sort_values(metric, ascending=True)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(data)))
        axes[idx].barh(data['Model'], data[metric], color=colors)
        axes[idx].set_xlabel(metric, fontweight='bold')
        axes[idx].set_xlim([0, 1])
        axes[idx].grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(data[metric]):
            axes[idx].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def plot_neural_network_history(history):
    """
    Plot training and validation accuracy/loss over epochs.
    
    Parameters:
    -----------
    history : keras.callbacks.History
        Training history object from model.fit()
    
    Returns:
    --------
    fig : matplotlib.figure.Figure
        Figure object
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Neural Network Training History', fontsize=13, fontweight='bold')
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Model Accuracy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Model Loss')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    return fig


def print_classification_report(y_test, y_pred, model_name):
    """
    Print full classification report for a model.
    
    Parameters:
    -----------
    y_test : np.ndarray
        True test labels
    y_pred : np.ndarray
        Predicted labels
    model_name : str
        Name of the model
    
    Returns:
    --------
    None (prints to console)
    """
    print("\n" + "="*60)
    print(f"CLASSIFICATION REPORT - {model_name}")
    print("="*60 + "\n")
    print(classification_report(y_test, y_pred, 
                              target_names=['No Disease', 'Disease']))
