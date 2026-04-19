"""
Model training module for CVD prediction project.

Trains multiple machine learning models with optional hyperparameter tuning
and provides cross-validation and feature importance analysis.
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import f1_score
import xgboost as xgb
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import StackingClassifier


def train_logistic_regression(X_train, y_train, hyperparameter_tuning=False):
    """
    Train Logistic Regression model.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    hyperparameter_tuning : bool
        Whether to perform GridSearchCV (default: False)
    
    Returns:
    --------
    model : LogisticRegression
        Trained model
    """
    print("Training Logistic Regression...", end=" ")
    
    if hyperparameter_tuning:
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'max_iter': [200, 500]
        }
        model = GridSearchCV(
            LogisticRegression(random_state=42, n_jobs=-1),
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        print(f"✓ (Best C={model.best_params_['C']}, max_iter={model.best_params_['max_iter']})")
        return model.best_estimator_
    else:
        model = LogisticRegression(
            C=1,
            max_iter=500,
            random_state=42,
            n_jobs=4
        )
        model.fit(X_train, y_train)
        print("✓")
        return model


def train_random_forest(X_train, y_train, hyperparameter_tuning=False):
    """
    Train Random Forest Classifier.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    hyperparameter_tuning : bool
        Whether to perform GridSearchCV (default: False)
    
    Returns:
    --------
    model : RandomForestClassifier
        Trained model
    """
    print("Training Random Forest...", end=" ")
    
    if hyperparameter_tuning:
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [10, 15],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 4]
        }
        model = GridSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=4),
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=4
        )
        model.fit(X_train, y_train)
        print(f"✓ (Best params found)")
        return model.best_estimator_
    else:
        model = RandomForestClassifier(
            n_estimators=80,
            max_depth=12,
            random_state=42,
            n_jobs=4
        )
        model.fit(X_train, y_train)
        print("✓")
        return model


def train_xgboost(X_train, y_train, hyperparameter_tuning=False):
    """
    Train XGBoost Classifier.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    hyperparameter_tuning : bool
        Whether to perform GridSearchCV (default: False)
    
    Returns:
    --------
    model : xgb.XGBClassifier
        Trained model
    """
    print("Training XGBoost...", end=" ")
    
    if hyperparameter_tuning:
        param_grid = {
            'max_depth': [5, 8],
            'learning_rate': [0.05, 0.1],
            'n_estimators': [50, 100]
        }
        model = GridSearchCV(
            xgb.XGBClassifier(
                objective='binary:logistic',
                random_state=42,
                n_jobs=4,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=4
        )
        model.fit(X_train, y_train)
        print(f"✓ (Best params found)")
        return model.best_estimator_
    else:
        model = xgb.XGBClassifier(
            max_depth=8,
            learning_rate=0.1,
            n_estimators=80,
            objective='binary:logistic',
            random_state=42,
            n_jobs=4,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        model.fit(X_train, y_train)
        print("✓")
        return model


def train_svm(X_train, y_train, hyperparameter_tuning=False):
    """
    Train Support Vector Machine Classifier.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    hyperparameter_tuning : bool
        Whether to perform GridSearchCV (default: False)
    
    Returns:
    --------
    model : SVC
        Trained model
    """
    print("Training SVM...", end=" ")
    
    if hyperparameter_tuning:
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
        model = GridSearchCV(
            SVC(probability=True, random_state=42),
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        print(f"✓ (Best params found)")
        return model.best_estimator_
    else:
        model = SVC(
            C=10,
            kernel='rbf',
            probability=True,
            random_state=42
        )
        model.fit(X_train, y_train)
        print("✓")
        return model


def train_neural_network(X_train, y_train, X_val, y_val):
    """
    Train Neural Network using TensorFlow/Keras.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_val : np.ndarray
        Validation features
    y_val : np.ndarray
        Validation labels
    
    Returns:
    --------
    model : tf.keras.Sequential
        Trained model
    history : tf.keras.History
        Training history object
    """
    print("Training Neural Network...", end=" ")
    
    n_features = X_train.shape[1]
    
    model = Sequential([
        Dense(128, activation='relu', input_shape=(n_features,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=[early_stopping],
        verbose=0
    )
    
    print("✓")
    return model, history


def train_stacking_ensemble(X_train, y_train, base_models, meta_learner):
    """
    Train Stacking Ensemble Classifier.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    base_models : list
        List of base model instances
    meta_learner : sklearn model
        Meta learner model
    
    Returns:
    --------
    model : StackingClassifier
        Trained stacking ensemble
    """
    print("Training Stacking Ensemble...", end=" ")
    
    base_models_list = [
        ('lr', base_models[0]),
        ('rf', base_models[1]),
        ('xgb', base_models[2]),
        ('svm', base_models[3])
    ]
    
    model = StackingClassifier(
        estimators=base_models_list,
        final_estimator=meta_learner,
        cv=5
    )
    
    model.fit(X_train, y_train)
    print("✓")
    return model


def train_models(X_train, y_train, X_val=None, y_val=None, hyperparameter_tuning=False):
    """
    Train all models and return as dictionary.
    
    Parameters:
    -----------
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    X_val : np.ndarray
        Validation features (for neural network)
    y_val : np.ndarray
        Validation labels (for neural network)
    hyperparameter_tuning : bool
        Whether to perform hyperparameter tuning
    
    Returns:
    --------
    models : dict
        Dictionary with model names as keys and trained models as values
    histories : dict
        Dictionary with training histories (only for neural network)
    """
    print("\n" + "="*60)
    print("MODEL TRAINING")
    print("="*60 + "\n")
    
    # Create validation set if not provided
    if X_val is None:
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train_temp, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        y_train = y_train_temp
    
    models = {}
    histories = {}
    
    # Train all models
    models['Logistic Regression'] = train_logistic_regression(
        X_train, y_train, hyperparameter_tuning
    )
    
    models['Random Forest'] = train_random_forest(
        X_train, y_train, hyperparameter_tuning
    )
    
    models['XGBoost'] = train_xgboost(
        X_train, y_train, hyperparameter_tuning
    )
    
    models['SVM'] = train_svm(
        X_train, y_train, hyperparameter_tuning
    )
    
    # Neural Network
    nn_model, nn_history = train_neural_network(
        X_train, y_train, X_val, y_val
    )
    models['Neural Network'] = nn_model
    histories['Neural Network'] = nn_history
    
    # Stacking Ensemble (using first 4 models as base)
    base_models = [
        models['Logistic Regression'],
        models['Random Forest'],
        models['XGBoost'],
        models['SVM']
    ]
    meta_learner = LogisticRegression(max_iter=500, random_state=42)
    models['Stacking Ensemble'] = train_stacking_ensemble(
        X_train, y_train, base_models, meta_learner
    )
    
    print("\n✓ All models trained successfully\n")
    
    return models, histories


def cross_validate_models(models, X_train, y_train, cv=5):
    """
    Perform cross-validation on all models using F1-score.
    
    Parameters:
    -----------
    models : dict
        Dictionary of trained models
    X_train : np.ndarray
        Training features
    y_train : np.ndarray
        Training labels
    cv : int
        Number of cross-validation folds (default: 5)
    
    Returns:
    --------
    cv_results : dict
        Dictionary with cross-validation results
    """
    print("="*60)
    print("CROSS-VALIDATION RESULTS (F1-Score)")
    print("="*60 + "\n")
    
    cv_results = {}
    
    for model_name, model in models.items():
        # Skip neural network for now (requires special handling)
        if model_name == 'Neural Network':
            continue
            
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv,
            scoring='f1',
            n_jobs=-1
        )
        
        cv_results[model_name] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
        
        print(f"{model_name:20s}: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    print("\n")
    
    return cv_results


def get_feature_importance(model, feature_names, model_name):
    """
    Extract feature importance from trained model.
    
    Parameters:
    -----------
    model : sklearn or xgb model object
        Trained model
    feature_names : list
        Names of features
    model_name : str
        Name of the model
    
    Returns:
    --------
    importance_df : pandas.DataFrame
        DataFrame with features ranked by importance
    """
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': 0.0
    })
    
    try:
        # For tree-based models (Random Forest, XGBoost)
        if hasattr(model, 'feature_importances_'):
            importance_df['Importance'] = model.feature_importances_
        
        # For linear models (Logistic Regression)
        elif hasattr(model, 'coef_'):
            importance_df['Importance'] = np.abs(model.coef_[0])
        
        # For SVM, use permutation importance
        elif model_name == 'SVM':
            # SVM doesn't have built-in feature importance
            # Return uniform importance
            importance_df['Importance'] = np.ones(len(feature_names)) / len(feature_names)
        
    except Exception as e:
        print(f"Warning: Could not extract importance for {model_name}: {e}")
        importance_df['Importance'] = np.ones(len(feature_names)) / len(feature_names)
    
    importance_df = importance_df.sort_values('Importance', ascending=False).reset_index(drop=True)
    importance_df['Rank'] = range(1, len(importance_df) + 1)
    
    return importance_df
