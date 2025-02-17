import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, mean_squared_error
from sklearn.model_selection import learning_curve, cross_val_score
#from sklearn.inspection import plot_partial_dependence
from sklearn.tree import plot_tree
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve


# Function to train RandomForest model and generate performance metrics
def train_random_forest(X_train, y_train, X_test, y_test):
    # Initialize RandomForest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on train and test data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    
    return model, mae_train, mae_test, r2_train, r2_test, y_train_pred, y_test_pred

# Function to plot Model Performance Comparison
def plot_performance(mae_train, mae_test, r2_train, r2_test):
    metrics = {
        'Training': [mae_train, r2_train],
        'Testing': [mae_test, r2_test]
    }
    metrics_df = pd.DataFrame(metrics, index=['MAE', 'R²'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    metrics_df.plot(kind='bar', color=['blue', 'green'], ax=ax)
    ax.set_title('Model Performance Comparison (Training vs Testing)')
    ax.set_ylabel('Score')
    
    return fig

# Plot Feature Importance
def plot_feature_importance(model, X_train):
    feature_importances = model.feature_importances_
    features = X_train.columns

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=features, ax=ax)
    ax.set_title('Feature Importance')
    ax.set_xlabel('Importance')
    ax.set_ylabel('Feature')
    
    return fig

# Plot Learning Curve
def plot_learning_curve(model, X_train, y_train):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_sizes, -train_scores.mean(axis=1), label='Training Error')
    ax.plot(train_sizes, -test_scores.mean(axis=1), label='Validation Error')
    ax.set_title('Learning Curve')
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('Error (MSE)')
    ax.legend()
    
    return fig

# Cross-Validation Results (Box Plot)
def plot_cross_validation(model, X_train, y_train):
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(data=cv_scores, color='lightblue', ax=ax)
    ax.set_title('Cross-Validation Performance')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Negative Mean Squared Error')
    
    return fig

# Actual vs Predicted Plot
def plot_actual_vs_predicted(y_test, y_test_pred):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_test, y_test_pred, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)  # Perfect prediction line
    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    ax.set_title('Actual vs Predicted')
    
    return fig

# Error Distribution Plot
def plot_error_distribution(y_test, y_test_pred):
    errors = y_test - y_test_pred

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(errors, kde=True, color='purple', ax=ax)
    ax.set_title('Error Distribution')
    ax.set_xlabel('Error (Actual - Predicted)')
    ax.set_ylabel('Frequency')
    
    return fig

# Random Forest Tree Visualization
def plot_random_forest_tree(model, X_train):
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(model.estimators_[0], filled=True, feature_names=X_train.columns, ax=ax)
    ax.set_title('Random Forest - Decision Tree')
    
    return fig

# ROC Curve (for classification tasks)
def plot_roc_curve(y_test, model, X_test):
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc='lower right')
    
    return fig

# Calibration Curve
def plot_calibration_curve(y_test, model, X_test):
    prob_true, prob_pred = calibration_curve(y_test, model.predict_proba(X_test)[:, 1], n_bins=10)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(prob_pred, prob_true, marker='o', label="Random Forest")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfectly calibrated")
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration Curve')
    ax.legend()
    
    return fig
