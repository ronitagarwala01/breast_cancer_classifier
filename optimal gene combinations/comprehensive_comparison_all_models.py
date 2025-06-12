

import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                           roc_curve, auc, roc_auc_score, precision_score, recall_score, f1_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')


# CONFIGURATION AND DATA LOADING


def load_and_prepare_data():
    """Load and prepare training and validation datasets"""
    
    # Set working directory
    current_path = '/Users/ramanarayanankizhuttil/Documents/PhD/Courses/Spring25/Beng 203/Project'
    os.chdir(current_path)
    
    # Load data
    
    
    training_df = pd.read_csv('onco_tpm_431genes.csv')
    validation_df = pd.read_csv('validation_dataset_onco.csv')
    
    # training_df = pd.read_csv('validation_dataset_onco.csv')
    # validation_df = pd.read_csv('onco_tpm_431genes.csv')
    
    print(f"Training data shape: {training_df.shape}")
    print(f"Validation data shape: {validation_df.shape}")
    
    # Define gene features
    common_genes_across = ['ENSG00000141736', 'ENSG00000134057', 'ENSG00000126561', 
                          'ENSG00000166851', 'ENSG00000080986']
    
    
    
    # Prepare features and targets
    x_train = training_df[common_genes_across]
    y_train = training_df['target']
    x_valid = validation_df[common_genes_across]
    y_valid = validation_df['target']
    
   
    
    # Calculate class weights for imbalanced data
    class_counts = y_train.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) == 2 else 1
    
    return x_train, y_train, x_valid, y_valid, scale_pos_weight

def calculate_metrics(model_name, best_estimator, cv_score, y_valid, y_valid_pred, y_valid_proba, best_params):
    """Calculate comprehensive metrics for a trained model"""
    
    metrics = {
        'Model': model_name,
        'Best_Params': str(best_params),
        'CV_AUC': cv_score,
        'Validation_Accuracy': accuracy_score(y_valid, y_valid_pred),
        'Validation_Precision': precision_score(y_valid, y_valid_pred, average='weighted'),
        'Validation_Recall': recall_score(y_valid, y_valid_pred, average='weighted'),
        'Validation_F1': f1_score(y_valid, y_valid_pred, average='weighted'),
        'Validation_AUC': roc_auc_score(y_valid, y_valid_proba)
    }
    
    return metrics


# CLASSIFIER TRAINING FUNCTIONS


def train_logistic_regression(x_train, y_train, x_valid, y_valid, cv_strategy):
    """Train and optimize Logistic Regression classifier"""
    
    print("\nTraining Logistic Regression...")
    
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000],
        'class_weight': ['balanced', None]
    }
    
    grid_search = GridSearchCV(
        LogisticRegression(random_state=42),
        param_grid,
        cv=cv_strategy,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_valid_pred = best_model.predict(x_valid)
    y_valid_proba = best_model.predict_proba(x_valid)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(
        'Logistic_Regression', 
        best_model, 
        grid_search.best_score_, 
        y_valid, 
        y_valid_pred, 
        y_valid_proba,
        grid_search.best_params_
    )
    
    print(f"Logistic Regression - Best CV AUC: {grid_search.best_score_:.4f}")
    print(f"   Validation AUC: {metrics['Validation_AUC']:.4f}")
    
    return metrics, y_valid_proba

def train_xgboost(x_train, y_train, x_valid, y_valid, cv_strategy, scale_pos_weight):
    """Train and optimize XGBoost classifier"""
    
    print("\nTraining XGBoost...")
    
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 4, 5, 6],
        'learning_rate': [0.05, 0.1, 0.15],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 1.5, 2],
        'scale_pos_weight': [scale_pos_weight]
    }
    
    grid_search = GridSearchCV(
        xgb.XGBClassifier(
            random_state=42,
            eval_metric='logloss'
        ),
        param_grid,
        cv=cv_strategy,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_valid_pred = best_model.predict(x_valid)
    y_valid_proba = best_model.predict_proba(x_valid)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(
        'XGBoost', 
        best_model, 
        grid_search.best_score_, 
        y_valid, 
        y_valid_pred, 
        y_valid_proba,
        grid_search.best_params_
    )
    
    print(f"XGBoost - Best CV AUC: {grid_search.best_score_:.4f}")
    print(f"   Validation AUC: {metrics['Validation_AUC']:.4f}")
    
    return metrics, y_valid_proba

def train_random_forest(x_train, y_train, x_valid, y_valid, cv_strategy):
    """Train and optimize Random Forest classifier"""
    
    print("\nTraining Random Forest...")
    
    param_grid = {
        'n_estimators': [50, 100, 150, 200],
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=cv_strategy,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_valid_pred = best_model.predict(x_valid)
    y_valid_proba = best_model.predict_proba(x_valid)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(
        'Random_Forest', 
        best_model, 
        grid_search.best_score_, 
        y_valid, 
        y_valid_pred, 
        y_valid_proba,
        grid_search.best_params_
    )
    
    print(f"Random Forest - Best CV AUC: {grid_search.best_score_:.4f}")
    print(f"   Validation AUC: {metrics['Validation_AUC']:.4f}")
    
    return metrics, y_valid_proba

def train_svm_rbf(x_train, y_train, x_valid, y_valid, cv_strategy):
    """Train and optimize SVM with RBF kernel"""
    
    print("\nTraining SVM with RBF kernel...")
    
    param_grid = {
    'C': [0.01, 0.1, 1, 10],  # Smaller C values for regularization
    'gamma': ['scale', 'auto', 0.001, 0.01],  # More conservative gamma
    'class_weight': ['balanced']  # Always use balanced for imbalanced data
}
    
    grid_search = GridSearchCV(
        SVC(kernel='rbf', probability=True, random_state=42),
        param_grid,
        cv=cv_strategy,
        scoring='roc_auc',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Predictions
    y_valid_pred = best_model.predict(x_valid)
    y_valid_proba = best_model.predict_proba(x_valid)[:, 1]
    
    # Calculate metrics
    metrics = calculate_metrics(
        'SVM_RBF', 
        best_model, 
        grid_search.best_score_, 
        y_valid, 
        y_valid_pred, 
        y_valid_proba,
        grid_search.best_params_
    )
    
    print(f"SVM RBF - Best CV AUC: {grid_search.best_score_:.4f}")
    print(f"   Validation AUC: {metrics['Validation_AUC']:.4f}")
    
    return metrics, y_valid_proba


# VISUALIZATION AND RESULTS FUNCTIONS


def generate_roc_plot(roc_data, save_path='roc_curves_comparison.png'):
    """Generate and save ROC curve comparison plot"""
    
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange']
    model_names = ['Logistic_Regression', 'XGBoost', 'Random_Forest', 'SVM_RBF']
    
    for i, model_name in enumerate(model_names):
        if model_name in roc_data:
            fpr, tpr, auc_score = roc_data[model_name]
            plt.plot(fpr, tpr, color=colors[i], lw=2, 
                     label=f'{model_name.replace("_", " ")} (AUC = {auc_score:.3f})')
    
    # Add diagonal line for random classifier
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison - All Classifiers', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return save_path

def save_results(results):
    """Save results to CSV files"""
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).T
    results_df = results_df.round(4)
    
    # Save individual CSV files for each classifier
    saved_files = []
    for model_name, metrics in results.items():
        filename = f'{model_name}_metrics.csv'
        model_df = pd.DataFrame([metrics])
        model_df.to_csv(filename, index=False)
        saved_files.append(filename)
        print(f"Saved: {filename}")
    
    # Save combined results
    combined_filename = 'all_classifiers_comparison.csv'
    results_df.to_csv(combined_filename)
    saved_files.append(combined_filename)
    print(f"Saved: {combined_filename}")
    
    return results_df, saved_files

def print_summary(results_df):
    """Print comprehensive performance summary"""
    
    
    print("FINAL PERFORMANCE SUMMARY")
    
    
    # Display key metrics
    display_cols = ['CV_AUC', 'Validation_AUC', 'Validation_Accuracy', 
                   'Validation_Precision', 'Validation_Recall', 'Validation_F1']
    print(results_df[display_cols])
    
    # Find best performing model
    best_model = results_df['Validation_AUC'].idxmax()
    best_auc = results_df.loc[best_model, 'Validation_AUC']
    
    print(f"\nBest performing model: {best_model}")
    print(f"Best validation AUC: {best_auc:.4f}")


# MAIN EXECUTION

def main():
    """Main execution function"""
    
    
    
    # Load and prepare data
    x_train, y_train, x_valid, y_valid, scale_pos_weight = load_and_prepare_data()
    
    # Initialize storage
    results = {}
    roc_data = {}
    
    # Define cross-validation strategy
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Train all classifiers
    print("\n" + "="*50)
    print("TRAINING CLASSIFIERS")
    print("="*50)
    
    # 1. Logistic Regression
    lr_metrics, lr_proba = train_logistic_regression(x_train, y_train, x_valid, y_valid, cv_strategy)
    results['Logistic_Regression'] = lr_metrics
    fpr_lr, tpr_lr, _ = roc_curve(y_valid, lr_proba)
    roc_data['Logistic_Regression'] = (fpr_lr, tpr_lr, lr_metrics['Validation_AUC'])
    
    # 2. XGBoost
    xgb_metrics, xgb_proba = train_xgboost(x_train, y_train, x_valid, y_valid, cv_strategy, scale_pos_weight)
    results['XGBoost'] = xgb_metrics
    fpr_xgb, tpr_xgb, _ = roc_curve(y_valid, xgb_proba)
    roc_data['XGBoost'] = (fpr_xgb, tpr_xgb, xgb_metrics['Validation_AUC'])
    
    # 3. Random Forest
    rf_metrics, rf_proba = train_random_forest(x_train, y_train, x_valid, y_valid, cv_strategy)
    results['Random_Forest'] = rf_metrics
    fpr_rf, tpr_rf, _ = roc_curve(y_valid, rf_proba)
    roc_data['Random_Forest'] = (fpr_rf, tpr_rf, rf_metrics['Validation_AUC'])
    
    # 4. SVM RBF
    svm_metrics, svm_proba = train_svm_rbf(x_train, y_train, x_valid, y_valid, cv_strategy)
    results['SVM_RBF'] = svm_metrics
    fpr_svm, tpr_svm, _ = roc_curve(y_valid, svm_proba)
    roc_data['SVM_RBF'] = (fpr_svm, tpr_svm, svm_metrics['Validation_AUC'])
    
    # Generate results and visualizations
    print("\n" + "="*50)
    print("GENERATING RESULTS AND VISUALIZATIONS")
    print("="*50)
    
    # Save results to CSV
    results_df, saved_files = save_results(results)
    
    # Generate ROC plot
    plot_path = generate_roc_plot(roc_data)
    print(f"Saved: {plot_path}")
    
    # Print summary
    print_summary(results_df)
    
    # Final output summary
    print("\nAnalysis complete! Generated files:")
    for file in saved_files:
        print(f"- {file}")
    print(f"- {plot_path}")

if __name__ == "__main__":
    main()
