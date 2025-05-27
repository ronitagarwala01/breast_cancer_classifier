import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# Load the data
training_df = pd.read_csv('../data/combined_tpm_5000genes.csv')
validation_df = pd.read_csv('../data/validation_dataset.csv')

# Split features and class labels
x_train = training_df.iloc[:, :5000]
y_train = training_df['target']

x_valid = validation_df.iloc[:, :5000]
y_valid = validation_df['target']

# Initialize k-fold cross validation
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# Initialize lists to store performance metrics
cv_train_accuracies = []
cv_val_accuracies = []
cv_reports = []

# Perform k-fold cross validation
print("Performing 5-fold Cross Validation...")
for fold, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
    # Split data for this fold
    X_train_fold = x_train.iloc[train_idx]
    y_train_fold = y_train.iloc[train_idx]
    X_val_fold = x_train.iloc[val_idx]
    y_val_fold = y_train.iloc[val_idx]
    
    # Train model on this fold
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_classifier.fit(X_train_fold, y_train_fold)
    
    # Make predictions
    y_train_pred = rf_classifier.predict(X_train_fold)
    y_val_pred = rf_classifier.predict(X_val_fold)
    
    # Calculate metrics
    train_accuracy = accuracy_score(y_train_fold, y_train_pred)
    val_accuracy = accuracy_score(y_val_fold, y_val_pred)
    fold_report = classification_report(y_val_fold, y_val_pred)
    
    # Store metrics
    cv_train_accuracies.append(train_accuracy)
    cv_val_accuracies.append(val_accuracy)
    cv_reports.append(fold_report)
    
    print(f"\nFold {fold+1} Results:")
    print(f"Train Accuracy: {train_accuracy:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    print(f"Classification Report:\n{fold_report}")

# Calculate and print average performance
print("\nOverall Cross-Validation Results:")
print(f"Average Train Accuracy: {np.mean(cv_train_accuracies):.4f} (+/- {np.std(cv_train_accuracies):.4f})")
print(f"Average Validation Accuracy: {np.mean(cv_val_accuracies):.4f} (+/- {np.std(cv_val_accuracies):.4f})")

# Train final model on entire training set
print("\nTraining final model on entire training set...")
final_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
final_rf.fit(x_train, y_train)

# Evaluate on validation set
y_valid_pred = final_rf.predict(x_valid)
y_train_pred = final_rf.predict(x_train)

train_accuracy = accuracy_score(y_train, y_train_pred)
valid_accuracy = accuracy_score(y_valid, y_valid_pred)
final_report = classification_report(y_valid, y_valid_pred)

print("\nFinal Model Performance:")
print(f"Train Accuracy: {train_accuracy:.4f}")
print(f"Validation Accuracy: {valid_accuracy:.4f}")
print(f"Classification Report on Validation Set:\n{final_report}") 