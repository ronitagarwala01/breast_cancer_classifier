#importing relevant libraries
import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mygene

mg = mygene.MyGeneInfo()


#current directory
current_path = '/Users/ramanarayanankizhuttil/Documents/PhD/Courses/Spring25/Beng 203/Project'
os.chdir(current_path)


# Get training and validation data

#training_df = pd.read_csv('onco_tpm_431genes.csv')

#validation_df = pd.read_csv('validation_dataset_onco.csv')

training_df = pd.read_csv('validation_dataset_onco.csv')
validation_df = pd.read_csv('onco_tpm_431genes.csv')

print("Validation data shape:", validation_df.shape)
validation_df.head()


# breast_ensembl = ['ENSG00000091831', 'ENSG00000148773', 'ENSG00000141510', 'ENSG00000026508', 'ENSG00000141736', 'ENSG00000136997', 'ENSG00000089685']
# breast_ensembl = breast_ensembl[2::]


#a list of 17 most prominent cancer/breast cancer genes
breast_ensembl = ['ENSG00000091831', 'ENSG00000100985', 'ENSG00000148773', 'ENSG00000141510', 'ENSG00000117399', 'ENSG00000026508', 'ENSG00000087586', 'ENSG00000134057', 'ENSG00000138180', 'ENSG00000141736', 'ENSG00000126561', 'ENSG00000136997', 'ENSG00000080986', 'ENSG00000142945', 'ENSG00000105173', 'ENSG00000166851', 'ENSG00000089685']
#breast_ensembl = breast_ensembl[10::]

print(len(breast_ensembl))

# Split features and class labels

x_train = training_df[breast_ensembl]
y_train = training_df['target']

x_valid = validation_df[breast_ensembl]
y_valid = validation_df['target']


# A function that goes through each subset of above breast cancer genes
# to give a combination that maximizes roc-auc
def find_optimal_gene_combination(x_train, y_train, x_valid, y_valid, 
                                 gene_names, n_features=5, 
                                 model_type='logistic'):
    """
    Find the best combination of n_features genes that maximizes AUC
    """
    
    
    
    
    
    best_auc = 0
    best_genes = None
    best_model = None
    results = []
    
    # Calculate class weight for models
    class_counts = y_train.value_counts()
    scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) == 2 else 1
    
    # Test all combinations
    for i, gene_indices in enumerate(combinations(range(len(gene_names)), n_features)):
        
        # Progress tracking
        if (i + 1) % 1000 == 0:
            print(f"   Tested {i+1:,} combinations... Current best AUC: {best_auc:.4f}")
        
        # Select features
        selected_genes = [gene_names[j] for j in gene_indices]
        x_train_subset = x_train[selected_genes]
        x_valid_subset = x_valid[selected_genes]
        
        try:
            # Choose model
            if model_type == 'logistic':
                model = LogisticRegression(
                    penalty='l2', 
                    class_weight='balanced',
                    random_state=42,
                    max_iter=1000
                )
            elif model_type == 'xgboost':
                model = xgb.XGBClassifier(
                    n_estimators=30,
                    max_depth=3,
                    learning_rate=0.1,
                    reg_alpha=1.0,
                    reg_lambda=1.0,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    eval_metric='logloss'
                )
            elif model_type == 'random_forest':
                model = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=5,
                    class_weight='balanced',
                    random_state=42
                )
            elif model_type == 'svm_rbf':
                model = SVC(
                    kernel='rbf',
                    class_weight='balanced',
                    probability=True,
                    random_state=42
                )
            
            # Train model
            model.fit(x_train_subset, y_train)
            
            # Cross-validation AUC
            cv_scores = cross_val_score(model, x_train_subset, y_train, 
                                      cv=5, scoring='roc_auc')
            cv_auc = cv_scores.mean()
            
            # Validation AUC
            y_valid_proba = model.predict_proba(x_valid_subset)[:, 1]
            valid_auc = roc_auc_score(y_valid, y_valid_proba)
            
            # Training AUC
            y_train_proba = model.predict_proba(x_train_subset)[:, 1]
            train_auc = roc_auc_score(y_train, y_train_proba)
            
            # Store results
            results.append({
                'genes': selected_genes,
                'gene_indices': gene_indices,
                'cv_auc': cv_auc,
                'cv_std': cv_scores.std(),
                'train_auc': train_auc,
                'valid_auc': valid_auc,
                'overfitting': train_auc - valid_auc
            })
            
            # Update best if this is better
            if valid_auc > best_auc:
                best_auc = valid_auc
                best_genes = selected_genes
                best_model = model
                
        except Exception as e:
            # Skip combinations that cause errors
            continue
    
    
    print(f"   Best validation AUC: {best_auc:.4f}")
    print(f"   Best gene combination: {best_genes}")
    
    # Sort results by validation AUC
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('valid_auc', ascending=False)
    
    return results_df, best_genes, best_model





# Test models
for model_type in ['random_forest']:
    print(f"\n Testing {model_type.upper()} model:")
    
    results_df, best_genes, best_model = find_optimal_gene_combination(
        x_train, y_train, x_valid, y_valid, 
        breast_ensembl, n_features=10, model_type=model_type)
      
    
    # Test the best combination
    print(f"\n BEST {model_type.upper()} COMBINATION:")
    best_result = results_df.iloc[0]
    print(f"   Genes: {best_result['genes']}")
    print(f"   Validation AUC: {best_result['valid_auc']:.4f}")
    print(f"   Cross-validation: {best_result['cv_auc']:.4f} ± {best_result['cv_std']:.4f}")
    print(f"   Overfitting gap: {best_result['overfitting']:.4f}")
    
    best_ensemble = best_result['genes']
    
    
    
    best_gene_ids = mg.querymany(best_ensemble, scopes='ensembl.gene', fields='symbol', species='human')
    best_gene = [i['symbol'] for i in best_gene_ids]
    print(best_gene)
    
    
    
    
    
    
    
    
    
