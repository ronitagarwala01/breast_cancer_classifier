#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 14:57:53 2025

@author: ramanarayanankizhuttil
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os
import mygene

current_path = '/Users/ramanarayanankizhuttil/Documents/PhD/Courses/Spring25/Beng 203/Project'
os.chdir(current_path)

mg = mygene.MyGeneInfo()

ensemble_ids = pd.read_csv('bc_oncogenes.csv')

ensemble_ids_list = ensemble_ids['Ensembl Gene ID'].tolist()


results = mg.querymany(ensemble_ids_list, scopes='ensembl.gene', fields='symbol', species='human')

gene_list = []

for i in results:
    
    gene_list.append(i['symbol'])
    
    



breast_cancer_genes = [
    "TP53",
    "ESR1",
    "CD44",
    "MKI67",
    "MYC",
    "BIRC5",
    "AURKA",
    "PLK1",
    "CCNB1",
    "CCNE1",
    "STAT5A",
    "ERBB2",
    "MMP9",
    "KIF2C",
    "NDC80",
    "CEP55",
    "CDC20"
]



breast_cancer_ensembl = []

for i in results:
    
    if i['symbol'] in breast_cancer_genes:
        
        breast_cancer_ensembl.append(i['query'])



print('breast_cancer_id:', breast_cancer_ensembl)
selected = mg.querymany(breast_cancer_ensembl, scopes='ensembl.gene', fields='symbol', species='human')

selected_list = [i['symbol'] for i in selected]
print(selected_list)

## all the ensembles that were obtained using optimization with different classifiers

best_ensemble_lg_5 = ['ENSG00000134057', 'ENSG00000126561', 'ENSG00000136997', 'ENSG00000142945', 'ENSG00000166851']
best_ensemble_xg_5 = ['ENSG00000148773', 'ENSG00000134057', 'ENSG00000126561', 'ENSG00000080986', 'ENSG00000166851']
best_ensemble_xg_10 = ['ENSG00000148773', 'ENSG00000141510', 'ENSG00000087586', 'ENSG00000134057', 'ENSG00000141736', 'ENSG00000126561', 'ENSG00000136997', 'ENSG00000080986', 'ENSG00000105173', 'ENSG00000166851']

best_ensemble_lg_12 = ['ENSG00000148773', 'ENSG00000141510', 'ENSG00000087586', 'ENSG00000134057', 'ENSG00000138180', 'ENSG00000141736', 'ENSG00000126561', 'ENSG00000136997', 'ENSG00000080986', 'ENSG00000142945', 'ENSG00000105173', 'ENSG00000166851']
best_ensemble_lg_10 = ['ENSG00000148773', 'ENSG00000141510', 'ENSG00000087586', 'ENSG00000134057', 'ENSG00000141736', 'ENSG00000126561', 'ENSG00000136997', 'ENSG00000080986', 'ENSG00000142945', 'ENSG00000166851']

best_ensemble_rf_10 = ['ENSG00000100985', 'ENSG00000148773', 'ENSG00000141510', 'ENSG00000087586', 'ENSG00000134057', 'ENSG00000138180', 'ENSG00000141736', 'ENSG00000126561', 'ENSG00000080986', 'ENSG00000166851']
best_ensemble_svm_rbf = ['ENSG00000091831', 'ENSG00000148773', 'ENSG00000087586', 'ENSG00000134057', 'ENSG00000138180', 'ENSG00000141736', 'ENSG00000126561', 'ENSG00000080986', 'ENSG00000142945', 'ENSG00000166851']


#reversed
best_ensemble_lg_10_rev = ['ENSG00000148773', 'ENSG00000087586', 'ENSG00000134057', 'ENSG00000138180', 'ENSG00000141736', 'ENSG00000126561', 'ENSG00000136997', 'ENSG00000080986', 'ENSG00000142945', 'ENSG00000166851']
best_ensemble_xg_10_rev = ['ENSG00000091831', 'ENSG00000100985', 'ENSG00000134057', 'ENSG00000138180', 'ENSG00000141736', 'ENSG00000126561', 'ENSG00000136997', 'ENSG00000080986', 'ENSG00000166851', 'ENSG00000089685']
best_ensemble_svm_rbf_10_rev = ['ENSG00000091831', 'ENSG00000100985', 'ENSG00000087586', 'ENSG00000134057', 'ENSG00000138180', 'ENSG00000141736', 'ENSG00000126561', 'ENSG00000136997', 'ENSG00000080986', 'ENSG00000166851']
best_ensemble_rf_10_rev = ['ENSG00000100985', 'ENSG00000148773', 'ENSG00000117399', 'ENSG00000087586', 'ENSG00000134057', 'ENSG00000141736', 'ENSG00000126561', 'ENSG00000080986', 'ENSG00000142945', 'ENSG00000166851']


best_gene_ids = mg.querymany(best_ensemble_xg_10, scopes='ensembl.gene', fields='symbol', species='human')
best_gene = [i['symbol'] for i in best_gene_ids]
print('xg5',best_gene)


best_ensemble_list = [best_ensemble_lg_10, best_ensemble_xg_10, best_ensemble_rf_10]

common = set(best_ensemble_lg_10) & set(best_ensemble_xg_10) & set(best_ensemble_rf_10) & set(best_ensemble_svm_rbf) 

common_rev = set(best_ensemble_lg_10_rev) & set(best_ensemble_xg_10_rev) & set(best_ensemble_svm_rbf_10_rev) & set(best_ensemble_rf_10_rev)

intersection_list = list(common)
intersection_list_rev = list(common_rev)

print(len(intersection_list))
print(len(intersection_list_rev))

intersection_genes_ids = mg.querymany(intersection_list, scopes='ensembl.gene', fields='symbol', species='human')
intersection_genes_ids_rev = mg.querymany(intersection_list_rev, scopes='ensembl.gene', fields='symbol', species='human')
best_gene_intersection = [i['symbol'] for i in intersection_genes_ids]
best_gene_intersection_rev = [i['symbol'] for i in intersection_genes_ids_rev]

print('intersection', best_gene_intersection)
print('intersection_rev', best_gene_intersection_rev)

common_genes_accross = ['ENSG00000141736', 'ENSG00000134057', 'ENSG00000126561', 'ENSG00000166851', 'ENSG00000080986']



union_ensemble_lg = list(set(best_ensemble_lg_10) | set(best_ensemble_lg_10_rev))
union_ensemble_xg = list(set(best_ensemble_xg_10) | set(best_ensemble_xg_10_rev))
union_ensemble_svm = list(set(best_ensemble_svm_rbf) | set(best_ensemble_svm_rbf_10_rev))
union_ensemble_rf = list(set(best_ensemble_rf_10) | set(best_ensemble_rf_10_rev))

print(union_ensemble_lg)
print(union_ensemble_xg)
print(union_ensemble_svm)
print(union_ensemble_rf)















