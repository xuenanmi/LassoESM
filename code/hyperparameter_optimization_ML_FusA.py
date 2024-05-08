import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.pipeline import Pipeline
from datasets import Dataset
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, average_precision_score, confusion_matrix, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


data = pd.read_excel("../data/231130_FusA_Mutants_SEBedit.xlsx")
seqs = data.iloc[:,0].tolist()

Xs = np.load('../data/FusA_embs_from_RODEO_ESM_650M_lr_5e-05_batch_size_8.npy')
ys = data.iloc[:,1].tolist()

print("Optimization for Models trained on FusA")

model_list = [RandomForestClassifier, AdaBoostClassifier, SVC, MLPClassifier]
model_names = ['RF', 'AdaBoost', 'SVC', 'MLP']
parameters_list = [
    {'classifier__n_neighbors': [5, 10, 25, 50], 'classifier__weights': ('uniform', 'distance')},
    {'classifier__n_estimators': [20, 50, 100, 200], 'classifier__max_depth': [10,20,50,100], 'classifier__max_features':('sqrt','log2')},
    {'classifier__n_estimators': [20, 50, 100, 200], 'classifier__learning_rate': [0.1, 1, 5, 10]},
    {'classifier__kernel':('linear', 'rbf'), 'classifier__C':[0.1, 1, 10]},
    {'classifier__hidden_layer_sizes': [32,64,128,256,512, (512, 64), (256, 32), (128,32)], 'classifier__batch_size':[16,32], 'classifier__learning_rate_init':[0.01, 0.001], 'classifier__max_iter':[1000], 'classifier__early_stopping':[True]},
]

for model, name, parameters in zip(model_list, model_names, parameters_list):
    result_list = []
    steps = [
        ('pca', PCA(n_components=100)),
        ('classifier', model())
    ]
    pipeline = Pipeline(steps)
    grid_search = GridSearchCV(pipeline, parameters,cv=10, scoring='roc_auc')
    grid_search.fit(Xs, ys)
    result_list.append(pd.DataFrame.from_dict(grid_search.cv_results_))
    aa = result_list[0].sort_values('rank_test_score')
    print(f'Best paramters for {name} {grid_search.best_params_}')
    print(f'Best model for {name} {grid_search.best_estimator_}')
    print(aa.iloc[:5,4:])
    
