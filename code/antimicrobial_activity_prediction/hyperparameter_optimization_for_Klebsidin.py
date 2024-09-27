import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.model_selection import GridSearchCV, cross_val_predict, KFold
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.neural_network import MLPRegressor

data = pd.read_csv('klebsidin_enrichment_score.csv')

Xs = np.load('klebsidin_embs_from_RODEO_ESM_650M_lr_5e-05_batch_size_8.npy')  #LassoESM embeddings
print(Xs.shape)
ys = data.iloc[:, 1].tolist()
print(len(ys))

# Define the models and their corresponding parameter grids
model_list = [SVR, RandomForestRegressor, AdaBoostRegressor, MLPRegressor]
model_names = ['SVM', 'RandomForest', 'AdaBoost', 'MLP']
parameters_list = [
    {
        'regressor__C': [0.1, 1, 10],
        'regressor__kernel': ['linear', 'rbf', 'poly','sigmoid'],
    },
    {
        'regressor__n_estimators': [100, 200, 500],
        'regressor__max_depth': [None, 10, 20, 30],
        'regressor__min_samples_split': [2, 5, 10],
    },
    {
        'regressor__n_estimators': [50, 100, 200],
        'regressor__learning_rate': [0.01, 0.1, 1],
    },
    {
        'regressor__hidden_layer_sizes': [64, 256, (512, 64), (256, 32), (128, 32)],
        'regressor__batch_size': [16, 32],
        'regressor__learning_rate_init': [0.01, 0.001],
        'regressor__max_iter': [1000],
        'regressor__early_stopping': [True]
    },
]

# Iterate over the models and their respective parameter grids
for model, name, parameters in zip(model_list, model_names, parameters_list):
    result_list = []
    steps = [
        ('regressor', model())
    ]
    pipeline = Pipeline(steps)
    grid_search = GridSearchCV(pipeline, parameters, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(Xs, ys)
    result_list.append(pd.DataFrame.from_dict(grid_search.cv_results_))
    aa = result_list[0].sort_values('rank_test_score')
    print(f'Best parameters for {name}: {grid_search.best_params_}')
    print(f'Best model for {name}: {grid_search.best_estimator_}')
    print(aa.iloc[:5, 4:])

    # Extract the best model
    best_model = grid_search.best_estimator_

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=random.seed(42))
    y_pred = cross_val_predict(best_model, Xs, ys, cv=kf, n_jobs=-1)

    # Calculate evaluation metrics
    pearson_corr = pearsonr(ys, y_pred)[0]
    spearman_corr = spearmanr(ys, y_pred)[0]
    mae = mean_absolute_error(ys, y_pred)

    print(f'Pearson correlation for {name}: {pearson_corr}')
    print(f'Spearman correlation for {name}: {spearman_corr}')
    print(f'Mean Absolute Error (MAE) for {name}: {mae}')

