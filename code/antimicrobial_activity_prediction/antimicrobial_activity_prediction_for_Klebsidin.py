import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import AdaBoostRegressor
from sklearn.pipeline import Pipeline

# Load the dataset
data = pd.read_csv('../data/klebsidin_enrichment_score.csv')
ys = data.iloc[:, 1].tolist()

# Define a dictionary of embeddings and corresponding best models
models = {
    'VanillaESM': {
        'Xs': np.load('../data/klebsidin_embs_from_VanillaESM.npy'),
        'best_model': Pipeline(steps=[('regressor', AdaBoostRegressor(learning_rate=1, n_estimators=100))])
    },
    'PeptideESM': {
        'Xs': np.load('../data/klebsidin_embs_from_PeptideESM.npy'),
        'best_model': Pipeline(steps=[('regressor', AdaBoostRegressor(learning_rate=1, n_estimators=100))])
    },
    'LassoESM': {
        'Xs': np.load('../data/klebsidin_embs_from_LassoESM.npy'),
        'best_model': Pipeline(steps=[('regressor', AdaBoostRegressor(learning_rate=1, n_estimators=200))])
    }
}

# Initialize variables to store results
all_results = []

# Loop over each feature set and its best model
for model_name, model_data in models.items():
    Xs = model_data['Xs']
    best_model = model_data['best_model']
    
    results = {
        'Embeddings': [],
        'Repeat': [],
        'Pearson': [],
        'Spearman': [],
        'MAE': []
    }
    
    # Perform 10 repeats of 10-fold cross-validation
    for repeat in range(1, 11):
        print(f"Starting repeat {repeat} for {model_name}")
        kf = KFold(n_splits=10, shuffle=True, random_state=random.seed(repeat))
        
        y_pred = cross_val_predict(best_model, Xs, ys, cv=kf, n_jobs=-1)
        
        # Calculate evaluation metrics
        pearson_corr = pearsonr(ys, y_pred)[0]
        spearman_corr = spearmanr(ys, y_pred)[0]
        mae = mean_absolute_error(ys, y_pred)
        
        # Store the results
        results['Embeddings'].append(model_name)
        results['Repeat'].append(repeat)
        results['Pearson'].append(pearson_corr)
        results['Spearman'].append(spearman_corr)
        results['MAE'].append(mae)
    
    # Convert results to DataFrame and append to all_results
    results_df = pd.DataFrame(results)
    all_results.append(results_df)

# Concatenate all results and save them to a CSV file
final_results_df = pd.concat(all_results, ignore_index=True)
final_results_df.to_csv('All_Feature_Sets_AdaBoost_10_repeats_10_fold_CV_results.csv', index=False)

print(final_results_df)

