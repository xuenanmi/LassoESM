import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, RepeatedKFold
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import sem
import scipy.stats as stats


def cv_res(Xs, ys, RF_estimators, RF_depth, RF_features, Ada_estimators, Ada_lr, SVC_C, SVC_kernel, MLP_hidden_layers, MLP_batch, MLP_lr):
    random_seed = 42  
    
    RF = RandomForestClassifier(n_estimators=RF_estimators, max_depth=RF_depth, max_features=RF_features, random_state=random_seed)
    Ada = AdaBoostClassifier(n_estimators=Ada_estimators, learning_rate=Ada_lr, random_state=random_seed)
    SVM = SVC(C=SVC_C, kernel=SVC_kernel, random_state=random_seed)
    MLP = MLPClassifier(hidden_layer_sizes=MLP_hidden_layers, batch_size=MLP_batch, learning_rate_init=MLP_lr, early_stopping=True, max_iter=1000, random_state=random_seed)

    cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=random_seed)
    MLP_CV = cross_val_score(MLP, Xs, ys, cv=cv, n_jobs=-1, scoring='balanced_accuracy')
    RF_CV = cross_val_score(RF, Xs, ys, cv=cv, n_jobs=-1, scoring='balanced_accuracy')
    Ada_CV = cross_val_score(Ada, Xs, ys, cv=cv, n_jobs=-1, scoring='balanced_accuracy')
    SVC_CV = cross_val_score(SVM, Xs, ys, cv=cv, n_jobs=-1, scoring='balanced_accuracy')

    cv_res = {
        'RF': np.mean(RF_CV.reshape(-1, 10), axis=1),
        'Ada': np.mean(Ada_CV.reshape(-1, 10), axis=1),
        'SVC': np.mean(SVC_CV.reshape(-1, 10), axis=1),
        'MLP': np.mean(MLP_CV.reshape(-1, 10), axis=1)
    }
    #print(cv_res)
    return cv_res


if __name__ == '__main__':
    # Load existing data and embeddings
    one_hot_Xs = np.load('../data/FusA_one_hot_encoding.npy')
    vanilla_Xs = np.load('../data/FusA_VanillaESM.npy')
    Peptide_Xs = np.load('../data/FusA_PeptideESM_650M.npy')
    rodeo_Xs = np.load('../data/FusA_LassoESM.npy')

    data = pd.read_csv('../data/FusA_tolerance_dataset.csv')
    ys = data.iloc[:, 1].tolist()
     
    # Call cv_res and get results for each embedding type
    res_one_hot = cv_res(one_hot_Xs, ys, 100, 20, 'log2', 100, 1, 1, 'rbf', 64, 64, 0.01)
    print('Done one_hot')
    res_vanilla = cv_res(vanilla_Xs, ys, 100, 50, 'sqrt', 200, 0.1, 10, 'linear', (512, 64), 16, 0.01)
    print('Done')
    res_peptide = cv_res(Peptide_Xs, ys, 100, 10, 'sqrt', 200, 0.1, 10, 'linear', (256, 32), 16, 0.001)
    print('Done')
    res_rodeo = cv_res(rodeo_Xs, ys, 200, 20, 'sqrt', 200, 0.1, 10, 'linear', 256, 16, 0.001)
    print('Done')

    # Create dataframes for each embedding type
    one_hot = pd.DataFrame(res_one_hot)
    vanilla_ESM = pd.DataFrame(res_vanilla)
    Peptide_ESM = pd.DataFrame(res_peptide)
    rodeo_ESM = pd.DataFrame(res_rodeo)

    # Add a column for each embedding type
    one_hot['Model'] = 'One hot'
    vanilla_ESM['Model'] = 'Vanilla_ESM'
    Peptide_ESM['Model'] = 'Peptide_ESM'
    rodeo_ESM['Model'] = 'Lasso_ESM'

    # Concatenate all results
    result = pd.concat([one_hot, vanilla_ESM, Peptide_ESM, rodeo_ESM], ignore_index=True) 
    print(result)
    result.to_csv('diff_4_embs_cv_BA_res.csv', index=False)
    
