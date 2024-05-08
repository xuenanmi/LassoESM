import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, cross_val_score, RepeatedKFold, train_test_split

def percentage_true_prediction(test_fraction):
    Xs = np.load('../data/FusA_embs_from_RODEO_ESM_650M_lr_5e-05_batch_size_8.npy')
    data = pd.read_excel('../data/231130_FusA_Mutants_SEBedit.xlsx')
    ys = data.iloc[:,1].tolist()
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, stratify=ys, test_size= test_fraction)
    print(len(ys_test))
    pca = PCA(n_components = 100) 
    Xs_train = pca.fit_transform(Xs_train)
    Xs_test = pca.transform(Xs_test)
    opt_SVC = SVC(C = 10, kernel = 'linear', probability=True).fit(Xs_train, ys_train)
    ys_pred_prob = opt_SVC.predict_proba(Xs_test)
    ys_pred = opt_SVC.predict(Xs_test)
    ys_pred_prob_ls = np.round(ys_pred_prob[:,1], 1)
    ys_pred = ys_pred.tolist()

    true_pred_res, false_pred_res = [], []
    for i in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        idx = np.where(ys_pred_prob_ls == i)[0].tolist()
        #print(len(idx))
        ys_pred_tmp = np.array([ys_pred[j] for j in idx])
        ys_test_tmp = np.array([ys_test[k] for k in idx])
        true_pred = np.mean(ys_pred_tmp == ys_test_tmp)
        false_pred = 1 - true_pred
        true_pred_res.append(true_pred)
        false_pred_res.append(false_pred)

    return true_pred_res, false_pred_res

if __name__ == "__main__":
   true_res = []
   false_res = []
   for k in range(10):   
       true_pred_res, false_pred_res = percentage_true_prediction(0.2)
       true_res.append(true_pred_res)
       false_res.append(false_pred_res)
   
   true_mean = np.mean(true_res, axis = 0)
   true_sd = np.std(true_res, axis = 0)
   
   f, axs = plt.subplots(ncols=1, nrows=1, figsize=(10,7))
   # make scatter plot
   x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
   plt.plot(x, true_mean, marker='o', linestyle='-', color='#8F66C2', label='LassoESM', linewidth=4, markersize=6)    
   plt.fill_between(x, true_mean - true_sd, true_mean + true_sd, color='#C4B5D9', alpha = 0.6)      
   #plt.plot(x, true_mean, marker='o', linestyle='-', color='#87B765', label='PeptideESM', linewidth=4, markersize=6) 
   #plt.fill_between(x, true_mean - true_sd, true_mean + true_sd, color='#BDD7AB', alpha = 0.6)      

   axs.spines['bottom'].set_linewidth(2)
   axs.spines['left'].set_linewidth(2)
   axs.spines['top'].set_linewidth(2)
   axs.spines['right'].set_linewidth(2)
   plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],fontsize=15)
   plt.yticks(fontsize=15)
   plt.xlim(-0.05, 1.05)
   plt.xlabel('Predicted probability',  fontsize=20)
   plt.ylabel('True prediction percentage',  fontsize=20)
   plt.legend(fontsize = 'xx-large', loc = 'lower right')
   plt.tight_layout()
   plt.savefig('LassoESM_uncertainty_analysis.png', dpi = 300)

