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

def cv_res(random_seed,Xs, ys, RF_estimators, RF_depth, RF_features, Ada_estimators, Ada_lr, SVC_C, SVC_kernel, MLP_hidden_layers, MLP_batch, MLP_lr):
    MLP = MLPClassifier(hidden_layer_sizes= MLP_hidden_layers, batch_size = MLP_batch, learning_rate_init = MLP_lr, early_stopping=True, max_iter=1000, random_state = random_seed)
    KNN = KNeighborsClassifier(n_neighbors = KNN_neighbors , weights= KNN_weights)
    RF = RandomForestClassifier(n_estimators = RF_estimators, max_depth = RF_depth, max_features = RF_features,random_state = random_seed)
    Ada = AdaBoostClassifier(n_estimators = Ada_estimators, learning_rate = Ada_lr ,random_state = random_seed)
    SVM = SVC(C = SVC_C, kernel = SVC_kernel,random_state = random_seed)

    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=random_seed)
    MLP_CV = cross_val_score(MLP, Xs, ys, cv = cv, n_jobs = -1, scoring = 'balanced_accuracy')
    pca = PCA(n_components = 100) 
    Xs_pca = pca.fit_transform(Xs)
    RF_CV = cross_val_score(RF, Xs_pca, ys, cv = cv, n_jobs = -1, scoring = 'balanced_accuracy')
    Ada_CV = cross_val_score(Ada, Xs_pca, ys, cv = cv, n_jobs = -1, scoring = 'balanced_accuracy')
    SVC_CV = cross_val_score(SVM, Xs_pca, ys, cv = cv, n_jobs = -1, scoring = 'balanced_accuracy')
    
    cv_res = [np.mean(KNN_CV), np.mean(RF_CV), np.mean(Ada_CV), np.mean(SVC_CV), np.mean(MLP_CV)]
    return cv_res

if __name__ == '__main__':
    
   random_seed_ls = random.sample(range(100), 3)
   ##vanilla ESM    
   vanilla_Xs = np.load('vanilla_ESM_embs.npy')
   data = pd.read_excel('231130_FusA_Mutants_SEBedit.xlsx')
   ys = data.iloc[:,1].tolist()
   ##peptide ESM 
   Peptide_Xs = np.load('FusA_embs_from_PeptideESM_650M.npy')
   ##Lasso ESM Galore
   rodeo_Xs = np.load('FusA_embs_from_RODEO_ESM_650M_lr_5e-05_batch_size_8.npy')
   

   vanilla_ESM_cv_res = []
   Peptide_ESM_cv_res = []
   rodeo_ESM_cv_res = []

   for i in random_seed_ls:
       vanilla_ESM_cv_res.append(cv_res(i, vanilla_Xs, ys, 100, 50, 'sqrt', 200, 0.1, 10, 'linear', (512,64), 16, 0.01))
       Peptide_ESM_cv_res.append(cv_res(i, Peptide_Xs, ys, 100, 10, 'sqrt', 200, 0.1, 10, 'linear', (256,32), 16, 0.001))
       rodeo_ESM_cv_res.append(cv_res(i, rodeo_Xs, ys,     200, 20, 'sqrt', 200, 0.1, 10, 'linear', 256, 16, 0.001))
       
   vanilla_ESM = pd.DataFrame(vanilla_ESM_cv_res, columns = ['RF','Ada','SVC','MLP'])
   Peptide_ESM = pd.DataFrame(Peptide_ESM_cv_res, columns = ['RF','Ada','SVC','MLP'])
   RODEO_ESM = pd.DataFrame(rodeo_ESM_cv_res, columns = ['RF','Ada','SVC','MLP'])
   
   vanilla_ESM['Model'] = 'Vanilla_ESM'
   Peptide_ESM['Model'] = 'Peptide_ESM'
   RODEO_ESM['Model'] = 'Lasso_ESM'
   
   result = pd.concat([vanilla_ESM, Peptide_ESM, RODEO_ESM], ignore_index=True) 
   print(result)
   result.to_csv('diff_3_embs_cv_BA_res.csv', index=False)
   
   result = pd.read_csv('diff_3_embs_cv_BA_res.csv')
   model = result['Model'].tolist()
   reshape_df = pd.DataFrame(result.iloc[:,:4].values.ravel('F'))
   reshape_df.columns = ['BA']
   reshape_df['Downstream_model'] = ['RF'] * 9 + ['AdaBoost'] * 9 + ['SVC'] * 9 + ['MLP'] * 9
   reshape_df['Embeddings'] = 4*model
   f, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,7))
   flatui = ["#82ee98","#f7df94","#f493cf"]

   sns.set_palette(sns.color_palette(flatui))
   aa = sns.barplot(x="Downstream_model", y="BA",hue="Embeddings",data=reshape_df, alpha = 1.0,capsize=.1,errorbar=('ci', 68), errwidth=1, linewidth = 1) #, edgecolor=1)
   for patch in aa.patches:
       clr = patch.get_facecolor()
       patch.set_edgecolor(clr)
   ax.spines['bottom'].set_linewidth(2)
   ax.spines['left'].set_linewidth(2)
   ax.spines['top'].set_linewidth(2)
   ax.spines['right'].set_linewidth(2)
   plt.ylim(0.3,1.0)
   plt.xticks(fontsize=18)
   plt.yticks(fontsize=18)
   plt.xlabel('Downstream Models', fontsize = 20)
   plt.ylabel('Balanced Accuracy', fontsize = 20)
   #ax.get_legend().remove()
   plt.savefig("Diff_3embs_cv_BA_res.png", dpi =300) 
   
   for i in range(4):
       print(result.columns[i])
       print(stats.ttest_ind(result.iloc[:3,i], result.iloc[3:6,i], equal_var = False))
       print(stats.ttest_ind(result.iloc[:3,i], result.iloc[6:9,i], equal_var = False)) 
