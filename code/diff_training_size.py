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

def model_performance(X_test, y_test):
    pca = PCA(n_components = 100) 
    Xs_pca = pca.fit_transform(X_test)
    cv = RepeatedKFold(n_splits=10, n_repeats=3)
    SVM = SVC(C = 10, kernel = 'linear')
    SVC_CV_acc = cross_val_score(SVM, Xs_pca, y_test, cv = cv, n_jobs = 10, scoring = 'balanced_accuracy')
    SVC_CV_auc = cross_val_score(SVM, Xs_pca, y_test, cv = cv, n_jobs = 10, scoring = 'roc_auc')
    return [np.mean(SVC_CV_acc), np.mean(SVC_CV_auc)]
    
if __name__ == "__main__":
   Xs = np.load('../exp_data_embeddings/FusA_embs_from_RODEO_ESM_650M_lr_5e-05_batch_size_8.npy')
   data = pd.read_excel('231130_FusA_Mutants_SEBedit.xlsx')
   ys = data.iloc[:,1].tolist()
   diff_train_size_acc = []
   diff_train_size_auc = []
   for m in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9]:
       train_size_acc = []
       train_size_auc = []
       for i in range(10):
           X_train, X_test, y_train, y_test = train_test_split(Xs, ys, stratify=ys, test_size= m)
           print(Counter(y_test))
           cv_acc, cv_auc = model_performance(X_test, y_test)
           train_size_acc.append(cv_acc)
           train_size_auc.append(cv_auc)

       print(f"Fraction of Training size:{m}")
       diff_train_size_acc.append(train_size_acc)
       diff_train_size_auc.append(train_size_auc)
   
   x = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9]
   acc_mean, acc_sd, auc_mean, auc_sd = [], [], [], []
   for i in range(len(x)):
       acc_mean.append(np.mean(diff_train_size_acc[i]))
       acc_sd.append(np.std(diff_train_size_acc[i]))
       auc_mean.append(np.mean(diff_train_size_auc[i]))
       auc_sd.append(np.std(diff_train_size_auc[i]))
   
   # Convert lists to NumPy arrays
   acc_mean = np.array(acc_mean)
   acc_sd = np.array(acc_sd)
   auc_mean = np.array(auc_mean)
   auc_sd = np.array(auc_sd) 
   
   f, axs = plt.subplots(ncols=1, nrows=1, figsize=(10,7))
   # Plotting the first scatter plot
   plt.plot(x, auc_mean, marker='o', linestyle='-', color='#C85289', label='AUROC', linewidth=4, markersize=6)
   plt.fill_between(x, auc_mean - auc_sd, auc_mean + auc_sd, color='#E59FB4', alpha = 0.6)
   # Plotting the second scatter plot
   plt.plot(x, acc_mean, marker='o', linestyle='-',color='#459BB8', label='Balanced Accuracy', linewidth=4, markersize=6)
   plt.fill_between(x, acc_mean - acc_sd, acc_mean + acc_sd, color='#ABCFE3', alpha = 0.6)
   axs.spines['bottom'].set_linewidth(2)
   axs.spines['left'].set_linewidth(2)
   axs.spines['top'].set_linewidth(2)
   axs.spines['right'].set_linewidth(2)
   plt.ylim(0.6, 0.9)
   plt.yticks([0.6,0.65, 0.7,0.75, 0.8,0.85 ,0.9], fontsize = 15)
   plt.xticks(fontsize=15)
   # Adding labels and title
   plt.xlabel('Fraction of training data used',  fontsize=20)
   plt.ylabel('AUROC or Accuracy',  fontsize=20)
   plt.legend(fontsize = 'x-large', loc = 'lower right', title = 'PeptideESM', title_fontsize ='xx-large')
   plt.tight_layout()
   plt.savefig('LassoESM_diff_train_size.png', dpi = 300)




