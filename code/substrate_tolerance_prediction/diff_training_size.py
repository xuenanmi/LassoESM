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
    """
    Evaluate model performance using cross-validation with balanced accuracy and ROC AUC.
    
    Args:
        X_test (numpy array): Test feature data.
        y_test (list): Test labels.
    
    Returns:
        list: Mean balanced accuracy and ROC AUC scores.
    """
    cv = RepeatedKFold(n_splits=10, n_repeats=3)
    SVM = SVC(C = 10, kernel = 'linear')
    SVC_CV_acc = cross_val_score(SVM, X_test, y_test, cv = cv, n_jobs = -1, scoring = 'balanced_accuracy')
    SVC_CV_auc = cross_val_score(SVM, X_test, y_test, cv = cv, n_jobs = -1, scoring = 'roc_auc')
    return [np.mean(SVC_CV_acc), np.mean(SVC_CV_auc)]
    
if __name__ == "__main__":
   # Load data
   Xs = np.load('../data/FusA_LassoESM.npy')
   data = pd.read_csv('../data/FusA_tolerance_dataset.csv')
   ys = data.iloc[:,1].tolist()

   diff_train_size_acc, diff_train_size_auc = [], []
   for m in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7,0.8, 0.9]:
       train_size_acc, train_size_auc = [], []
       for i in range(10):
           X_train, X_test, y_train, y_test = train_test_split(Xs, ys, stratify=ys, test_size= m)
           print(Counter(y_test))
           cv_acc, cv_auc = model_performance(X_test, y_test)
           train_size_acc.append(cv_acc)
           train_size_auc.append(cv_auc)

       print(f"Fraction of Training size:{m}")
       diff_train_size_acc.append(train_size_acc)
       diff_train_size_auc.append(train_size_auc)
   
   # Prepare data for plotting
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
   plt.plot(x, auc_mean, marker='^', linestyle='-', color='#66A182', label='AUROC', linewidth=4, markersize=7)
   plt.fill_between(x, auc_mean - auc_sd, auc_mean + auc_sd, color='#A8D5BA', alpha=0.6)

   # Plotting the second scatter plot
   plt.plot(x, acc_mean, marker='o', linestyle='--', color='#66A182', label='Balanced Accuracy', linewidth=4, markersize=7)
   plt.fill_between(x, acc_mean - acc_sd, acc_mean + acc_sd, color='#A8D5BA', alpha=0.6)

   for spine in ['bottom', 'left', 'top', 'right']:
      axs.spines[spine].set_linewidth(2)
   plt.ylim(0.65, 0.9)
   plt.yticks([0.65, 0.7,0.75, 0.8,0.85 ,0.9], fontsize = 20)
   plt.xticks(fontsize=20)
   # Adding labels and legend
   plt.xlabel('Fraction of training data used',  fontsize=20)
   plt.ylabel('AUROC or Accuracy',  fontsize=20)
   plt.legend(fontsize = 'x-large', loc = 'lower right')
   plt.tight_layout()
   plt.savefig('LassoESM_diff_train_size_new.png', dpi = 300)

