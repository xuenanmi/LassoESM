import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import  train_test_split

def percentage_true_prediction(test_fraction):
    """
    Calculate the percentage of true predictions based on predicted probabilities.
    
    Args:
        test_fraction (float): Fraction of the data to use as the test set.
    
    Returns:
        Lists of true prediction percentages and false prediction percentages for each probability bin.
    """
    
    Xs = np.load('../data/exp_data_embeddings/FusA_embs_from_RODEO_ESM_650M_lr_5e-05_batch_size_8.npy')
    data = pd.read_excel('../data/231130_FusA_Mutants_SEBedit.xlsx')
    ys = data.iloc[:,1].tolist()
    # Split data into training and test sets
    Xs_train, Xs_test, ys_train, ys_test = train_test_split(Xs, ys, stratify=ys, test_size= test_fraction)
    print(len(ys_test))
    opt_SVC = SVC(C = 10, kernel = 'linear', probability=True).fit(Xs_train, ys_train)
    ys_pred_prob = opt_SVC.predict_proba(Xs_test)
    ys_pred = opt_SVC.predict(Xs_test)
    # Round predicted probabilities to the nearest tenth
    ys_pred_prob_ls = np.round(ys_pred_prob[:,1], 1)
    ys_pred = ys_pred.tolist()

    # Calculate true and false prediction percentages for each probability bin
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
   true_res, false_res = [], []
   # Perform the analysis multiple times for getting error bar
   for k in range(10):   
       true_pred_res, false_pred_res = percentage_true_prediction(0.2)
       true_res.append(true_pred_res)
       false_res.append(false_pred_res)

   print(true_res)
   print(false_res)  
   
   true_mean = np.mean(true_res, axis = 0)
   true_sd = np.std(true_res, axis = 0)
   
   f, axs = plt.subplots(ncols=1, nrows=1, figsize=(10,7))
   # Make scatter plot
   x = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
   # Plot the average accuracy across different predicted probability
   plt.plot(x, true_mean, marker='o', linestyle='-', color='#66A182',  linewidth=4, markersize=6) 
   # Fill the area between (mean - sd) and (mean + sd) for model accuracy
   plt.fill_between(x, true_mean - true_sd, true_mean + true_sd, color='#A8D5BA', alpha = 0.6)      
    
   for spine in ['bottom', 'left', 'top', 'right']:
      axs.spines[spine].set_linewidth(2)  
   plt.xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],fontsize=20)
   plt.yticks(fontsize=20)
   plt.xlim(-0.05, 1.05)
   plt.xlabel('Predicted probability',  fontsize=20)
   plt.ylabel('Accuracy',  fontsize=20)
   plt.tight_layout()
   plt.savefig('LassoESM_uncertainty_new_plot.png', dpi = 300)

