import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import sys

def plot_tSNE(Xs, ys, title):
    pca = PCA(n_components = 100)
    xt = pca.fit_transform(Xs)
    tsne = TSNE(n_components=2,perplexity=20, random_state=12)
    X_tsne = tsne.fit_transform(xt)
    print('Done')
    classes = np.unique(ys)
    f, ax = plt.subplots(ncols=1, nrows=1,figsize=(10,7))
    colors = ['#88C1E080', '#83B663', '#EAB24A', '#D13878']
    labels = ['Initial training set', 'Test set1', 'Test set2', 'Test set 3']
    for i, c in enumerate(classes):
        plt.scatter(X_tsne[ys == c, 0], X_tsne[ys == c, 1], c=colors[i],label = labels[i])

    plt.xlabel("t-SNE1", fontsize=20)
    plt.ylabel("t-SNE2", fontsize=20)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(title='', loc='upper right')
    plt.savefig(title, dpi = 300)
    plt.close()

if __name__ == "__main__":
   Xs_ori = np.load('../data/FusA_embs_from_RODEO_ESM_650M_lr_5e-05_batch_size_8.npy')
   data = pd.read_excel('../data/231130_FusA_Mutants_SEBedit.xlsx')
   ys_ori = Xs_ori.shape[0] * [0]

   Xs_r2 = np.load("../data/new_adversarial_round_exp_verified_embs_from_RODEO_ESM_650M_lr_5e-05_batch_size_8.npy")
   r2 =  pd.read_csv('../data/new_adversarial.csv')
   ys_r2 = Xs_r2.shape[0] * [1]

   Xs_r1 = np.load('../data/new_first_round_exp_verified_embs_from_RODEO_ESM_650M_lr_5e-05_batch_size_8.npy')
   ys_r1 = Xs_r1.shape[0] * [2]

   Xs_r3 = np.load("../data/fourth_round_exp_verified_embs_from_RODEO_ESM_650M_lr_5e-05_batch_size_8.npy")
   ys_r3 = Xs_r3.shape[0] * [3]


   Xs = np.vstack((Xs_ori, Xs_r2, Xs_r1, Xs_r3))
   ys = ys_ori + ys_r2 +  ys_r1 + ys_r3
   plot_tSNE(Xs, ys, 'ori_r1_r2_r3_tSNE.png')
   

   
   

  


