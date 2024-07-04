# Example
This folder contains an example with Cyclase/Lasso peptide sequence file and the code for get embeddings from LassoESM and the general model for predicting cyclase-lasso peptide pairs.

- Data
  - `Cyclase_LassoPeptide_pairs_test.csv`: Cyclase and Lasso peptide sequence
    
- Code
  - `get_embeddings_LassoESM`: extract embeddings for lasso peptide variants from LassoESM on Hugging Face
  - `predict_cyclase_peptide_pairs_with_CrossAttention`: predict the cyclase-lasso peptide pairs with cross-attention layer
