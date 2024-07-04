# Example
This folder contains an example with Cyclase/Lasso peptide sequence file and the code for extracting embeddings from LassoESM and predicting cyclase-lasso peptide pairs.

- Data
  - `Cyclase_LassoPeptide_pairs_test.csv`: Contains cyclase and lasso peptide sequences for testing
    
- Code
  - `get_embeddings_LassoESM`: Extracts embeddings for lasso peptide variants using the LassoESM model on Hugging Face
  - `predict_cyclase_peptide_pairs_with_CrossAttention`: Predicts cyclase-lasso peptide pairs using a cross-attention layer
