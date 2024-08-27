# LassoESM: A tailored language model to enhance lasso peptide property prediction

![LassoESM](image_workflow.png)
This repository contains all the source code for predicting substrate selectivity of lasso cyclase using protein language model.

## Repository Structure
- LassoESM
  - `LassoESM_pretraining.py`: Pretraining Lasso Peptide-Specific Language Model
  - `get_embeddings_LassoESM`: extract embeddings for peptide variants in training set from LassoESM, then feed them into various downstream classification models
    
- Downstream task1: cyclase_specificity_prediction  
  - `hyperparameter_optimization_ML_FusA.py`: grid-search for hyperparameters of downstream classification models
  - `downstream_models_performance_diff_3embs.py`: compare downstream model performance with different embeddings (VanillaESM, PeptideESM, LassoESM)
  - `diff_training_size.py`: evaulate downstream model performance using different training size
  - `cal_uncertainty.py`: explore uncertainty of classification model output
  - `tSNE-plot.py`: visualize the training data and new experimental verified data sets

- Downstream task 2: cycalse_peptide_pair_prediction
  - `generate_neg_pairs.py`: generate the synthetic cyclase-peptide pairs (negative samples)
  - `predict_cyclase_peptide_pairs`: a general model (MLP) to predict cyclase(embedded by VanillaESM)-peptide(embedded by LassoESM) pairs
  - `predict_cyclase_peptide_pairs_with_CrossAttention`: add a crUbonodin_antimicrobial_activity_predictionoss-attention layer, where lasso peptide embeddings reweight its corresponding cyclase embeddings

- Downstream task 3: antimicrobial_activity_prediction
  - `antimicrobial_activity_prediction_Ubonodin.py`: predict the antimicrobial activity of lasso peptide using LassoESM embeddings

## Dependency
To set up the environment for this project, use the provided `environment.yml` file. This file contains all necessary dependencies.

## Authors

- **Xuenan Mi** - [xmi4@illinois.edu](mailto:xmi4@illinois.edu)



