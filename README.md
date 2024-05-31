# LassoESM

This repository contains all the source code for predicting substrate selectivity of lasso cyclase with protein language model.

## Repository Structure
- Pretraining lasso peptide-specific language model (LassoESM)
  - `LassoESM_pretraining.py`
- Evaulate LassoESM in predicting substrate of lasso cyclase task
  - `get_embeddings_LassoESM`: extract embeddings for peptide variants in training set from LassoESM, then feed in various downstream tasks
  - `hyperparameter_optimization_ML_FusA.py`: grid-search hyperparameters of downstream classification models
  - `downstream_models_performance_diff_3embs.py`: compare the downstream model performance with different embeddings (VanillaESM, PeptideESM, LassoESM)
  - `diff_training_size.py`: evaulate the downstream performance using different training size
  - `cal_uncertainty.py`: explore uncertainty of classification model output
- Model verification and optimization
  - `tSNE-plot.py`: visualize the training data and new experimental verified data sets

## Authors

- **Xuenan Mi** - [xmi4@illinois.edu](mailto:xmi4@illinois.edu)

## License
