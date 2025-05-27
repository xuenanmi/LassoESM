# LassoESM: A tailored language model to enhance lasso peptide property prediction

![LassoESM](LassoESM_workflow.png) 
## Quick Start
### Getting started with this repo
```bash
git clone https://github.com/xuenanmi/LassoESM.git
cd LassoESM
conda env create -f environment.yml
conda activate lassoesm
```
### Usage
Extract embeddings from LassoESM

Below is a minimal example for extracting LassoESM embeddings from a peptide sequence list:
```bash
from transformers import AutoTokenizer, AutoModel
import torch

# Load LassoESM model
tokenizer = AutoTokenizer.from_pretrained("ShuklaGroupIllinois/LassoESM")
model = AutoModel.from_pretrained("ShuklaGroupIllinois/LassoESM")
model.eval()

sequences = ["WYTAEWGLELIFVFPRFI", "GGAGHVPEYFVGIGTPISFYG"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for seq in sequences:
    # Tokenize with special tokens
    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][0]  # shape: [seq_len, 1280]

    # Convert input_ids to tokens to check positions
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Exclude special tokens [CLS], [EOS] by slicing [1:-1]
    residue_embeddings = last_hidden[1:-1]  # shape: [L, 1280], L = length of sequence

    # Per-sequence embedding: mean over residues
    mean_embedding = residue_embeddings.mean(dim=0)  # shape: [1280]

    print(f"Per-residue embedding shape: {residue_embeddings.shape}")
    print(f"Per-sequence embedding shape: {mean_embedding.shape}")
```

## Repository Structure
- LassoESM
  - `LassoESM_pretraining.py`: Pretraining Lasso Peptide-Specific Language Model
  - `get_embeddings_LassoESM`: extract embeddings for peptide variants in training set from LassoESM/PeptideESM/VanillaESM, then feed them into various downstream classification models
    
- Downstream task 1: substrate_tolerance_prediction  
  - `hyperparameter_optimization_ML_FusA.py`: grid-search for hyperparameters of downstream classification models
  - `downstream_models_performance_diff_3embs.py`: compare downstream model performance with different embeddings (VanillaESM, PeptideESM, LassoESM)
  - `diff_training_size.py`: evaulate downstream model performance using different training size
  - `cal_uncertainty.py`: explore uncertainty of classification model output

- Downstream task 2: cycalse_peptide_pair_prediction
  - `generate_negative_cyclase_peptide_pairs.py`: generate the synthetic cyclase-peptide pairs (negative samples)
  - `predict_cyclase_peptide_pairs`: a general model (MLP) to predict cyclase(embedded by VanillaESM)-peptide(embedded by LassoESM) pairs
  - `predict_cyclase_peptide_pairs_with_CrossAttention`: add a cross-attention layer, where lasso peptide embeddings reweight its corresponding cyclase embeddings
  - `predict_non_natural_cyclase_peptide_pairs.py`: use the trained cyclase-peptide pair prediction model to assess the compatibility of FusC with other predicted naturally occuring lasso peptides

- Downstream task 3: antimicrobial_activity_prediction
  - `get_embeddings_for_Ubonodin.py`: extracting embeddings for Ubonodin variant sequences from LassoESM/PeptideESM/VanillaESM models
  - `antimicrobial_activity_prediction_for_Ubonodin.py`: predict the antimicrobial activity of Ubonodin variant sequences
  - `get_embeddings_for_Klebsidin.py`: extracting embeddings for Klebsidin variant sequences from LassoESM/PeptideESM/VanillaESM models
  - `antimicrobial_activity_prediction_for_Klebsidin.py`: predict the antimicrobial activity of Klebsidin variant sequences

## Dependency
To set up the environment for this project, use the provided `environment.yml` file. This file contains all necessary dependencies.

## Authors

- **Xuenan Mi** - [xmi4@illinois.edu](mailto:xmi4@illinois.edu)



