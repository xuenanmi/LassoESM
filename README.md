# LassoESM: A tailored language model to enhance lasso peptide property prediction

![LassoESM](LassoESM_workflow.png) 

## Table of Contents

- [Quick Start](#quick-start)
  - [Getting started with this repo](#getting-started-with-this-repo)
  - [Usage](#usage)
- [Repository Structure](#-repository-structure)
  - [`code`](#code)
  - [`example_notebook`](#examplenotebook)
  - [`data`](#data)
- [Authors](#authors)

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
    inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1][0]  # shape: [seq_len, 1280]

    # Extract per-residue representations
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

    # Exclude special tokens [CLS], [EOS] by slicing [1:-1]
    residue_embeddings = last_hidden[1:-1]  # shape: [L, 1280], L = length of sequence

    # Per-sequence embedding: mean over residues
    mean_embedding = residue_embeddings.mean(dim=0)  # shape: [1280]

    print(f"Per-residue embedding shape: {residue_embeddings.shape}")
    print(f"Per-sequence embedding shape: {mean_embedding.shape}")
```

## Repository Structure

### `code` 
Source code for model training and downstream tasks:

- **LassoESM pretraining**:
  - `LassoESM_pretraining.py`: Pretraining Lasso Peptide-Specific Language Model
  - `get_embeddings_LassoESM`: extract embeddings for peptide variants in dataset from LassoESM/PeptideESM/VanillaESM, then feed them into various downstream classification models
    
- **Downstream Task 1: Substrate Tolerance Prediction** 
  - `hyperparameter_optimization_ML_FusA.py`: grid-search for hyperparameters of downstream classification models
  - `downstream_models_performance_diff_4embs.py`: compare downstream model performance with different embeddings (VanillaESM, PeptideESM, LassoESM)
  - `diff_training_size.py`: evaulate downstream model performance using different training size
  - `cal_uncertainty.py`: explore uncertainty of classification model output

- **Downstream Task 2: Cyclase-Peptide Pair Prediction**
  - `generate_negative_cyclase_peptide_pairs.py`: generate the synthetic cyclase-peptide pairs (negative samples)
  - `predict_cyclase_peptide_pairs`: a general model (MLP) to predict cyclase(embedded by VanillaESM)-peptide(embedded by LassoESM) pairs
  - `predict_cyclase_peptide_pairs_with_CrossAttention`: add a cross-attention layer, where lasso peptide embeddings reweight its corresponding cyclase embeddings
  - `predict_non_natural_cyclase_peptide_pairs.py`: use the trained cyclase-peptide pair prediction model to assess the compatibility of FusC with other predicted naturally occuring lasso peptides

- **Downstream Task 3: Antimicrobial Activity Prediction**
  - `get_embeddings_for_Ubonodin.py`: extracting embeddings for Ubonodin variant sequences from LassoESM/PeptideESM/VanillaESM models
  - `antimicrobial_activity_prediction_for_Ubonodin.py`: predict the antimicrobial activity of Ubonodin variant sequences
  - `get_embeddings_for_Klebsidin.py`: extracting embeddings for Klebsidin variant sequences from LassoESM/PeptideESM/VanillaESM models
  - `antimicrobial_activity_prediction_for_Klebsidin.py`: predict the antimicrobial activity of Klebsidin variant sequences

---

### `example_notebook`  
Example Jupyter notebooks for key analysis:

- `extract_embeddings.ipynb`: Demonstrates embedding extraction workflow.
- `predict_substrate_tolerance.ipynb`: Substrate tolerance classification pipeline.
- `predict_cyclase_peptide_pairs.ipynb`: Predict cyclase-peptide compatibility.
- `predict_antimicrobial_activity.ipynb`: Evaluate peptide antimicrobial activity.

---

### `data`  
Dataset used in the paper

- `data_for_LassoESM_training`
- `data_for_substrate_tolerance_prediction`
- `data_for_cyclase_peptide_pair_prediction`
- `data_for_antimicrobial_activity_prediction`

---

## Authors

- **Xuenan Mi** - [xmi4@illinois.edu](mailto:xmi4@illinois.edu)



