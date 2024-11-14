import torch
import random
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from datasets import Dataset


# Check device availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration for different models
MODELS = {
    "Lasso_ESM": {
        "model_path": "ShuklaGroupIllinois/LassoESM",
        "output_file": "FusA_LassoESM.npy",                 # The embedding matrix is saved in "data" folder
    },
    "VanillaESM": {
        "model_path": "facebook/esm2_t33_650M_UR50D",
        "output_file": "FusA_VanillaESM.npy",
    },
    "PeptideESM": {
        "model_path": "ShuklaGroupIllinois/PeptideESM2_650M",
        "output_file": "FusA_PeptideESM_650M.npy",
    },
}

# Function to extract mean embeddings for a sequence
def get_mean_rep(sequence, model, tokenizer):
    token_ids = tokenizer(sequence, return_tensors='pt').to(device)
    with torch.no_grad():
        results = model(token_ids.input_ids, output_hidden_states=True)
    representations = results.hidden_states[-1][0]  # Use the last hidden layer
    mean_embedding = representations.mean(dim=0)
    return mean_embedding.cpu().numpy()

# Main function to process embeddings
def process_embeddings(data_file, model_name):
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not configured. Available models: {list(MODELS.keys())}")
    
    config = MODELS[model_name]
    print(f"Loading model: {model_name}")
    
    model = AutoModelForMaskedLM.from_pretrained(config["model_path"]).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    model.eval()
    
    # Load data
    data = pd.read_csv(data_file)
    seq_ls = data.iloc[:, 0].tolist()  # Extract sequences from the first column
    print(f"Number of sequences: {len(seq_ls)}")
    
    # Extract embeddings
    seq_embs = [get_mean_rep(seq, model, tokenizer) for seq in seq_ls]
    seq_embs = np.array(seq_embs)
    
    # Save embeddings
    np.save(config["output_file"], seq_embs)
    print(f"Embeddings saved to: {config['output_file']}")
    print(f"Shape of embeddings: {seq_embs.shape}")

# Example usage
if __name__ == "__main__":
    data_file = '../data/FusA_tolerance_dataset.csv'
    for model_name in MODELS.keys():
        process_embeddings(data_file, model_name)
