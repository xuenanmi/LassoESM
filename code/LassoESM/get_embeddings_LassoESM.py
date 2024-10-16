import torch
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from datasets import Dataset
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Load pre-trained model and tokenizer from the checkpoint
model = AutoModelForMaskedLM.from_pretrained("ShuklaGroupIllinois/LassoESM").to(device)  #the checkpoint is saved in huggingface
tokenizer = AutoTokenizer.from_pretrained("ShuklaGroupIllinois/LassoESM")


def get_mean_rep(sequence):
    """
    Get the mean representation (embedding) of a given sequence.
    
    Args:
        sequence (str): Input sequence to encode.
    
    Returns:
        np.ndarray: Mean embedding of the input sequence.
    """
    # Tokenize the sequence
    token_ids = tokenizer(sequence, return_tensors='pt').to(device)
    with torch.no_grad():
        results = model(token_ids.input_ids, output_hidden_states=True)
    # Extract the hidden states of the last layer
    representations = results.hidden_states[33][0]
    # Compute the mean embedding
    mean_embedding = representations.mean(dim=0)
    return mean_embedding.cpu().numpy()


if __name__ == "__main__":
   # Set the model to evaluation mode
   model.eval()
   # Extract embeddings for sequences initial dataset
   data = pd.read_excel('../data/231130_FusA_Mutants_SEBedit.xlsx')
   seq_embs = []
   seq_ls = data.iloc[:,0].tolist()
   print(len(seq_ls))
   # Process each sequence and get its embedding
   for seq in seq_ls:
       seq_embs.append(get_mean_rep(seq))
   seq_embs = np.array(seq_embs)
   print(seq_embs.shape)
   np.save('../data/FusA_embs_from_RODEO_ESM_650M_lr_5e-05_batch_size_8.npy', seq_embs) 
