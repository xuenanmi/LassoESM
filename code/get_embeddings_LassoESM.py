import torch
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from datasets import Dataset
from collections import Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForMaskedLM.from_pretrained("../RODEO_high_score_ESM/checkpoint-3592").to(device)  
tokenizer = AutoTokenizer.from_pretrained("../RODEO_high_score_ESM/checkpoint-3592")


def get_mean_rep(sequence):
    token_ids = tokenizer(sequence, return_tensors='pt').to(device)
    with torch.no_grad():
        results = model(token_ids.input_ids, output_hidden_states=True)
    representations = results.hidden_states[33][0]
    mean_embedding = representations.mean(dim=0)
    return mean_embedding.cpu().numpy()


if __name__ == "__main__":
   model.eval()
   ##extract embeddings
   data = pd.read_excel('../231130_FusA_Mutants_SEBedit.xlsx')
   seq_embs = []
   seq_ls = data.iloc[:,0].tolist()
   print(len(seq_ls))
   for seq in seq_ls:
       seq_embs.append(get_mean_rep(seq))
   seq_embs = np.array(seq_embs)
   print(seq_embs.shape)
   np.save('FusA_embs_from_RODEO_ESM_650M_lr_5e-05_batch_size_8.npy', seq_embs)   
  
