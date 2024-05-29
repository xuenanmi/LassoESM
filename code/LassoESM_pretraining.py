import torch
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from datasets import Dataset
import sys
import trl

# Determine the device to use for training (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load sequences from an Excel file
df = pd.read_excel('../data/RODEO_seq_high_score.xlsx')

# Extract the 'Core' column which contains the sequences
Lasso_seq_tmp = df['Core'].tolist()

# Remove duplicates
Lasso_seq = list(set(Lasso_seq_tmp))
print(f"Total unique sequences: {len(Lasso_seq)}")
random.shuffle(Lasso_seq)

# Split the data into training and testing sets (80% train, 20% test)
split_index = int(len(Lasso_seq) * 0.8)
Lasso_seq_train = Lasso_seq[:split_index]
Lasso_seq_test = Lasso_seq[split_index:]

# Load the pre-trained model and tokenizer
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

# Create a data collator for language modeling with a masking probability of 0.15
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Tokenize and create datasets for training and testing
train_dataset = Dataset.from_dict(tokenizer(Lasso_seq_train, truncation=True, padding=True)).shuffle(seed=42)
test_dataset = Dataset.from_dict(tokenizer(Lasso_seq_test, truncation=True, padding=True))

# Define the training arguments
args = TrainingArguments(
    output_dir='./RODEO_high_score_ESM', #output directory
    evaluation_strategy='epoch',  # Evaluate at the end of each epoch
    save_strategy='epoch',        # Save the model at the end of each epoch
    learning_rate=5e-5,
    gradient_checkpointing=True,
    optim="galore_adamw",         # Gradient Low-Rank Projection (GaLore) is a memory-efficient low-rank training strategy that allows full-parameter learning but is more memory-efficient
    optim_target_modules=["encoder", 'contact_head'],
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    push_to_hub=False,
)

# Create a Trainer object for training and evaluation
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Start training the model
trainer.train()

