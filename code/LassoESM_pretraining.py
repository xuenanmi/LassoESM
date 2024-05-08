import torch
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling
from datasets import Dataset
import sys
import trl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load sequences from csv
df = pd.read_excel('../RODEO_seq_high_score.xlsx')
Lasso_seq_tmp = df['Core'].tolist()
Lasso_seq = list(set(Lasso_seq_tmp))
print(len(Lasso_seq))
random.shuffle(Lasso_seq)
Lasso_seq_train = Lasso_seq[:int(len(Lasso_seq) * 0.8)]
Lasso_seq_test = Lasso_seq[int(len(Lasso_seq) * 0.8):]

# Define model, tokenizer, and MLM data set object
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
#print(model)
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
train = Dataset.from_dict(tokenizer(Lasso_seq_train)).shuffle(seed=42)
test = Dataset.from_dict(tokenizer(Lasso_seq_test))


args = TrainingArguments(
    output_dir='./RODEO_high_score_ESM',
    evaluation_strategy='epoch',
    save_strategy='epoch',
    learning_rate=5e-5,
    gradient_checkpointing = True,
    optim="galore_adamw",
    optim_target_modules=["encoder",'contact_head'],
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    push_to_hub=False,
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train,
    eval_dataset=test,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
