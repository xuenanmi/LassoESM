from peft import get_peft_config, PeftModel, PeftConfig, inject_adapter_in_model, LoraConfig
from transformers import AutoModelForMaskedLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import pandas as pd
import random
from datasets import Dataset
import torch
import pandas
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# Load sequences from an Excel file
df = pd.read_excel('RODEO_seq_high_score.xlsx')

# Extract the 'Core' column which contains the sequences
Lasso_seq_tmp = df['Core'].tolist()

# Remove duplicates
Lasso_seq = list(set(Lasso_seq_tmp))
print(f"Total unique sequences: {len(Lasso_seq)}")
random.shuffle(Lasso_seq)
# Calculate the maximum sequence length in Lasso_seq
max_seq_length = max(len(seq) for seq in Lasso_seq)
print(f"Maximum sequence length: {max_seq_length}")
#max_seq_length = 512

# Split the data into training and testing sets (80% train, 20% test)
split_index = int(len(Lasso_seq) * 0.8)
Lasso_seq_train = Lasso_seq[:split_index]
Lasso_seq_test = Lasso_seq[split_index:]


# Define LoRA configuration (Parameter-efficient fine-tuning method)
lora_config = LoraConfig(
    r=4,               # Rank of the LoRA matrix
    lora_alpha=1,     # Scaling factor for the LoRA weights
    bias = "all",
    target_modules=["query","key","value","dense"], 
)

# Load the pre-trained model and tokenizer
model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")

model = inject_adapter_in_model(lora_config, model)

trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())

print(f"Trainable parameters: {trainable_params}")
print(f"Total parameters: {total_params}")
print(f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%")

# Create a data collator for language modeling with a masking probability of 0.15
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

# Tokenize and create datasets for training and testing
train_dataset = Dataset.from_dict(tokenizer(Lasso_seq_train, truncation=True, padding='max_length', max_length=max_seq_length)).shuffle(seed=42)
test_dataset = Dataset.from_dict(tokenizer(Lasso_seq_test, truncation=True, padding='max_length', max_length=max_seq_length))


# Define the training arguments
args = TrainingArguments(
    output_dir='./RODEO_high_score_ESM_LoRA',  # output directory for LoRA
    evaluation_strategy='epoch',  # Evaluate at the end of each epoch
    save_strategy='epoch',        # Save the model at the end of each epoch
    learning_rate=5e-5,
    optim="adamw_torch",          
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=2,
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



# Extract the logs from the Trainer's state
training_logs = trainer.state.log_history

# Initialize lists to store loss values
train_loss = []
eval_loss = []
epochs = []

# Iterate over the logs to extract loss values
for log in training_logs:
    if "loss" in log:  # Training loss
        train_loss.append(log["loss"])
        epochs.append(log["epoch"])
    if "eval_loss" in log:  # Validation loss
        eval_loss.append(log["eval_loss"])

print(train_loss)
print(eval_loss)
print(epochs)

# Save the losses to a CSV file
loss_df = pd.DataFrame({
    'Epoch': epochs[:len(train_loss)],
    'Training Loss': train_loss,
    'Validation Loss': eval_loss + [None] * (len(train_loss) - len(eval_loss))  # Fill with None if fewer validation losses
})

loss_df.to_csv('training_validation_losses.csv', index=False)

# Plot the losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs[:len(eval_loss)], eval_loss, label='Validation Loss')  # Validation loss is typically less frequent

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('LassoESM_Pretraining_LoRA_Train_Val_loss.png', dpi = 300)




