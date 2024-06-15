import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, balanced_accuracy_score, roc_auc_score, precision_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForMaskedLM

# Define the CrossAttention module
class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()
        self.W_query = nn.Parameter(torch.rand(1280, 1280))  # Weight matrix for queries
        self.W_key = nn.Parameter(torch.rand(1280, 1280))    # Weight matrix for keys
        self.W_value = nn.Parameter(torch.rand(1280, 1280))  # Weight matrix for values

    def forward(self, x_1, x_2, attn_mask=None):
        # Compute queries, keys, and values
        """
        query: Tensor of shape [batch_size, len_peptide, esm_dim]
        value: Tensor of shape [batch_size, len_cyclase, esm_dim]
        """
        query = torch.matmul(x_1, self.W_query)
        key = torch.matmul(x_2, self.W_key)
        value = torch.matmul(x_2, self.W_value)

        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        scaled_attn_scores = attn_scores / math.sqrt(query.size(-1))
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, float('-inf'))  # mask the padding residues
        
        attn_weights = F.softmax(scaled_attn_scores, dim=-1)
        
        # Apply attention weights to values
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights

# Define the MLP model with CrossAttention
class MLPWithAttention(nn.Module):
    def __init__(self, input_size):
        super(MLPWithAttention, self).__init__()
        self.cross_attention = CrossAttention()
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, cyclase, substrate, cyclase_mask, substrate_mask):
        # Apply cross-attention mechanism
        attn_mask = torch.matmul(substrate_mask.unsqueeze(-1).float(), cyclase_mask.unsqueeze(1).float())
        x_1, _ = self.cross_attention(substrate, cyclase, attn_mask)   # reweighted cyclase embeddings

        # Average embeddings along the sequence length dimension
        x_1_avg = torch.mean(x_1, dim=1)
        substrate_avg = torch.mean(substrate, dim=1)
        
        # Concatenate averaged embeddings and pass through MLP layers
        x = torch.cat((x_1_avg, substrate_avg), dim=1)
        return self.mlp(x)

# Function to get representation from Vanilla ESM model
def get_rep_from_VanillaESM(sequence):
    token_ids = esm_tokenizer(sequence, return_tensors='pt').to(device)
    with torch.no_grad():
        results = esm_model(token_ids.input_ids, output_hidden_states=True)
    representations = results.hidden_states[33][0]
    return representations.cpu().numpy()

# Function to get representation from LassoESM model
def get_rep_from_LassoESM(sequence):
    token_ids = LassoESM_tokenizer(sequence, return_tensors='pt').to(device)
    with torch.no_grad():
        results = LassoESM_model(token_ids.input_ids, output_hidden_states=True)
    representations = results.hidden_states[33][0]
    return representations.cpu().numpy()

# Function to pad the ESM embeddings
def pad_esm_embedding(embedding, max_length):
    embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
    pad_length = max_length - embedding.shape[0]
    padding = torch.zeros((pad_length, embedding.shape[1]), dtype=torch.float32)
    embedding_tensor = torch.cat((embedding_tensor, padding), dim=0)
    
    # Create attention mask
    attn_mask = torch.ones(max_length, dtype=torch.float32)
    attn_mask[embedding.shape[0]:] = 0
    attn_mask[0] = 0  # BOS token
    attn_mask[embedding.shape[0] - 1] = 0  # EOS token
    
    return embedding_tensor, attn_mask

# Define the custom dataset class
class CustomDataset(Dataset):
    def __init__(self, cyclase_sequences, substrate_sequences, labels, max_cyclase_length, max_substrate_length):
        self.cyclase_sequences = cyclase_sequences
        self.substrate_sequences = substrate_sequences
        self.labels = labels
        self.max_cyclase_length = max_cyclase_length
        self.max_substrate_length = max_substrate_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        cyclase_seq = self.cyclase_sequences[idx]
        substrate_seq = self.substrate_sequences[idx]
        
        # Pad cyclase sequence and create mask
        cyclase_embedding, cyclase_mask = pad_esm_embedding(get_rep_from_VanillaESM(cyclase_seq), self.max_cyclase_length)
        
        # Pad substrate sequence and create mask
        substrate_embedding, substrate_mask = pad_esm_embedding(get_rep_from_LassoESM(substrate_seq), self.max_substrate_length)
        
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return cyclase_embedding, substrate_embedding, cyclase_mask, substrate_mask, label

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=25):
    min_val_loss = np.inf
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for cyclase, substrate, cyclase_mask, substrate_mask, labels in train_loader:
            cyclase, substrate, cyclase_mask, substrate_mask, labels = cyclase.to(device), substrate.to(device), cyclase_mask.to(device), substrate_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(cyclase, substrate, cyclase_mask, substrate_mask)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        train_loss = running_loss / len(train_loader)
        
        # Evaluate the model on validation data
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for cyclase, substrate, cyclase_mask, substrate_mask, labels in val_loader:
                cyclase, substrate, cyclase_mask, substrate_mask, labels = cyclase.to(device), substrate.to(device), cyclase_mask.to(device), substrate_mask.to(device), labels.to(device)
                outputs = model(cyclase, substrate, cyclase_mask, substrate_mask)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if min_val_loss > val_loss:
            print(f'Val Loss Decreased({min_val_loss:.4f} to {val_loss:.4f}) Saving The Model')
            min_val_loss = val_loss
            torch.save(model.state_dict(), 'saved_best_model.pth')

# Function to evaluate the model
def evaluate_model(model, dataloader):
    model.load_state_dict(torch.load("saved_best_model.pth"))
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for cyclase, substrate, cyclase_mask, substrate_mask, labels in dataloader:
            cyclase, substrate, cyclase_mask, substrate_mask, labels = cyclase.to(device), substrate.to(device), cyclase_mask.to(device), substrate_mask.to(device), labels.to(device)
            outputs = model(cyclase, substrate, cyclase_mask, substrate_mask)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.squeeze().tolist())
            all_labels.extend(labels.tolist())
    
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    
    return balanced_accuracy, recall, auc, precision

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    esm_model = AutoModelForMaskedLM.from_pretrained("facebook/esm2_t33_650M_UR50D").to(device)
    esm_tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
    esm_model.eval()
    LassoESM_model = AutoModelForMaskedLM.from_pretrained("../../fourth_round/RODEO_high_score_ESM/checkpoint-3592").to(device)
    LassoESM_tokenizer = AutoTokenizer.from_pretrained("../../fourth_round/RODEO_high_score_ESM/checkpoint-3592")
    LassoESM_model.eval()

    # Load labels
    data = pd.read_csv('Cyclase_substrate_pairs_pos_neg.csv')
    Cyclase_seq = data.iloc[:, 0].tolist()
    substrate_seq = data.iloc[:, 1].tolist()
    labels = data.iloc[:, 2].tolist()

    # Calculate max lengths for padding
    max_cyclase_length = max(len(seq) for seq in Cyclase_seq) + 2
    max_substrate_length = max(len(seq) for seq in substrate_seq) + 2

    # Split data into train, validation, and test sets
    Cyclase_train, Cyclase_temp, Substrate_train, Substrate_temp, ys_train, ys_temp = train_test_split(Cyclase_seq, substrate_seq, labels, test_size=0.3, stratify=labels, random_state=42)
    Cyclase_val, Cyclase_test, Substrate_val, Substrate_test, ys_val, ys_test = train_test_split(Cyclase_temp, Substrate_temp, ys_temp, test_size=0.5, stratify=ys_temp, random_state=42)

    # Create dataset and dataloaders
    train_dataset = CustomDataset(Cyclase_train, Substrate_train, ys_train, max_cyclase_length, max_substrate_length)
    val_dataset = CustomDataset(Cyclase_val, Substrate_val, ys_val, max_cyclase_length, max_substrate_length)
    test_dataset = CustomDataset(Cyclase_test, Substrate_test, ys_test, max_cyclase_length, max_substrate_length)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize model, loss function, and optimizer
    input_size = 1280 * 2
    model = MLPWithAttention(input_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=25)
    
    # Evaluate the model
    balanced_accuracy, recall, auc, precision = evaluate_model(model, test_loader)
    print("Balanced Accuracy:", balanced_accuracy)
    print("Recall (True Positive Rate):", recall)
    print("AUC:", auc)
    print("Precision:", precision)

