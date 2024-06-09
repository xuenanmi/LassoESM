import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, balanced_accuracy_score

# Define the CrossAttention module
class CrossAttention(nn.Module):
    def __init__(self):
        super(CrossAttention, self).__init__()
        self.W_query = nn.Parameter(torch.rand(1280, 1280))  # Weight matrix for queries
        self.W_key = nn.Parameter(torch.rand(1280, 1280))    # Weight matrix for keys
        self.W_value = nn.Parameter(torch.rand(1280, 1280))  # Weight matrix for values

    def forward(self, x_1, x_2, attn_mask=None):
        # Compute queries, keys, and values
        query = torch.matmul(x_2, self.W_query)
        key = torch.matmul(x_1, self.W_key)
        value = torch.matmul(x_1, self.W_value)

        # Compute attention scores
        attn_scores = torch.matmul(query, key.transpose(-2, -1))
        scaled_attn_scores = attn_scores / math.sqrt(query.size(-1))
        attn_weights = F.softmax(scaled_attn_scores, dim=-1)

        # Apply attention weights to values
        output = torch.matmul(attn_weights, value)
        return output, attn_weights

# Modify the MLP model to include CrossAttention
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
    
    def forward(self, x):
        # Split input into two parts
        x_1, x_2 = x[:, :1280], x[:, 1280:]
        # Apply cross-attention mechanism
        x_1, _ = self.cross_attention(x_1, x_2)
        # Concatenate outputs and pass through MLP layers
        x = torch.cat((x_1, x_2), dim=1)
        return self.mlp(x)

# Custom dataset class for handling the input features and labels
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=25):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device) 
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        
        # Evaluate the model on validation data
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device) 
                outputs = model(inputs)
                loss = criterion(outputs, labels.unsqueeze(1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Function to evaluate the model
def evaluate_model(model, dataloader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device) 
            outputs = model(inputs)
            preds = (outputs > 0.5).float()
            all_preds.extend(preds.squeeze().tolist())
            all_labels.extend(labels.tolist())
    
    return balanced_accuracy_score(all_labels, all_preds), recall_score(all_labels, all_preds)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    Cyclase = np.load('../data/All_cyclase_RODEO_from_VanillaESM.npy') ## Lasso cyclase were embedded by VanillaESM
    substrate = np.load('../data/lasso_RODEO_embs_from_RODEO_ESM_650M_lr_5e-05_batch_size_8.npy') ## Lasso peptides were embedded by LassoESM
    
    # Combine Cyclase and substrate data
    Cyclase_substrate = [np.concatenate((Cyclase[i, :], substrate[i, :])) for i in range(Cyclase.shape[0])]
    Xs = np.array(Cyclase_substrate)

    # Load labels
    data = pd.read_csv('../data/Cyclase_substrate_pairs_pos_neg.csv')
    ys = data.iloc[:, 2].tolist()
    
    # Split data into train, validation, and test sets
    Xs_train, Xs_temp, ys_train, ys_temp = train_test_split(Xs, ys, test_size=0.3, stratify=ys, random_state=42)
    Xs_val, Xs_test, ys_val, ys_test = train_test_split(Xs_temp, ys_temp, test_size=0.5, stratify=ys_temp, random_state=42)
    
    # Create dataset and dataloaders
    train_dataset = CustomDataset(Xs_train, ys_train)
    val_dataset = CustomDataset(Xs_val, ys_val)
    test_dataset = CustomDataset(Xs_test, ys_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    input_size = Xs_train.shape[1]
    model = MLPWithAttention(input_size).to(device)  # lasso peptide embeddings reweight its corresponding cyclase embeddings
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, val_loader, criterion, optimizer, epochs=10)
    
    # Evaluate the model
    balanced_accuracy, recall = evaluate_model(model, test_loader)
    print("Balanced Accuracy:", balanced_accuracy)
    print("Recall (True Positive Rate):", recall)

