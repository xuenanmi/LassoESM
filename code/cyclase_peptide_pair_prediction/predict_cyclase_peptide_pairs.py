import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, balanced_accuracy_score, roc_auc_score, precision_score

# Custom dataset class for handling the input features and labels
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Multi-Layer Perceptron model definition
class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs=25):
    min_val_loss = np.inf
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
                
        if min_val_loss > val_loss:
            print(f'Val Loss Decreased({min_val_loss:.4f} to {val_loss:.4f}) Saving The Model')
            min_val_loss = val_loss
            torch.save(model.state_dict(), 'saved_best_MLP_model.pth')

# Function to evaluate the model
def evaluate_model(model, dataloader):
    model.load_state_dict(torch.load("saved_best_MLP_model.pth"))
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
    
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    
    return balanced_accuracy, recall, auc, precision

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    Cyclase = np.load('Cyclase_RODEO_embs_from_VanillaESM.npy')   # Embeddings of Cyclase (from VanillaESM), shape of matrix is [The number of cyclase, 1280]
    substrate = np.load('Lasso_peptide_RODEO_embs_from_LassoESM.npy') # Embeddings of lasso peptides (from LassoESM), shape of matrix is [The number of lasso peptides, 1280] 
    
    # Combine Cyclase and substrate data
    Cyclase_substrate = [np.concatenate((Cyclase[i, :], substrate[i, :])) for i in range(Cyclase.shape[0])]
    Xs = np.array(Cyclase_substrate)

    # Load labels
    data = pd.read_csv('Cyclase_substrate_pairs_pos_neg.csv')
    ys = data.iloc[:, 2].tolist()
    
    # Split data into train, validation, and test sets
    Xs_train, Xs_temp, ys_train, ys_temp = train_test_split(Xs, ys, test_size=0.3, stratify=ys, random_state=42)
    Xs_val, Xs_test, ys_val, ys_test = train_test_split(Xs_temp, ys_temp, test_size=0.5, stratify=ys_temp, random_state=42)
    
    # Create dataset and dataloader
    train_dataset = CustomDataset(Xs_train, ys_train)
    val_dataset = CustomDataset(Xs_val, ys_val)
    test_dataset = CustomDataset(Xs_test, ys_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model, loss function, and optimizer
    input_size = Xs_train.shape[1]
    model = MLP(input_size).to(device)
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


