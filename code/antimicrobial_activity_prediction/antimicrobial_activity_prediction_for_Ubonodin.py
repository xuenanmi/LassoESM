import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('../date/Deeplasso_full_seq_with_score.csv')    # Ubonodin dataset collected from paper "High-Throughput Screen Reveals the Structure–Activity Relationship of the Antimicrobial Lasso Peptide Ubonodin" ACS Cent. Sci. 2023
Xs = np.load('../data/Deeplasso_embs_from_LassoESM.npy')  # LassoESM embeddings for Ubonodin dataset (remove the sequences with stop codons)
ys = data.iloc[:, 1].values

# Convert data to PyTorch tensors
X_tensor = torch.tensor(Xs, dtype=torch.float32)
y_tensor = torch.tensor(ys, dtype=torch.float32).view(-1, 1)

# Create a dataset and split into training, validation, and test sets
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(Xs.shape[1], 256)
        self.hidden2 = nn.Linear(256, 32)
        self.output = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.hidden1(x))
        x = self.relu(self.hidden2(x))
        x = self.output(x)
        return x

model = MLP()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with early stopping
best_val_loss = float('inf')
patience = 10
trigger_times = 0

for epoch in range(100):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()

    val_loss /= len(val_loader)

    print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        # Save the best model
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        trigger_times += 1

    if trigger_times >= patience:
        print('Early stopping!')
        break

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate the model on the test set
model.eval()
test_preds = []
test_targets = []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        test_preds.extend(outputs.view(-1).cpu().numpy())
        test_targets.extend(y_batch.view(-1).cpu().numpy())

mae = mean_absolute_error(test_targets, test_preds)
pearson_corr = pearsonr(test_targets, test_preds)[0]
spearman_corr = spearmanr(test_targets, test_preds)[0]

print(f'Mean Absolute Error: {mae}')
print(f'Pearson Correlation: {pearson_corr}')
print(f'Spearman Correlation: {spearman_corr}')


# Scatter plot of test predictions vs. actual targets
f, ax = plt.subplots(ncols=1, nrows=1, figsize=(10,7))
plt.scatter(test_preds, test_targets, alpha=0.5, color = '#17becf')
# Fit a regression line
m, b = np.polyfit(test_preds, test_targets, 1)
plt.plot(test_preds, m * np.array(test_preds) + b, linewidth = 2, color= '#d62728')

plt.xlabel('Predicted enrichment value',fontsize =20)
plt.ylabel('Experimental enrichment value', fontsize = 20)
for spine in ['bottom', 'left', 'top', 'right']:
    ax.spines[spine].set_linewidth(2)
plt.xlim(-8,8)
plt.ylim(-8,8)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.savefig("Deeplasso_enrichment_score.png", dpi =300) 
