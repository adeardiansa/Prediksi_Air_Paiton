import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tab_transformer_pytorch import TabTransformer

# Example configuration
num_categories = [10]  # Number of unique PO types
num_continuous = 4     # cost, duration, quality, MPI
dim = 32               # Embedding dimension
depth = 6              # Number of transformer layers
heads = 8              # Number of attention heads

# Define the TabTransformer model
model = TabTransformer(
    categories=num_categories,
    num_continuous=num_continuous,
    dim=dim,
    depth=depth,
    heads=heads,
    attn_dropout=0.1,
    ff_dropout=0.1
)

# Load your dataset
df = pd.read_csv('dummy_data.csv')

# Encode categorical features
le = LabelEncoder()
df['PO_type_encoded'] = le.fit_transform(df['Equipment_ID'])

# Normalize continuous features
scaler = StandardScaler()
continuous_features = ['Cost', 'Time', 'Quality', 'Quality_Prediction']
df[continuous_features] = scaler.fit_transform(df[continuous_features])

# Prepare input tensors
categorical_data = torch.tensor(df['PO_type_encoded'].values, dtype=torch.long).unsqueeze(1)
continuous_data = torch.tensor(df[continuous_features].values, dtype=torch.float32)

# Assuming 'target' is the column to predict
target = torch.tensor(df['target'].values, dtype=torch.float32).unsqueeze(1)

# Create DataLoader
dataset = TensorDataset(categorical_data, continuous_data, target)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for cat_data, cont_data, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(cat_data, cont_data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader):.4f}')

# Switch to evaluation mode
model.eval()
with torch.no_grad():
    # Assuming you have a validation DataLoader
    val_loss = 0.0
    for cat_data, cont_data, targets in val_dataloader:
        outputs = model(cat_data, cont_data)
        loss = criterion(outputs, targets)
        val_loss += loss.item()
    print(f'Validation Loss: {val_loss/len(val_dataloader):.4f}')
