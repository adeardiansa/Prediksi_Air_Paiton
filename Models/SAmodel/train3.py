import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tab_transformer_pytorch import TabTransformer

# Load your dataset
df = pd.read_csv('dummy_data.csv')

# Encode 'Equipment_ID' as a categorical feature
label_encoder = LabelEncoder()
df['Equipment_ID_encoded'] = label_encoder.fit_transform(df['Equipment_ID'])

# Define continuous features
continuous_features = ['Cost', 'Time', 'Quality']

# Create a mask for missing values in continuous features
mask = df[continuous_features].isna().astype(float).values  # 1 for missing, 0 for present

# Replace NaNs with zeros in the DataFrame
df[continuous_features] = df[continuous_features].fillna(0)

# Normalize continuous features
scaler = StandardScaler()
df[continuous_features] = scaler.fit_transform(df[continuous_features])

# Define the target variable
target_column = 'Quality_Prediction'  # Update this to your target column name
target = df[target_column].values

train_cat, val_cat, train_cont, val_cont, train_mask, val_mask, train_target, val_target = train_test_split(
    df['Equipment_ID_encoded'].values,
    df[continuous_features].values,
    mask,
    target,
    test_size=0.2,  # 80% training, 20% validation
    random_state=42
)

# Convert to PyTorch tensors
train_cat_tensor = torch.tensor(train_cat, dtype=torch.long).unsqueeze(1)
val_cat_tensor = torch.tensor(val_cat, dtype=torch.long).unsqueeze(1)
train_cont_tensor = torch.tensor(train_cont, dtype=torch.float32)
val_cont_tensor = torch.tensor(val_cont, dtype=torch.float32)
train_mask_tensor = torch.tensor(train_mask, dtype=torch.float32)
val_mask_tensor = torch.tensor(val_mask, dtype=torch.float32)
train_target_tensor = torch.tensor(train_target, dtype=torch.float32).unsqueeze(1)
val_target_tensor = torch.tensor(val_target, dtype=torch.float32).unsqueeze(1)

# Create TensorDatasets including the mask
train_dataset = TensorDataset(train_cat_tensor, train_cont_tensor, train_mask_tensor, train_target_tensor)
val_dataset = TensorDataset(val_cat_tensor, val_cont_tensor, val_mask_tensor, val_target_tensor)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Configuration
num_unique_equipment_ids = len(label_encoder.classes_)  # Number of unique Equipment_IDs
num_continuous_features = len(continuous_features)      # Number of continuous features
embedding_dim = 32
num_transformer_layers = 6
num_attention_heads = 8

# Define the TabTransformer model
model = TabTransformer(
    categories=(num_unique_equipment_ids,),  # Tuple with number of unique Equipment_IDs
    num_continuous=num_continuous_features,
    dim=embedding_dim,
    depth=num_transformer_layers,
    heads=num_attention_heads,
    attn_dropout=0.1,
    ff_dropout=0.1
)

class TabTransformerWithMask(nn.Module):
    def __init__(self, categories, num_continuous, dim, depth, heads, attn_dropout, ff_dropout):
        super(TabTransformerWithMask, self).__init__()
        self.tab_transformer = TabTransformer(
            categories=categories,
            num_continuous=num_continuous,
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout
        )
        self.output_layer = nn.Linear(dim, 1)  # Assuming a regression task

    def forward(self, x_categ, x_cont, mask):
        # Apply the mask to continuous features
        x_cont = x_cont * (1 - mask)  # Zero out missing features
        # Pass through the TabTransformer
        x = self.tab_transformer(x_categ, x_cont)
        # Output layer
        return self.output_layer(x)

# Define the TabTransformer model
model = TabTransformer(
    categories=(num_unique_equipment_ids,),  # Tuple with number of unique Equipment_IDs
    num_continuous=num_continuous_features,
    dim=embedding_dim,
    depth=num_transformer_layers,
    heads=num_attention_heads,
    attn_dropout=0.1,
    ff_dropout=0.1
)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Function to compute evaluation metrics
def compute_metrics(predictions, targets):
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    return mse, mae, rmse, r2

def plot_metrics(train_losses, val_losses, val_mse, val_mae):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))

    # Plot Losses
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot MSE
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_mse, label='Validation MSE', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('Validation Mean Squared Error')
    plt.legend()

    # Plot MAE
    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_mae, label='Validation MAE', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.title('Validation Mean Absolute Error')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Training and validation loop
num_epochs = 50
train_losses = []
val_losses = []
val_mse = []
val_mae = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    for cat_data, cont_data, masks, targets in train_dataloader:
        optimizer.zero_grad()
        # Apply mask to continuous data
        cont_data = cont_data * (1 - masks)
        outputs = model(cat_data, cont_data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_train_loss = running_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    # Validation phase
    model.eval()
    val_running_loss = 0.0
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        for cat_data, cont_data, masks, targets in val_dataloader:
            # Apply mask to continuous data
            cont_data = cont_data * (1 - masks)
            outputs = model(cat_data, cont_data)
            loss = criterion(outputs, targets)
            val_running_loss += loss.item()
            all_predictions.append(outputs)
            all_targets.append(targets)
    avg_val_loss = val_running_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)

    # Compute metrics
    mse, mae, rmse, r2 = compute_metrics(all_predictions, all_targets)

    val_mse.append(mse)
    val_mae.append(mae)

    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {avg_train_loss:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}, '
          f'MSE: {mse:.4f}, '
          f'MAE: {mae:.4f}, '
          f'RMSE: {rmse:.4f}, '
          f'R²: {r2:.4f}')
    
# Plot the metrics after training
plot_metrics(train_losses, val_losses, val_mse, val_mae)