import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tab_transformer_pytorch import TabTransformer  # original TabTransformer model

# Load your dataset
df = pd.read_csv('dummy_data.csv')

# Encode 'Equipment_ID' as a categorical feature
label_encoder = LabelEncoder()
df['Equipment_ID_encoded'] = label_encoder.fit_transform(df['Equipment_ID'])

# Define continuous features
continuous_features = ['Cost', 'Time', 'Quality']

# Create a mask for missing values in continuous features (1 for missing, 0 for present)
mask = df[continuous_features].isna().astype(float).values

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

# Basic configuration parameters
num_unique_equipment_ids = len(label_encoder.classes_)  # Number of unique Equipment_IDs
num_continuous_features = len(continuous_features)      # Number of continuous features
embedding_dim = 32
num_transformer_layers = 6
num_attention_heads = 8

#############################################
# Define alternative models for prediction  #
#############################################

# 1. CNN + Dense Layers Model
class CNN_Dense_Model(nn.Module):
    def __init__(self, num_unique_equipment_ids, num_continuous, embedding_dim, 
                 cnn_out_channels=16, cnn_kernel_size=2, hidden_dim=64):
        super(CNN_Dense_Model, self).__init__()
        self.embedding = nn.Embedding(num_unique_equipment_ids, embedding_dim)
        # For continuous branch: treat input as (batch, 1, num_continuous)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=cnn_kernel_size)
        conv_output_size = num_continuous - cnn_kernel_size + 1  # output length after conv
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim + cnn_out_channels * conv_output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x_cat, x_cont):
        cat_embed = self.embedding(x_cat).squeeze(1)  # shape: (batch, embedding_dim)
        # Process continuous features with CNN: (batch, num_continuous) -> (batch, 1, num_continuous)
        cont_input = x_cont.unsqueeze(1)
        conv_out = self.conv1(cont_input)  # shape: (batch, cnn_out_channels, conv_output_size)
        conv_out = conv_out.view(conv_out.size(0), -1)  # flatten
        # Concatenate categorical and CNN outputs
        x = torch.cat([cat_embed, conv_out], dim=1)
        return self.fc(x)

# 2. Neural Network with Dense Layers Only (Fully-Connected)
class DenseOnly_Model(nn.Module):
    def __init__(self, num_unique_equipment_ids, num_continuous, embedding_dim, hidden_dim=64):
        super(DenseOnly_Model, self).__init__()
        self.embedding = nn.Embedding(num_unique_equipment_ids, embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim + num_continuous, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x_cat, x_cont):
        cat_embed = self.embedding(x_cat).squeeze(1)
        x = torch.cat([cat_embed, x_cont], dim=1)
        return self.fc(x)

# 3. Huber Regression Model (Benchmark)
# This is a simple linear regression model with Huber (SmoothL1) loss.
class HuberRegression_Model(nn.Module):
    def __init__(self, num_unique_equipment_ids, num_continuous, embedding_dim):
        super(HuberRegression_Model, self).__init__()
        self.embedding = nn.Embedding(num_unique_equipment_ids, embedding_dim)
        self.fc = nn.Linear(embedding_dim + num_continuous, 1)
        
    def forward(self, x_cat, x_cont):
        cat_embed = self.embedding(x_cat).squeeze(1)
        x = torch.cat([cat_embed, x_cont], dim=1)
        return self.fc(x)

#############################################
# Select which model to use via model_choice #
#############################################

# Options: 'tabformer', 'cnn_dense', 'dense', 'huber'
model_choice = 'cnn_dense'  # Change this value to try a different model

if model_choice == 'tabformer':
    # Use the original TabTransformer model
    model = TabTransformer(
        categories=(num_unique_equipment_ids,),  # Tuple with number of unique Equipment_IDs
        num_continuous=num_continuous_features,
        dim=embedding_dim,
        depth=num_transformer_layers,
        heads=num_attention_heads,
        attn_dropout=0.1,
        ff_dropout=0.1
    )
    criterion = nn.MSELoss()
    
elif model_choice == 'cnn_dense':
    model = CNN_Dense_Model(num_unique_equipment_ids, num_continuous_features, embedding_dim,
                            cnn_out_channels=16, cnn_kernel_size=2, hidden_dim=64)
    criterion = nn.MSELoss()
    
elif model_choice == 'dense':
    model = DenseOnly_Model(num_unique_equipment_ids, num_continuous_features, embedding_dim, hidden_dim=64)
    criterion = nn.MSELoss()
    
elif model_choice == 'huber':
    model = HuberRegression_Model(num_unique_equipment_ids, num_continuous_features, embedding_dim)
    criterion = nn.SmoothL1Loss()  # Huber loss
else:
    raise ValueError("Unknown model_choice. Please select one of: 'tabformer', 'cnn_dense', 'dense', 'huber'.")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#############################################
# Define evaluation metrics and plotting    #
#############################################
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

#############################################
# Training and validation loop              #
#############################################
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
        # Apply mask to continuous data (zero out missing features)
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
            cont_data = cont_data * (1 - masks)
            outputs = model(cat_data, cont_data)
            loss = criterion(outputs, targets)
            val_running_loss += loss.item()
            all_predictions.append(outputs)
            all_targets.append(targets)
    avg_val_loss = val_running_loss / len(val_dataloader)
    val_losses.append(avg_val_loss)

    # Concatenate predictions and targets for metric computation
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
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
print(model_choice)