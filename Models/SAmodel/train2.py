import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tab_transformer_pytorch import TabTransformer

def compute_metrics(predictions, targets):
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    mse = mean_squared_error(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    rmse = root_mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    return mse, mae, rmse, r2

# Configuration
num_unique_equipment_ids = 10  # Update this based on your dataset
num_continuous_features = 3    # cost, duration, quality (excluding MPI)
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

# Load your dataset
df = pd.read_csv('dummy_data.csv')

# Encode 'Equipment_ID' as a categorical feature
label_encoder = LabelEncoder()
df['Equipment_ID_encoded'] = label_encoder.fit_transform(df['Equipment_ID'])

# Normalize continuous features: 'Cost', 'Time', 'Quality'
continuous_features = ['Cost', 'Time', 'Quality']
scaler = StandardScaler()
df[continuous_features] = scaler.fit_transform(df[continuous_features])

# Define the target variable
# Replace 'target_column' with the actual column name you want to predict
target_column = 'Quality_Prediction'  # Update this to your target column name
target = df[target_column].values

# Split the data into training and validation sets
train_cat, val_cat, train_cont, val_cont, train_target, val_target = train_test_split(
    df['Equipment_ID_encoded'].values,
    df[continuous_features].values,
    target,
    test_size=0.2,  # 80% training, 20% validation
    random_state=42
)

# Convert to PyTorch tensors
train_cat_tensor = torch.tensor(train_cat, dtype=torch.long).unsqueeze(1)
val_cat_tensor = torch.tensor(val_cat, dtype=torch.long).unsqueeze(1)
train_cont_tensor = torch.tensor(train_cont, dtype=torch.float32)
val_cont_tensor = torch.tensor(val_cont, dtype=torch.float32)
train_target_tensor = torch.tensor(train_target, dtype=torch.float32).unsqueeze(1)
val_target_tensor = torch.tensor(val_target, dtype=torch.float32).unsqueeze(1)

# Create TensorDatasets
train_dataset = TensorDataset(train_cat_tensor, train_cont_tensor, train_target_tensor)
val_dataset = TensorDataset(val_cat_tensor, val_cont_tensor, val_target_tensor)

# Create DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
model.train()
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for cat_data, cont_data, targets in train_dataloader:
        optimizer.zero_grad()
        outputs = model(cat_data, cont_data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    mse, mae, rmse, r2 = compute_metrics(outputs, targets)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_dataloader):.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')


model.eval()
val_loss = 0.0
all_predictions = []
all_targets = []
with torch.no_grad():
    for cat_data, cont_data, targets in val_dataloader:
        outputs = model(cat_data, cont_data)
        loss = criterion(outputs, targets)
        val_loss += loss.item()
        all_predictions.append(outputs)
        all_targets.append(targets)
# Concatenate all batches
all_predictions = torch.cat(all_predictions)
all_targets = torch.cat(all_targets)
# Compute metrics for validation
mse, mae, rmse, r2 = compute_metrics(all_predictions, all_targets)
print(f'Validation Loss: {val_loss/len(val_dataloader):.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}')

