import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Parameters for dummy data generation
np.random.seed(42)  # For reproducibility
torch.manual_seed(42)  # For reproducibility
n_equipment = 10  # Number of equipment
time = np.linspace(0, 100, 100)  # Time points from 0 to 100
mean_time_constant = 20  # Mean value for time constants
std_time_constant = 5    # Standard deviation for time constants

# Function to simulate underdamped response
def overdamped_function(t, tau=None, tau1=None, tau2=None, A1=1, A2=0.5):
    if tau is not None:
        tau1 = tau
        tau2 = tau / 2  # For simplicity, second decay rate is half the first one
        return 1 - np.exp(-t / tau1) - 0.5 * np.exp(-t / tau2)
    else:
        raise ValueError("You must provide a time constant 'tau'.")

def cost_function(t):
    a = np.random.uniform(1, 10)
    b = np.random.uniform(1, 10)
    c = np.random.uniform(1000, 50000)
    return a * t**2 + b * t + c

def save_data_to_csv(dataframe, filename="dummy_data.csv"):
    dataframe.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# PyTorch model definition
class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Generate data for each equipment
data = []

for equipment_id in range(1, n_equipment + 1):
    tau = np.random.normal(mean_time_constant, std_time_constant)
    base_quality = overdamped_function(time, tau)
    quality_variance = np.random.normal(0, 0.05, size=time.shape)
    quality = np.clip(base_quality + quality_variance, 0, 1)

    cost = cost_function(time)
    cost_variance = np.random.normal(0, 1000, size=time.shape)
    cost += cost_variance

    # Prepare data for PyTorch
    time_tensor = torch.tensor(time, dtype=torch.float32).view(-1, 1)
    quality_tensor = torch.tensor(quality, dtype=torch.float32).view(-1, 1)
    cost_tensor = torch.tensor(cost, dtype=torch.float32).view(-1, 1)

    # Define model, loss, and optimizer
    input_dim = 1
    hidden_dim = 64
    output_dim = 1
    model_q = RegressionModel(input_dim, hidden_dim, output_dim)
    model_c = RegressionModel(input_dim, hidden_dim, output_dim)
    
    criterion = nn.MSELoss()
    optimizer_q = optim.Adam(model_q.parameters(), lr=0.01)
    optimizer_c = optim.Adam(model_c.parameters(), lr=0.01)

    # Train the model for quality prediction
    epochs = 1000
    for epoch in range(epochs):
        model_q.train()
        optimizer_q.zero_grad()
        outputs = model_q(time_tensor)
        loss = criterion(outputs, quality_tensor)
        loss.backward()
        optimizer_q.step()

    # Predict quality
    model_q.eval()
    with torch.no_grad():
        q_pred = model_q(time_tensor).numpy()

    # Train the model for cost prediction
    for epoch in range(epochs):
        model_c.train()
        optimizer_c.zero_grad()
        outputs = model_c(time_tensor)
        loss = criterion(outputs, cost_tensor)
        loss.backward()
        optimizer_c.step()

    # Predict cost
    model_c.eval()
    with torch.no_grad():
        c_pred = model_c(time_tensor).numpy()

    # Store the data and formulas for each equipment
    for t, q, qq, c, cc in zip(time, quality, q_pred, cost, c_pred):
        data.append([equipment_id, t, q, tau, qq[0], c, cc[0]])

# Convert data to a DataFrame for analysis
df = pd.DataFrame(data, columns=["Equipment_ID", "Time", "Quality", "Time_Constant", "Quality_Prediction", "Cost", "Cost_Prediction"])

# Save the generated data to CSV
save_data_to_csv(df)

# Create subplots for Quality Prediction
fig, axes = plt.subplots(2, 5, figsize=(20, 12), sharex=True)
axes = axes.flatten()

for i, equipment_id in enumerate(df['Equipment_ID'].unique()):
    time_data = df[df['Equipment_ID'] == equipment_id]['Time'].values
    actual_quality = df[df['Equipment_ID'] == equipment_id]['Quality'].values
    predicted_quality = df[df['Equipment_ID'] == equipment_id]['Quality_Prediction'].values

    axes[i].scatter(time_data, actual_quality, label=f'Actual Quality (Equipment {equipment_id})', alpha=0.6)
    axes[i].plot(time_data, predicted_quality, label='Predicted Quality', color='red')
    axes[i].set_title(f'Equipment {equipment_id}', fontsize=10)
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Quality')
    axes[i].legend()
    axes[i].grid(True)

# Hide any unused subplots if there are less than 10 equipments
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Create subplots for Cost Prediction
fig, axes = plt.subplots(2, 5, figsize=(20, 12), sharex=True)
axes = axes.flatten()

for i, equipment_id in enumerate(df['Equipment_ID'].unique()):
    time_data = df[df['Equipment_ID'] == equipment_id]['Time'].values
    actual_cost = df[df['Equipment_ID'] == equipment_id]['Cost'].values
    predicted_cost = df[df['Equipment_ID'] == equipment_id]['Cost_Prediction'].values

    axes[i].scatter(time_data, actual_cost, label=f'Actual Cost (Equipment {equipment_id})', alpha=0.6)
    axes[i].plot(time_data, predicted_cost, label='Predicted Cost', color='red')
    axes[i].set_title(f'Equipment {equipment_id}', fontsize=10)
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Cost')
    axes[i].legend()
    axes[i].grid(True)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()

# Convert tensors to numpy arrays if they aren't already
q_pred_np = q_pred if isinstance(q_pred, np.ndarray) else q_pred.numpy()
c_pred_np = c_pred if isinstance(c_pred, np.ndarray) else c_pred.numpy()
quality_np = quality if isinstance(quality, np.ndarray) else quality.numpy()
cost_np = cost if isinstance(cost, np.ndarray) else cost.numpy()

# Compute metrics for quality prediction
mse_quality = mean_squared_error(quality_np, q_pred_np)
mae_quality = mean_absolute_error(quality_np, q_pred_np)

# Compute metrics for cost prediction
mse_cost = mean_squared_error(cost_np, c_pred_np)
mae_cost = mean_absolute_error(cost_np, c_pred_np)

# Print the results
print(f'Quality Prediction - MSE: {mse_quality:.4f}, MAE: {mae_quality:.4f}')
print(f'Cost Prediction - MSE: {mse_cost:.4f}, MAE: {mae_cost:.4f}')
