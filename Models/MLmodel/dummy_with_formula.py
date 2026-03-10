import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# Parameters for dummy data generation
np.random.seed(42)  # For reproducibility
n_equipment = 10  # Number of equipment
time = np.linspace(0, 100, 100)  # Time points from 0 to 100
mean_time_constant = 20  # Mean value for time constants
std_time_constant = 5    # Standard deviation for time constants

# Function to simulate underdamped response
def underdamped_function(t, tau):
    return 1 - np.exp(-t/tau) * (np.cos(2 * np.pi * t / tau) + np.sin(2 * np.pi * t / tau))

def overdamped_function(t, c=None, tau=None, tau1=None, tau2=None, omega_n=1.0, A1=1, A2=0.5):
    """
    Simulate an overdamped response. The function can handle:
    
    1. Overdamped response with damping coefficient 'c'
    2. Overdamped response with time constants 'tau1' and 'tau2'
    3. Overdamped response with minimal input (single 'tau')
    
    Parameters:
    t (array): Time array
    c (float, optional): Damping coefficient (must be > 1 for overdamped response). If provided, it will be used.
    tau (float, optional): Single time constant. If provided, the simplified response will be used.
    tau1 (float, optional): First time constant. If provided (along with tau2), these will be used.
    tau2 (float, optional): Second time constant.
    omega_n (float, optional): Natural frequency (used when 'c' is provided).
    A1 (float, optional): Amplitude scaling factor for the first term.
    A2 (float, optional): Amplitude scaling factor for the second term.
    
    Returns:
    array: Quality values over time following an overdamped response.
    
    Raises:
    ValueError: If neither `c`, `tau`, nor `tau1, tau2` is provided.
    """
    # Case 1: Overdamped response using damping coefficient 'c'
    if c is not None:
        if c <= 1:
            raise ValueError("Damping coefficient must be greater than 1 for overdamped response")
        # Calculate the two exponents based on c
        alpha1 = omega_n * (c + np.sqrt(c**2 - 1))
        alpha2 = omega_n * (c - np.sqrt(c**2 - 1))
        return 1 - A1 * np.exp(-alpha1 * t) - A2 * np.exp(-alpha2 * t)
    
    # Case 2: Overdamped response using two time constants tau1 and tau2
    elif tau1 is not None and tau2 is not None:
        return 1 - A1 * np.exp(-t / tau1) - A2 * np.exp(-t / tau2)
    
    # Case 3: Overdamped response using single time constant 'tau' (minimal input)
    elif tau is not None:
        tau1 = tau
        tau2 = tau / 2  # For simplicity, second decay rate is half the first one
        return 1 - np.exp(-t / tau1) - 0.5 * np.exp(-t / tau2)
    
    else:
        raise ValueError("You must provide either a damping coefficient 'c', a single time constant 'tau', or time constants 'tau1' and 'tau2'.")

def cost_function(t):
    # Randomize a and b between 1 and 10
    a = np.random.uniform(1, 10)
    b = np.random.uniform(1, 10)
    
    # Randomize c between 1000 and 50000
    c = np.random.uniform(1000, 50000)
    
    # Calculate and return the cost value
    return a * t**2 + b * t + c

# Define the loss function: minimize cost and maximize quality
def loss_function(t, huber_model_c, huber_model_q, alpha=1, beta=1):
    # Ensure t is 2D array for predict
    t = np.array([[t]])
    
    # Predict cost using the cost model
    c_pred = huber_model_c.predict(t)[0]
    
    # Predict quality using the quality model
    q_pred = huber_model_q.predict(t)[0]
    
    # Calculate the loss (combination of cost and quality)
    return alpha * c_pred - beta * q_pred

# Function to save the dummy data to a CSV file with headers
def save_data_to_csv(dataframe, filename="dummy_data.csv"):
    """
    Save the given DataFrame to a CSV file with headers.

    Parameters:
    - dataframe (pd.DataFrame): The data to save.
    - filename (str): The filename for the CSV file.
    """
    dataframe.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

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

    estimator = HuberRegressor()
    huber_model_q = make_pipeline(PolynomialFeatures(2), estimator)
    huber_model_q.fit(time.reshape(-1, 1), quality)
    q_pred = huber_model_q.predict(time.reshape(-1, 1))

    huber_model_c = make_pipeline(PolynomialFeatures(2), estimator)
    huber_model_c.fit(time.reshape(-1, 1), cost)
    c_pred = huber_model_c.predict(time.reshape(-1, 1))

    # Get regression formula coefficients for quality and cost
    q_coefficients = huber_model_q.named_steps['huberregressor'].coef_
    q_intercept = huber_model_q.named_steps['huberregressor'].intercept_

    c_coefficients = huber_model_c.named_steps['huberregressor'].coef_
    c_intercept = huber_model_c.named_steps['huberregressor'].intercept_

    # Construct formulas as text
    q_formula = f"Quality: {q_coefficients[2]:.2f}t² + {q_coefficients[1]:.2f}t + {q_intercept:.2f}"
    c_formula = f"Cost: {c_coefficients[2]:.2f}t² + {c_coefficients[1]:.2f}t + {c_intercept:.2f}"

    # Store the data and formulas for each equipment
    for t, q, qq, c, cc in zip(time, quality, q_pred, cost, c_pred):
        data.append([equipment_id, t, q, tau, qq, c, cc, q_formula, c_formula])

# Convert data to a DataFrame for analysis
df = pd.DataFrame(data, columns=["Equipment_ID", "Time", "Quality", "Time_Constant", "Quality_Prediction", "Cost", "Cost_Prediction", "Q_Formula", "C_Formula"])

# Save the generated data to CSV
save_data_to_csv(df)

# Create subplots for Quality Prediction
fig, axes = plt.subplots(2, 5, figsize=(20, 12), sharex=True)
axes = axes.flatten()

for i, equipment_id in enumerate(df['Equipment_ID'].unique()):
    time_data = df[df['Equipment_ID'] == equipment_id]['Time'].values
    actual_quality = df[df['Equipment_ID'] == equipment_id]['Quality'].values
    predicted_quality = df[df['Equipment_ID'] == equipment_id]['Quality_Prediction'].values
    q_formula = df[df['Equipment_ID'] == equipment_id]['Q_Formula'].values[0]

    axes[i].scatter(time_data, actual_quality, label=f'Actual Quality (Equipment {equipment_id})', alpha=0.6)
    axes[i].plot(time_data, predicted_quality, label='Predicted Quality', color='red')
    axes[i].set_title(f'Equipment {equipment_id}\n{q_formula}', fontsize=10)
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
    c_formula = df[df['Equipment_ID'] == equipment_id]['C_Formula'].values[0]

    axes[i].scatter(time_data, actual_cost, label=f'Actual Cost (Equipment {equipment_id})', alpha=0.6)
    axes[i].plot(time_data, predicted_cost, label='Predicted Cost', color='red')
    axes[i].set_title(f'Equipment {equipment_id}\n{c_formula}', fontsize=10)
    axes[i].set_xlabel('Time')
    axes[i].set_ylabel('Cost')
    axes[i].legend()
    axes[i].grid(True)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
