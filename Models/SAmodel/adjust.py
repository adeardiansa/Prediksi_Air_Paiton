import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def adjust_dummy_data(dummy_file, real_file, output_file):
    """
    Adjusts an existing dummy data Excel file to match the pattern of real data.
    If the dummy file contains an 'Equipment ID' column (instead of 'Equipment'),
    it creates an 'Equipment' column by mapping each unique ID to real equipment names.
    It also adjusts cost values based on the distribution from the real data.

    Parameters:
      dummy_file: str - Path to the dummy data Excel file.
      real_file: str - Path to a representative real data Excel file (e.g., summary outage budget, R1.xlsx).
      output_file: str - Path to save the adjusted dummy data.
    """
    # Load the dummy data and keep a copy for visualization (before adjustment)
    dummy_df = pd.read_excel(dummy_file)
    dummy_before = dummy_df.copy()  # For visualization

    # Check for equipment column: if not present, try using 'Equipment ID'
    if 'Equipment' not in dummy_df.columns:
        if 'Equipment_ID' in dummy_df.columns:
            # Use Equipment ID to create a temporary Equipment column for mapping
            dummy_df['Equipment'] = dummy_df['Equipment_ID']
        else:
            raise ValueError("Neither 'Equipment' nor 'Equipment ID' column found in dummy data.")

    # Load the real data
    real_df = pd.read_excel(real_file)
    # Ensure real data has the required columns
    for col in ['Equipment', 'Cost']:
        if col not in real_df.columns:
            raise ValueError(f"Column '{col}' not found in {real_file}.")

    # Extract cost distribution parameters from real data
    cost_mean = real_df['Cost'].mean()
    cost_std = real_df['Cost'].std()

    # Get real equipment names from real data
    real_equipments = real_df['Equipment'].unique()

    # Create a mapping from dummy equipment values (IDs or names) to real equipment names.
    dummy_equipment_values = dummy_df['Equipment'].unique()
    equipment_mapping = {dummy_val: real_equipments[i % len(real_equipments)]
                         for i, dummy_val in enumerate(dummy_equipment_values)}

    # Replace the dummy equipment values with the real equipment names
    dummy_df['Equipment'] = dummy_df['Equipment'].map(equipment_mapping)

    # Adjust cost values in dummy data by sampling from a normal distribution
    dummy_df['Cost'] = np.random.normal(cost_mean, cost_std, size=len(dummy_df))
    dummy_df['Cost'] = dummy_df['Cost'].apply(lambda x: max(x, 0))

    # Save the adjusted dummy data to a new Excel file
    dummy_df.to_excel(output_file, index=False)
    print(f"Adjusted dummy data saved to {output_file}")

    # -------------------------------
    # Visualization: Before vs. After
    # -------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Cost Distribution Histograms
    axes[0, 0].hist(dummy_before['Cost'], bins=20, color='blue', alpha=0.7)
    axes[0, 0].set_title("Cost Distribution - Before Adjustment")
    axes[0, 0].set_xlabel("Cost")
    axes[0, 0].set_ylabel("Frequency")

    axes[0, 1].hist(dummy_df['Cost'], bins=20, color='green', alpha=0.7)
    axes[0, 1].set_title("Cost Distribution - After Adjustment")
    axes[0, 1].set_xlabel("Cost")
    axes[0, 1].set_ylabel("Frequency")

    # Equipment Frequency Bar Plots
    # For the 'before' plot, use 'Equipment' if it exists; otherwise, fall back on 'Equipment ID'
    if 'Equipment' in dummy_before.columns:
        equipment_counts_before = dummy_before['Equipment'].value_counts()
        label_before = "Equipment"
    elif 'Equipment_ID' in dummy_before.columns:
        equipment_counts_before = dummy_before['Equipment_ID'].value_counts()
        label_before = "Equipment_ID"
    else:
        equipment_counts_before = pd.Series()
        label_before = "Equipment"

    axes[1, 0].bar(equipment_counts_before.index.astype(str),
                   equipment_counts_before.values, color='blue', alpha=0.7)
    axes[1, 0].set_title(f"{label_before} Frequency - Before Adjustment")
    axes[1, 0].set_xlabel(label_before)
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].tick_params(axis='x', rotation=45)

    # After adjustment, plot Equipment frequency (which now uses real equipment names)
    equipment_counts_after = dummy_df['Equipment'].value_counts()
    axes[1, 1].bar(equipment_counts_after.index.astype(str),
                   equipment_counts_after.values, color='green', alpha=0.7)
    axes[1, 1].set_title("Equipment Frequency - After Adjustment")
    axes[1, 1].set_xlabel("Equipment")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()


def generate_dummy_data_from_real(real_file, num_records, output_file):
    """
    Generates new dummy data based solely on the pattern of the real data.
    Uses the real data's equipment names and cost distribution to create new records.

    Parameters:
      real_file: str - Path to a representative real data Excel file.
      num_records: int - Number of dummy records to generate.
      output_file: str - Path to save the generated dummy data.
    """
    # Load the real data
    real_df = pd.read_excel(real_file)
    for col in ['Equipment', 'Cost']:
        if col not in real_df.columns:
            raise ValueError(f"Column '{col}' not found in {real_file}.")

    # Extract equipment list and cost distribution parameters from real data
    equipments = real_df['Equipment'].unique()
    cost_mean = real_df['Cost'].mean()
    cost_std = real_df['Cost'].std()

    # Generate new dummy data with random selections from real equipment and sampled cost values
    dummy_data = {
        'Equipment': np.random.choice(equipments, size=num_records),
        'Cost': np.random.normal(cost_mean, cost_std, size=num_records)
    }
    dummy_df = pd.DataFrame(dummy_data)
    dummy_df['Cost'] = dummy_df['Cost'].apply(lambda x: max(x, 0))

    # Save the generated dummy data to an Excel file
    dummy_df.to_excel(output_file, index=False)
    print(f"Generated dummy data saved to {output_file}")

    # -------------------------------
    # Visualization: Real vs. Generated Dummy Data
    # -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(real_df['Cost'], bins=20, color='blue', alpha=0.7)
    axes[0].set_title("Real Data Cost Distribution")
    axes[0].set_xlabel("Cost")
    axes[0].set_ylabel("Frequency")

    axes[1].hist(dummy_df['Cost'], bins=20, color='green', alpha=0.7)
    axes[1].set_title("Generated Dummy Data Cost Distribution")
    axes[1].set_xlabel("Cost")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # ---------------------------------------------
    # Option 1: Adjust your existing dummy data
    # ---------------------------------------------
    adjust_dummy_data(
        dummy_file='dummy_data.xlsx',
        real_file='summary outage budget, R1.xlsx',
        output_file='adjusted_dummy_data.xlsx'
    )

    # ---------------------------------------------
    # Option 2: Generate new dummy data based solely on real data
    # Uncomment the block below to use this alternative approach.
    # ---------------------------------------------
    # generate_dummy_data_from_real(
    #     real_file='summary outage budget, R1.xlsx',
    #     num_records=100,  # Adjust the number of records as needed
    #     output_file='generated_dummy_data.xlsx'
    # )
