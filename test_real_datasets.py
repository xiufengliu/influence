"""
Test script to verify that our real datasets can be loaded and preprocessed correctly.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.preprocessing.data_loader import DataLoader

# Test energy_data dataset
print("Testing energy_data dataset...")
energy_loader = DataLoader(dataset_name="energy_data")
X_energy, y_energy, t_energy, c_energy = energy_loader.load_data(preprocess=True)

print(f"X_energy shape: {X_energy.shape}")
print(f"y_energy shape: {y_energy.shape}")
print(f"t_energy shape: {t_energy.shape}")
print(f"c_energy shape: {c_energy.shape}")
print()

# Test steel_industry dataset
print("Testing steel_industry dataset...")
steel_loader = DataLoader(dataset_name="steel_industry")
X_steel, y_steel, t_steel, c_steel = steel_loader.load_data(preprocess=True)

print(f"X_steel shape: {X_steel.shape}")
print(f"y_steel shape: {y_steel.shape}")
print(f"t_steel shape: {t_steel.shape}")
print(f"c_steel shape: {c_steel.shape}")
print()

# Create a simple visualization of the target variables
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(y_energy[:100])
plt.title('Energy Data - First 100 Appliances Values')
plt.xlabel('Index')
plt.ylabel('Appliances (Wh)')

plt.subplot(1, 2, 2)
plt.plot(y_steel[:100])
plt.title('Steel Industry - First 100 Usage Values')
plt.xlabel('Index')
plt.ylabel('Usage (kWh)')

plt.tight_layout()
plt.savefig('data_visualization.png')
print("Visualization saved to data_visualization.png")

print("Tests completed successfully!")
