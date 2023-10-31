import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from itertools import product

# Parameters
TIME_PERIOD = 365
INITIAL_BACTERIA = 1e4
BASE_DECAY_RATE = 0.05

def bacterial_growth_rate(initial_bacteria, sugar_intake, brushing_freq):
    growth_rate = sugar_intake * 1e4 - brushing_freq * 1e3
    return initial_bacteria * (1 + growth_rate/1e6)**np.arange(TIME_PERIOD)

def tooth_decay_rate(tooth_type, diet_abrasiveness, dental_checkups, sugar_intake, brushing_freq):
    if tooth_type == "molar":
        decay_factor = 1.0
    else:
        decay_factor = 1.2
    wear_rate = diet_abrasiveness * 0.01
    decay_reduction = dental_checkups * 0.01
    bacteria_over_time = bacterial_growth_rate(INITIAL_BACTERIA, sugar_intake, brushing_freq)
    decay_over_time = decay_factor * BASE_DECAY_RATE * bacteria_over_time / 1e6
    decay_over_time = decay_over_time - decay_reduction - wear_rate
    return np.clip(decay_over_time, 0, 100)

def visualize_3D_decay(molar_data, incisor_data):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    days = np.arange(TIME_PERIOD)
    ax.plot(days, molar_data, zs=0, zdir='y', label='Molar Decay')
    ax.plot(days, incisor_data, zs=1, zdir='y', label='Incisor Decay')
    ax.set_xlabel('Days')
    ax.set_ylabel('Tooth Type (0=Molar, 1=Incisor)')
    ax.set_zlabel('Tooth Decay Rate')
    ax.legend()
    plt.show()

# Synthetic Data Generation
diet_abrasiveness_values = range(1, 11)
dental_checkups_values = range(0, 5)
sugar_intake_values = range(1, 11)
brushing_freq_values = range(1, 8)
dataset = list(product(diet_abrasiveness_values, dental_checkups_values, sugar_intake_values, brushing_freq_values))

features = []
molar_decay_values = []

for data_point in dataset:
    diet_abrasiveness, dental_checkups, sugar_intake, brushing_freq = data_point
    decay = tooth_decay_rate("molar", diet_abrasiveness, dental_checkups, sugar_intake, brushing_freq)
    avg_decay = np.mean(decay)
    features.append([diet_abrasiveness, dental_checkups, sugar_intake, brushing_freq])
    molar_decay_values.append(avg_decay)

features = np.array(features)
molar_decay_values = np.array(molar_decay_values)

# Train a Linear Regression Model
model = LinearRegression().fit(features, molar_decay_values)

# Simulate tooth decay
molar_decay = tooth_decay_rate("molar", 5, 2, 5, 7)
incisor_decay = tooth_decay_rate("incisor", 5, 2, 5, 7)

# 3D Visualization
visualize_3D_decay(molar_decay, incisor_decay)

# Predict using the trained model
sample_feature = np.array([[5, 2, 5, 7]])
predicted_decay_avg = model.predict(sample_feature)
print(f"Predicted average decay for molars: {predicted_decay_avg[0]}")
