# Tooth Decay Simulation and Analysis

## Introduction

This Python application simulates and analyzes tooth decay over time based on various factors like diet, dental care habits, and tooth type. It uses machine learning (Linear Regression) to predict the average decay of teeth given certain conditions. The application provides both numerical analysis and 3D visualizations.

## Features

- **Tooth Decay Simulation:** Simulates tooth decay for molars and incisors over a period.
- **Variable Analysis:** Considers factors like sugar intake, brushing frequency, diet abrasiveness, and dental checkups.
- **3D Visualization:** Visualizes tooth decay rates in 3D for comparative analysis between molars and incisors.
- **Machine Learning Model:** Utilizes Linear Regression to predict average tooth decay based on the given factors.
- **Synthetic Data Generation:** Generates a dataset to train the machine learning model.

## Installation

To run this script, the following Python libraries are required:
- `numpy`: For numerical operations and data handling.
- `matplotlib`: For 3D visualization of the tooth decay data.
- `sklearn`: For implementing and training the Linear Regression model.

These can be installed using pip:
```bash
pip install numpy matplotlib scikit-learn
```

## Usage

1. **Run the Script:** Execute the script to start the simulation.
2. **Data Generation:** The script automatically generates a dataset based on predefined parameters.
3. **Machine Learning Model Training:** A Linear Regression model is trained using the generated dataset.
4. **Tooth Decay Simulation:** The script simulates tooth decay for molars and incisors based on different factors.
5. **Visualization:** View a 3D plot showing the decay rates for molars and incisors over time.
6. **Predict Decay Rates:** The trained model predicts the average decay rate based on input features.

## Extending the Application

- **Custom Parameters:** Modify the `TIME_PERIOD`, `INITIAL_BACTERIA`, and `BASE_DECAY_RATE` to simulate different scenarios.
- **Expand the Dataset:** Change the ranges of `diet_abrasiveness_values`, `dental_checkups_values`, `sugar_intake_values`, and `brushing_freq_values` to generate a larger or more diverse dataset.
- **Experiment with Different Models:** Try different machine learning models for potentially better predictions.

## Notes

- This application is for simulation and educational purposes and should not be used for medical advice.
- The accuracy of the simulation and predictions depends on the quality and range of the synthetic data.

## License

This project is open-source and free to use or modify.
