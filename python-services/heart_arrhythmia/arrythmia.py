import pandas as pd
import numpy as np
import random

def generate_arrhythmia_dataset(num_samples=1000):
    data = []
    for _ in range(num_samples):
        arrhythmia = random.choices([0, 1], weights=[0.8, 0.2])[0]  # 20% arrhythmia
        heart_rate = 0
        body_fat = 0
        basal_energy = 0
        total_calories = 0
        weight = 0
        steps = 0

        if arrhythmia == 0:  # No arrhythmia
            heart_rate = np.random.normal(70, 10)  # Normal range
            body_fat = np.random.normal(25, 5)
            basal_energy = np.random.normal(1500, 200)
            total_calories = np.random.normal(2500, 500)
            weight = np.random.normal(75, 15)
            steps = np.random.normal(8000, 2000)

        else:  # Arrhythmia
            heart_rate = np.random.normal(90, 20)  # Higher or irregular heart rate
            body_fat = np.random.normal(30, 8) #Higher body fat
            basal_energy = np.random.normal(1400, 300)
            total_calories = np.random.normal(2000, 600)
            weight = np.random.normal(85, 20)
            steps = np.random.normal(5000, 2500) #less active.

        # Add random noise
        heart_rate += np.random.normal(0, 3)
        body_fat += np.random.normal(0, 2)
        basal_energy += np.random.normal(0, 100)
        total_calories += np.random.normal(0, 200)
        weight += np.random.normal(0, 5)
        steps += np.random.normal(0, 500)

        data.append([heart_rate, body_fat, basal_energy, total_calories, weight, steps, arrhythmia])

    df = pd.DataFrame(data, columns=['HEART_RATE', 'BODY_FAT_PERCENTAGE', 'BASAL_ENERGY_BURNED', 'TOTAL_CALORIES_BURNED', 'WEIGHT', 'STEPS', 'ARRHYTHMIA'])
    return df

# Generate and save the dataset
dataset = generate_arrhythmia_dataset()
dataset.to_csv('arrhythmia_dataset.csv', index=False)

print("Dataset generated and saved to arrhythmia_dataset.csv")