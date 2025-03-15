import pandas as pd
import numpy as np
import random

def generate_sleep_apnea_dataset(num_samples=1000):
    data = []
    for _ in range(num_samples):
        sleep_apnea = random.choices([0, 1], weights=[0.7, 0.3])[0]  # Assuming 30% prevalence of sleep apnea
        heart_rate = 0
        body_fat_percentage = 0
        basal_energy_burned = 0
        weight = 0
        height = 0
        sleep_asleep = 0

        if sleep_apnea == 0:  # No sleep apnea
            heart_rate = np.random.normal(65, 8)  # Normal resting heart rate
            body_fat_percentage = np.random.normal(23, 6) # Healthy range
            basal_energy_burned = np.random.normal(1600, 250) # Based on average metabolic rates
            weight = np.random.normal(70, 15) # Average weight
            height = np.random.normal(175, 10) # Average height (in cm)
            sleep_asleep = np.random.normal(7.5, 1.5) # Average sleep hours

        else:  # Sleep apnea
            heart_rate = np.random.normal(75, 12) # Higher resting heart rate or fluctuations
            body_fat_percentage = np.random.normal(35, 10) # Higher body fat percentage
            basal_energy_burned = np.random.normal(1500, 300) # Can be slightly lower due to metabolic disruptions
            weight = np.random.normal(90, 25) # Higher weight
            height = np.random.normal(172, 12) # Slightly shorter on average, but variability exists
            sleep_asleep = np.random.normal(5.5, 2) # Shorter and more disrupted sleep

        # Add random noise
        heart_rate += np.random.normal(0, 5)
        body_fat_percentage += np.random.normal(0, 3)
        basal_energy_burned += np.random.normal(0, 150)
        weight += np.random.normal(0, 10)
        height += np.random.normal(0, 3)
        sleep_asleep += np.random.normal(0, 0.5)

        data.append([heart_rate, body_fat_percentage, basal_energy_burned, weight, height, sleep_asleep, sleep_apnea])

    df = pd.DataFrame(data, columns=['HEART_RATE', 'BODY_FAT_PERCENTAGE', 'BASAL_ENERGY_BURNED', 'WEIGHT', 'HEIGHT', 'SLEEP_ASLEEP', 'SLEEP_APNEA'])
    return df

# Generate and save the sleep apnea dataset
sleep_apnea_dataset = generate_sleep_apnea_dataset()
sleep_apnea_dataset.to_csv('sleep_apnea_dataset.csv', index=False)

print("Sleep apnea dataset generated and saved to sleep_apnea_dataset.csv")
