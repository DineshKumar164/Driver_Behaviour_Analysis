import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

class DriverDataGenerator:
    def __init__(self):
        self.routes = [
            "Route-1: Nehru Place - Connaught Place",
            "Route-2: Karol Bagh - Chandni Chowk",
            "Route-3: Dwarka - IGI Airport",
            "Route-4: Lajpat Nagar - India Gate"
        ]
        
    def generate_single_record(self):
        """Generate a single record of driver behavior data"""
        current_time = datetime.now()
        
        # Generate realistic values for each parameter
        speed = np.clip(np.random.normal(40, 15), 0, 80)  # Speed in km/h
        acceleration = np.clip(np.random.normal(0, 2), -4, 4)  # Acceleration in m/s²
        brake_intensity = np.clip(np.random.normal(0.3, 0.2), 0, 1)  # Normalized brake intensity
        route = random.choice(self.routes)
        
        # Calculate behavior score based on parameters
        score = self._calculate_behavior_score(speed, acceleration, brake_intensity)
        category = self._categorize_behavior(score)
        
        return {
            'timestamp': current_time,
            'speed': round(speed, 2),
            'acceleration': round(acceleration, 2),
            'brake_intensity': round(brake_intensity, 2),
            'route': route,
            'behavior_score': round(score, 2),
            'category': category
        }
    
    def generate_batch(self, n_records=100):
        """Generate a batch of driver behavior records"""
        records = [self.generate_single_record() for _ in range(n_records)]
        return pd.DataFrame(records)
    
    def _calculate_behavior_score(self, speed, acceleration, brake_intensity):
        """Calculate a behavior score based on driving parameters"""
        speed_score = 1 - (abs(speed - 40) / 40)  # Optimal speed around 40 km/h
        acc_score = 1 - (abs(acceleration) / 4)  # Penalize high acceleration/deceleration
        brake_score = 1 - brake_intensity  # Penalize heavy braking
        
        return (speed_score + acc_score + brake_score) / 3
    
    def _categorize_behavior(self, score):
        """Categorize behavior based on score"""
        if score >= 0.7:
            return 'Safe'
        elif score >= 0.4:
            return 'Moderate'
        else:
            return 'Risky'

if __name__ == "__main__":
    # Test the data generator
    generator = DriverDataGenerator()
    test_data = generator.generate_batch(5)
    print(test_data)
