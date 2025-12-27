"""
Predictive Maintenance Dataset Generator
Generates realistic simulated sensor data for industrial machine monitoring
with temporal patterns and degradation leading to failures.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_sensor_data(n_samples=10000, failure_rate=0.08, random_seed=42):
    """
    Generate realistic multi-sensor data for predictive maintenance.
    
    Parameters:
    - n_samples: Total number of samples to generate
    - failure_rate: Proportion of samples that represent failures
    - random_seed: Random seed for reproducibility
    
    Returns:
    - DataFrame with sensor readings and failure labels
    """
    np.random.seed(random_seed)
    
    # Calculate number of machines and samples per machine
    n_machines = 100
    samples_per_machine = n_samples // n_machines
    
    data = []
    
    for machine_id in range(1, n_machines + 1):
        # Determine if this machine will fail
        will_fail = np.random.random() < failure_rate
        
        # Generate timestamps
        start_date = datetime(2023, 1, 1) + timedelta(days=np.random.randint(0, 365))
        timestamps = [start_date + timedelta(hours=i) for i in range(samples_per_machine)]
        
        for idx, timestamp in enumerate(timestamps):
            # Calculate degradation factor (increases over time)
            degradation = idx / samples_per_machine
            
            if will_fail:
                # Machine deteriorates towards failure
                failure_proximity = degradation ** 2  # Accelerating degradation
            else:
                # Normal operation with minor variations
                failure_proximity = 0.1 * degradation
            
            # Generate sensor readings with realistic patterns
            
            # Vibration (mm/s): Normal 0-5, increases with degradation
            vibration = np.random.normal(2.0 + failure_proximity * 8, 0.5)
            vibration = max(0, vibration)  # No negative vibrations
            
            # Temperature (°C): Normal 60-75, increases with problems
            temperature = np.random.normal(68 + failure_proximity * 25, 3)
            temperature = np.clip(temperature, 50, 120)
            
            # Pressure (bar): Normal 5-6, drops when failing
            pressure = np.random.normal(5.5 - failure_proximity * 2, 0.3)
            pressure = max(0, pressure)
            
            # Runtime hours: Cumulative operational time
            runtime_hours = idx * 1.0 + np.random.normal(0, 0.1)
            runtime_hours = max(0, runtime_hours)
            
            # Voltage (V): Normal ~220V, fluctuates with issues
            voltage = np.random.normal(220 - failure_proximity * 15, 5)
            voltage = max(0, voltage)
            
            # Current (A): Normal 10-15A, spikes with mechanical issues
            current = np.random.normal(12 + failure_proximity * 8, 1)
            current = max(0, current)
            
            # Acoustic emission (dB): Normal 60-70, increases with bearing wear
            acoustic = np.random.normal(65 + failure_proximity * 20, 4)
            acoustic = max(0, acoustic)
            
            # Rotation speed (RPM): Normal 1450-1550, unstable when failing
            rotation_speed = np.random.normal(1500 - failure_proximity * 100, 10 + failure_proximity * 30)
            rotation_speed = max(0, rotation_speed)
            
            # Torque (Nm): Normal operation, drops with mechanical issues
            torque = np.random.normal(150 - failure_proximity * 50, 5)
            torque = max(0, torque)
            
            # Power consumption (kW): Increases with inefficiency
            power = np.random.normal(22 + failure_proximity * 10, 2)
            power = max(0, power)
            
            # Determine failure label
            # Last 5% of samples for failing machines are labeled as failure imminent
            if will_fail and degradation > 0.95:
                failure = 1
            else:
                failure = 0
            
            # Create sample record
            sample = {
                'timestamp': timestamp,
                'machine_id': f'MACHINE_{machine_id:03d}',
                'vibration': vibration,
                'temperature': temperature,
                'pressure': pressure,
                'runtime_hours': runtime_hours,
                'voltage': voltage,
                'current': current,
                'acoustic_emission': acoustic,
                'rotation_speed': rotation_speed,
                'torque': torque,
                'power_consumption': power,
                'failure': failure
            }
            
            data.append(sample)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some random missing values (realistic scenario)
    missing_rate = 0.02  # 2% missing data
    for col in df.columns:
        if col not in ['timestamp', 'machine_id', 'failure']:
            mask = np.random.random(len(df)) < missing_rate
            df.loc[mask, col] = np.nan
    
    # Shuffle the data
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    return df

if __name__ == "__main__":
    print("Generating Predictive Maintenance Dataset...")
    print("=" * 60)
    
    # Generate dataset
    df = generate_sensor_data(n_samples=10000, failure_rate=0.08)
    
    # Save to CSV
    output_file = os.path.join("data", "generated", "predictive_maintenance_data.csv")
    df.to_csv(output_file, index=False)
    
    print(f"\n✓ Dataset generated successfully!")
    print(f"✓ Saved to: {output_file}")
    print(f"\nDataset Summary:")
    print(f"  - Total samples: {len(df):,}")
    print(f"  - Number of machines: {df['machine_id'].nunique()}")
    print(f"  - Failure cases: {df['failure'].sum():,} ({df['failure'].mean()*100:.2f}%)")
    print(f"  - Normal cases: {(df['failure']==0).sum():,} ({(df['failure']==0).mean()*100:.2f}%)")
    print(f"  - Features: {len(df.columns) - 3}")  # Excluding timestamp, machine_id, failure
    print(f"  - Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nFeature columns:")
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'machine_id', 'failure']]
    for col in feature_cols:
        print(f"  - {col}")
    
    print(f"\n{'='*60}")
    print("Dataset ready for predictive maintenance analysis!")
