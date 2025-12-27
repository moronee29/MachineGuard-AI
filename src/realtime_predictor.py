"""
Real-Time Predictive Maintenance System
Simulates streaming sensor data and makes live predictions with animated dashboard.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
from datetime import datetime
import joblib
import os
from collections import deque
import warnings

warnings.filterwarnings('ignore')

class RealTimePredictiveSystem:
    """Real-time predictive maintenance system with live dashboard."""
    
    def __init__(self, model_path='models/xgboost_model.pkl', 
                 scaler_path='models/scaler.pkl',
                 features_path='models/selected_features.txt'):
        """
        Initialize the real-time prediction system.
        
        Args:
            model_path: Path to trained model
            scaler_path: Path to fitted scaler
            features_path: Path to feature names file
        """
        print("🔄 Initializing Real-Time Predictive Maintenance System...")
        
        # Load trained model and scaler
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # Load feature names
        with open(features_path, 'r') as f:
            self.features = [line.strip() for line in f.readlines()]
        
        print(f"✓ Model loaded: {model_path}")
        print(f"✓ Scaler loaded: {scaler_path}")
        print(f"✓ Features: {len(self.features)}")
        
        # Real-time data buffers
        self.max_history = 100  # Keep last 100 readings
        self.timestamps = deque(maxlen=self.max_history)
        self.predictions = deque(maxlen=self.max_history)
        self.probabilities = deque(maxlen=self.max_history)
        self.sensor_data = {feature: deque(maxlen=self.max_history) for feature in self.features}
        
        # Prediction log
        self.prediction_log = []
        
        # Machine state simulation
        self.machine_state = {
            'vibration_baseline': 2.5,
            'temperature_baseline': 70.0,
            'pressure_baseline': 5.5,
            'degradation_factor': 0.0,  # 0 = healthy, 1 = failing
            'failure_mode': False
        }
        
        print("✓ System initialized and ready!\n")
    
    def simulate_sensor_reading(self):
        """
        Simulate a single sensor reading from industrial machine.
        Includes realistic patterns and potential degradation.
        
        Returns:
            dict: Sensor readings for all features
        """
        # Gradually increase degradation (simulates wear over time)
        if np.random.random() < 0.02:  # 2% chance to enter failure mode
            self.machine_state['failure_mode'] = True
        
        if self.machine_state['failure_mode']:
            self.machine_state['degradation_factor'] = min(1.0, self.machine_state['degradation_factor'] + 0.05)
        else:
            self.machine_state['degradation_factor'] = max(0.0, self.machine_state['degradation_factor'] - 0.01)
        
        deg = self.machine_state['degradation_factor']
        
        # Generate sensor readings with degradation effects
        reading = {}
        
        # Core sensors (always present)
        reading['vibration'] = np.random.normal(2.5 + deg * 5, 0.3)
        reading['temperature'] = np.random.normal(70 + deg * 20, 2)
        reading['pressure'] = np.random.normal(5.5 - deg * 1.5, 0.2)
        
        # Check if feature exists before adding
        if 'runtime_hours' in self.features:
            reading['runtime_hours'] = len(self.timestamps)
        
        if 'voltage' in self.features:
            reading['voltage'] = np.random.normal(220 - deg * 10, 3)
        
        if 'current' in self.features:
            reading['current'] = np.random.normal(12 + deg * 6, 0.8)
        
        if 'acoustic_emission' in self.features:
            reading['acoustic_emission'] = np.random.normal(65 + deg * 15, 3)
        
        if 'rotation_speed' in self.features:
            reading['rotation_speed'] = np.random.normal(1500 - deg * 80, 5 + deg * 20)
        
        if 'torque' in self.features:
            reading['torque'] = np.random.normal(150 - deg * 40, 3)
        
        if 'power_consumption' in self.features:
            reading['power_consumption'] = np.random.normal(22 + deg * 8, 1.5)
        
        # Calculate rolling features if needed
        if 'vibration_rolling_avg' in self.features:
            if len(self.sensor_data.get('vibration', [])) >= 5:
                recent_vib = list(self.sensor_data['vibration'])[-5:]
                reading['vibration_rolling_avg'] = np.mean(recent_vib)
            else:
                reading['vibration_rolling_avg'] = reading.get('vibration', 2.5)
        
        if 'temperature_rolling_avg' in self.features:
            if len(self.sensor_data.get('temperature', [])) >= 5:
                recent_temp = list(self.sensor_data['temperature'])[-5:]
                reading['temperature_rolling_avg'] = np.mean(recent_temp)
            else:
                reading['temperature_rolling_avg'] = reading.get('temperature', 70)
        
        if 'pressure_rolling_avg' in self.features:
            if len(self.sensor_data.get('pressure', [])) >= 5:
                recent_press = list(self.sensor_data['pressure'])[-5:]
                reading['pressure_rolling_avg'] = np.mean(recent_press)
            else:
                reading['pressure_rolling_avg'] = reading.get('pressure', 5.5)
        
        if 'vibration_rolling_std' in self.features:
            if len(self.sensor_data.get('vibration', [])) >= 5:
                recent_vib = list(self.sensor_data['vibration'])[-5:]
                reading['vibration_rolling_std'] = np.std(recent_vib)
            else:
                reading['vibration_rolling_std'] = 0.0
        
        if 'current_rolling_std' in self.features:
            if len(self.sensor_data.get('current', [])) >= 5:
                recent_curr = list(self.sensor_data['current'])[-5:]
                reading['current_rolling_std'] = np.std(recent_curr)
            else:
                reading['current_rolling_std'] = 0.0
        
        if 'power_efficiency' in self.features:
            torque_val = reading.get('torque', 150)
            power_val = reading.get('power_consumption', 22)
            reading['power_efficiency'] = power_val / (torque_val + 1)
        
        if 'temp_vib_interaction' in self.features:
            temp_val = reading.get('temperature', 70)
            vib_val = reading.get('vibration', 2.5)
            reading['temp_vib_interaction'] = temp_val * vib_val
        
        return reading
    
    def predict(self, sensor_reading):
        """
        Make real-time prediction on sensor reading.
        
        Args:
            sensor_reading: dict of sensor values
            
        Returns:
            tuple: (prediction, probability)
        """
        # Prepare features in correct order
        features_ordered = [sensor_reading.get(f, 0) for f in self.features]
        
        # Convert to array and scale
        X = np.array(features_ordered).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Predict
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0, 1]
        
        return prediction, probability
    
    def log_prediction(self, timestamp, sensor_reading, prediction, probability):
        """Log prediction to file and memory."""
        log_entry = {
            'timestamp': timestamp,
            'prediction': 'FAILURE' if prediction == 1 else 'NORMAL',
            'probability': probability,
            'degradation': self.machine_state['degradation_factor'],
            **sensor_reading
        }
        self.prediction_log.append(log_entry)
    
    def run_simulation(self, duration_seconds=60, interval_ms=500):
        """
        Run real-time simulation for specified duration.
        
        Args:
            duration_seconds: How long to run simulation
            interval_ms: Milliseconds between readings
        """
        print("="*70)
        print("🚀 STARTING REAL-TIME PREDICTION SIMULATION")
        print("="*70)
        print(f"Duration: {duration_seconds} seconds")
        print(f"Interval: {interval_ms} ms")
        print(f"Expected readings: ~{int(duration_seconds * 1000 / interval_ms)}")
        print("\nPress Ctrl+C to stop early\n")
        
        start_time = time.time()
        reading_count = 0
        alerts = 0
        
        try:
            while (time.time() - start_time) < duration_seconds:
                # Get current timestamp
                timestamp = datetime.now()
                
                # Simulate sensor reading
                sensor_reading = self.simulate_sensor_reading()
                
                # Make prediction
                prediction, probability = self.predict(sensor_reading)
                
                # Update buffers
                self.timestamps.append(timestamp)
                self.predictions.append(prediction)
                self.probabilities.append(probability)
                
                for feature, value in sensor_reading.items():
                    if feature in self.sensor_data:
                        self.sensor_data[feature].append(value)
                
                # Log prediction
                self.log_prediction(timestamp, sensor_reading, prediction, probability)
                
                # Display real-time status
                reading_count += 1
                status = "🔴 ALERT - FAILURE PREDICTED" if prediction == 1 else "🟢 Normal"
                
                if prediction == 1:
                    alerts += 1
                
                print(f"[{timestamp.strftime('%H:%M:%S')}] "
                      f"Reading #{reading_count:3d} | "
                      f"Prob: {probability:5.1%} | "
                      f"Deg: {self.machine_state['degradation_factor']:4.1%} | "
                      f"{status}")
                
                # Sleep interval
                time.sleep(interval_ms / 1000)
                
        except KeyboardInterrupt:
            print("\n\n⚠️  Simulation stopped by user")
        
        # Summary
        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print("📊 SIMULATION SUMMARY")
        print("="*70)
        print(f"Total readings: {reading_count}")
        print(f"Duration: {elapsed:.1f} seconds")
        print(f"Alerts triggered: {alerts} ({alerts/reading_count*100:.1f}%)")
        print(f"Average probability: {np.mean(list(self.probabilities)):.1%}")
        print(f"Max probability: {np.max(list(self.probabilities)):.1%}")
        print("="*70 + "\n")
        
        # Save log
        self.save_prediction_log()
    
    def save_prediction_log(self):
        """Save prediction log to CSV."""
        if self.prediction_log:
            df = pd.DataFrame(self.prediction_log)
            log_file = f'results/realtime_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            df.to_csv(log_file, index=False)
            print(f"✓ Prediction log saved: {log_file}")
    
    def create_dashboard(self, save_path='realtime_dashboard.png'):
        """
        Create static dashboard visualization from collected data.
        
        Args:
            save_path: Where to save the dashboard image
        """
        if len(self.timestamps) == 0:
            print("⚠️  No data to visualize. Run simulation first.")
            return
        
        print("\n📊 Creating real-time dashboard visualization...")
        
        fig = plt.figure(figsize=(16, 12))
        
        # Convert timestamps to relative seconds
        time_seconds = [(t - self.timestamps[0]).total_seconds() for t in self.timestamps]
        
        # 1. Failure Probability Over Time
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(time_seconds, list(self.probabilities), color='#e74c3c', linewidth=2)
        ax1.axhline(y=0.7, color='red', linestyle='--', label='High Risk Threshold')
        ax1.fill_between(time_seconds, 0, list(self.probabilities), 
                        where=[p > 0.7 for p in self.probabilities],
                        alpha=0.3, color='red', label='Alert Zone')
        ax1.set_xlabel('Time (seconds)', fontweight='bold')
        ax1.set_ylabel('Failure Probability', fontweight='bold')
        ax1.set_title('Real-Time Failure Probability', fontweight='bold', fontsize=12)
        ax1.legend()
        ax1.grid(alpha=0.3)
        ax1.set_ylim([0, 1])
        
        # 2. Vibration Sensor
        ax2 = plt.subplot(3, 2, 2)
        if 'vibration' in self.sensor_data:
            ax2.plot(time_seconds, list(self.sensor_data['vibration']), 
                    color='#3498db', linewidth=1.5)
            ax2.set_xlabel('Time (seconds)', fontweight='bold')
            ax2.set_ylabel('Vibration (mm/s)', fontweight='bold')
            ax2.set_title('Vibration Sensor Reading', fontweight='bold', fontsize=12)
            ax2.grid(alpha=0.3)
        
        # 3. Temperature Sensor
        ax3 = plt.subplot(3, 2, 3)
        if 'temperature' in self.sensor_data:
            ax3.plot(time_seconds, list(self.sensor_data['temperature']), 
                    color='#e67e22', linewidth=1.5)
            ax3.set_xlabel('Time (seconds)', fontweight='bold')
            ax3.set_ylabel('Temperature (°C)', fontweight='bold')
            ax3.set_title('Temperature Sensor Reading', fontweight='bold', fontsize=12)
            ax3.grid(alpha=0.3)
        
        # 4. Pressure Sensor
        ax4 = plt.subplot(3, 2, 4)
        if 'pressure' in self.sensor_data:
            ax4.plot(time_seconds, list(self.sensor_data['pressure']), 
                    color='#9b59b6', linewidth=1.5)
            ax4.set_xlabel('Time (seconds)', fontweight='bold')
            ax4.set_ylabel('Pressure (bar)', fontweight='bold')
            ax4.set_title('Pressure Sensor Reading', fontweight='bold', fontsize=12)
            ax4.grid(alpha=0.3)
        
        # 5. Prediction Timeline
        ax5 = plt.subplot(3, 2, 5)
        colors = ['#2ecc71' if p == 0 else '#e74c3c' for p in self.predictions]
        ax5.scatter(time_seconds, list(self.predictions), c=colors, alpha=0.6, s=50)
        ax5.set_xlabel('Time (seconds)', fontweight='bold')
        ax5.set_ylabel('Prediction', fontweight='bold')
        ax5.set_title('Prediction Timeline (Green=Normal, Red=Failure)', 
                     fontweight='bold', fontsize=12)
        ax5.set_yticks([0, 1])
        ax5.set_yticklabels(['Normal', 'Failure'])
        ax5.grid(alpha=0.3)
        
        # 6. Machine Degradation
        ax6 = plt.subplot(3, 2, 6)
        degradation_history = [entry['degradation'] for entry in self.prediction_log]
        ax6.plot(time_seconds[:len(degradation_history)], degradation_history, 
                color='#c0392b', linewidth=2)
        ax6.fill_between(time_seconds[:len(degradation_history)], 0, degradation_history,
                        alpha=0.3, color='#c0392b')
        ax6.set_xlabel('Time (seconds)', fontweight='bold')
        ax6.set_ylabel('Degradation Factor', fontweight='bold')
        ax6.set_title('Machine Health Degradation', fontweight='bold', fontsize=12)
        ax6.grid(alpha=0.3)
        ax6.set_ylim([0, 1])
        
        plt.suptitle('🏭 Real-Time Predictive Maintenance Dashboard', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Dashboard saved: {save_path}")
        plt.close()


def main():
    """Main execution function."""
    print("\n" + "🏭 REAL-TIME PREDICTIVE MAINTENANCE SYSTEM" + "\n")
    
    # Check if models exist
    if not os.path.exists('models/xgboost_model.pkl'):
        print("⚠️  Models not found!")
        print("Please run 'python predictive_maintenance.py' first to train models.\n")
        return
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Initialize system
    system = RealTimePredictiveSystem()
    
    # Menu
    print("=" * 70)
    print("SELECT SIMULATION MODE")
    print("=" * 70)
    print("1. Quick Demo (30 seconds)")
    print("2. Standard Run (60 seconds)")
    print("3. Extended Run (120 seconds)")
    print("4. Custom duration")
    print("=" * 70)
    
    choice = input("\nEnter choice (1-4) [default: 1]: ").strip() or "1"
    
    duration_map = {'1': 30, '2': 60, '3': 120}
    
    if choice in duration_map:
        duration = duration_map[choice]
    elif choice == '4':
        duration = int(input("Enter duration in seconds: "))
    else:
        duration = 30
    
    # Run simulation
    system.run_simulation(duration_seconds=duration, interval_ms=500)
    
    # Create dashboard
    system.create_dashboard()
    
    print("\n✅ Real-time simulation complete!")
    print(f"📊 Check 'results/' folder for prediction logs")
    print(f"📈 Dashboard saved as 'realtime_dashboard.png'\n")


if __name__ == "__main__":
    main()
