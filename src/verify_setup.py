#!/usr/bin/env python3
"""
Verification script for Predictive Maintenance System
Checks that all dependencies are installed and files are present.
"""

import sys
import os

def check_dependencies():
    """Check if all required packages are installed."""
    print("="*60)
    print("CHECKING DEPENDENCIES")
    print("="*60)
    
    required_packages = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'sklearn',
        'xgboost',
        'plotly',
        'jupyter',
        'joblib'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"✓ {package:15s} - installed")
        except ImportError:
            print(f"✗ {package:15s} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies installed!")
        return True

def check_files():
    """Check if all required files are present."""
    print("\n" + "="*60)
    print("CHECKING FILES")
    print("="*60)
    
    required_files = [
        'data_generator.py',
        'predictive_maintenance.py',
        'realtime_predictor.py',
        'predictive_maintenance_data.csv',
        'requirements.txt',
        'README.md',
        'PROJECT_SUMMARY.md',
        'QUICK_START.md',
        'USER_GUIDE.md'
    ]
    
    required_dirs = [
        'models',
        'outputs',
        'results'
    ]
    
    missing = []
    for filename in required_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            size_str = f"{size:,} bytes" if size < 1024 else f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/(1024*1024):.1f} MB"
            print(f"✓ {filename:35s} - {size_str}")
        else:
            print(f"✗ {filename:35s} - MISSING")
            missing.append(filename)
    
    for dirname in required_dirs:
        if os.path.isdir(dirname):
            print(f"✓ {dirname + '/' :35s} - directory exists")
        else:
            print(f"✗ {dirname + '/' :35s} - MISSING")
            missing.append(dirname)
    
    if missing:
        print(f"\n⚠️  Missing files/dirs: {', '.join(missing)}")
        return False
    else:
        print("\n✅ All required files present!")
        return True

def check_dataset():
    """Verify dataset quality."""
    print("\n" + "="*60)
    print("CHECKING DATASET")
    print("="*60)
    
    try:
        import pandas as pd
        
        df = pd.read_csv('predictive_maintenance_data.csv')
        
        print(f"✓ Dataset loaded successfully")
        print(f"  - Rows: {len(df):,}")
        print(f"  - Columns: {len(df.columns)}")
        print(f"  - Machines: {df['machine_id'].nunique()}")
        print(f"  - Failures: {df['failure'].sum()}")
        print(f"  - Failure rate: {df['failure'].mean()*100:.2f}%")
        
        required_columns = ['timestamp', 'machine_id', 'vibration', 'temperature', 
                          'pressure', 'failure']
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            print(f"\n⚠️  Missing columns: {', '.join(missing_cols)}")
            return False
        else:
            print("\n✅ Dataset structure correct!")
            return True
            
    except Exception as e:
        print(f"\n✗ Error loading dataset: {e}")
        return False

def main():
    """Run all verification checks."""
    print("\n🔍 PREDICTIVE MAINTENANCE SYSTEM - VERIFICATION")
    print("This script checks if the project is properly set up.\n")
    
    deps_ok = check_dependencies()
    files_ok = check_files()
    data_ok = check_dataset()
    
    print("\n" + "="*60)
    print("VERIFICATION SUMMARY")
    print("="*60)
    
    if deps_ok and files_ok and data_ok:
        print("✅ All checks passed!")
        print("\n🚀 Ready to run:")
        print("   python predictive_maintenance.py   # Train models")
        print("   python realtime_predictor.py        # Real-time simulation")
        print("\nOr see QUICK_START.md or USER_GUIDE.md for detailed instructions.")
        return 0
    else:
        print("⚠️  Some checks failed. Please review above.")
        if not deps_ok:
            print("\n   Fix: pip install -r requirements.txt")
        if not data_ok:
            print("\n   Fix: python data_generator.py")
        return 1

if __name__ == "__main__":
    sys.exit(main())
