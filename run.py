"""
AI-Driven Predictive Maintenance System for Industrial Machines
Main entry point for the application.

Run this file to start the system.
"""

import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from launcher import main

if __name__ == "__main__":
    main()
