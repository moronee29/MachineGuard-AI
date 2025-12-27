"""
One-Click Launcher for Predictive Maintenance System
The easiest way to use the system - just run this!
"""

import os
import sys
import subprocess
import time
from datetime import datetime

class Colors:
    """ANSI color codes for terminal"""
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    """Print formatted header"""
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text.center(70)}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*70}{Colors.END}\n")

def print_success(text):
    """Print success message"""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_error(text):
    """Print error message"""
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

def print_warning(text):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def check_dependencies():
    """Check if all dependencies are installed"""
    print_header("CHECKING DEPENDENCIES")
    
    try:
        result = subprocess.run(
            [sys.executable, os.path.join("src", "verify_setup.py")],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print_success("All dependencies installed!")
            return True
        else:
            print_error("Some dependencies missing")
            print("\nWould you like to install them now? (y/n): ", end="")
            choice = input().lower()
            
            if choice == 'y':
                print_info("Installing dependencies...")
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
                print_success("Dependencies installed!")
                return True
            else:
                print_error("Please install dependencies manually: pip install -r requirements.txt")
                return False
    except Exception as e:
        print_error(f"Error checking dependencies: {e}")
        return False

def check_models():
    """Check if models are trained"""
    return os.path.exists("models/xgboost_model.pkl")

def train_models():
    """Train the ML models"""
    print_header("TRAINING ML MODELS")
    print_info("This will take 1-2 minutes...")
    print_info("Training Random Forest and XGBoost models...")
    
    try:
        subprocess.run([sys.executable, os.path.join("src", "predictive_maintenance.py")])
        print_success("Models trained successfully!")
        return True
    except Exception as e:
        print_error(f"Error training models: {e}")
        return False

def run_realtime_demo():
    """Run real-time prediction demo"""
    print_header("REAL-TIME PREDICTION DEMO")
    
    try:
        subprocess.run([sys.executable, os.path.join("src", "realtime_predictor.py")])
        return True
    except Exception as e:
        print_error(f"Error running demo: {e}")
        return False

def open_dashboard():
    """Open the interactive dashboard"""
    dashboard_path = os.path.join("dashboards", "dashboard.html")
    
    if not os.path.exists(dashboard_path):
        print_warning("Dashboard not found. Please train models first.")
        return False
    
    print_info("Opening interactive dashboard...")
    
    try:
        if sys.platform == "darwin":  # Mac
            subprocess.run(["open", dashboard_path])
        elif sys.platform == "linux":  # Linux
            subprocess.run(["xdg-open", dashboard_path])
        else:  # Windows
            os.startfile(dashboard_path)
        
        print_success(f"Dashboard opened: {dashboard_path}")
        return True
    except Exception as e:
        print_warning(f"Could not auto-open dashboard: {e}")
        print_info(f"Please open manually: {os.path.abspath(dashboard_path)}")
        return False

def view_results():
    """Show results summary"""
    print_header("RESULTS SUMMARY")
    
    # Check for result files
    results_file = "results/prediction_results.csv"
    high_risk_file = "results/high_risk_machines.csv"
    
    if os.path.exists(results_file):
        import pandas as pd
        df = pd.read_csv(results_file)
        
        print(f"📊 Total Machines Analyzed: {len(df)}")
        print(f"🔴 High Risk: {(df['Risk_Level'] == 'High').sum()}")
        print(f"🟡 Medium Risk: {(df['Risk_Level'] == 'Medium').sum()}")
        print(f"🟢 Low Risk: {(df['Risk_Level'] == 'Low').sum()}")
        
        if os.path.exists(high_risk_file):
            high_risk = pd.read_csv(high_risk_file)
            if len(high_risk) > 0:
                print(f"\n⚠️  {len(high_risk)} machines require immediate attention!")
                print("\nTop 5 High-Risk Machines:")
                print(high_risk.head()[['Machine_ID', 'Failure_Probability']].to_string(index=False))
        
        print(f"\n📁 Full results: {os.path.abspath(results_file)}")
        print(f"📁 High-risk list: {os.path.abspath(high_risk_file)}")
    else:
        print_warning("No results found. Please run training first.")
    
    input("\nPress Enter to continue...")

def show_menu():
    """Display main menu"""
    print_header("🏭 PREDICTIVE MAINTENANCE SYSTEM")
    print(f"{Colors.BOLD}Welcome! This launcher makes everything easy.{Colors.END}\n")
    
    # Check status
    models_trained = check_models()
    
    print("📊 System Status:")
    if models_trained:
        print_success("Models trained and ready")
    else:
        print_warning("Models not trained yet")
    
    print("\n" + "="*70)
    print(f"{Colors.BOLD}MAIN MENU{Colors.END}")
    print("="*70)
    print()
    print("1️⃣  Complete Setup (First-time users)")
    print("     → Check dependencies → Train models → Run demo")
    print()
    print("2️⃣  Train/Retrain Models")
    print("     → Train ML models on data (1-2 minutes)")
    print()
    print("3️⃣  Run Real-Time Demo")
    print("     → See live predictions streaming (30-120 seconds)")
    print()
    print("4️⃣  View Interactive Dashboard")
    print("     → Open dashboard.html in browser")
    print()
    print("5️⃣  View Results Summary")
    print("     → See prediction statistics and high-risk machines")
    print()
    print("6️⃣  Help & Documentation")
    print("     → Open user guides")
    print()
    print("7️⃣  Exit")
    print()
    print("="*70)

def complete_setup():
    """Run complete first-time setup"""
    print_header("COMPLETE SETUP")
    print_info("This will set up everything for first-time use\n")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        return False
    
    time.sleep(1)
    
    # Step 2: Train models
    print()
    if not train_models():
        return False
    
    time.sleep(1)
    
    # Step 3: Ask about demo
    print()
    print_success("Setup complete!")
    print("\nWould you like to run a real-time demo now? (y/n): ", end="")
    choice = input().lower()
    
    if choice == 'y':
        print()
        run_realtime_demo()
        print()
        open_dashboard()
    
    return True

def show_help():
    """Show help and documentation"""
    print_header("HELP & DOCUMENTATION")
    
    print("📚 Available Documentation:\n")
    print("1. START_HERE.txt      - Quick overview")
    print("2. HOW_TO_USE.md       - Simple usage guide")
    print("3. USER_GUIDE.md       - Detailed manual")
    print("4. README.md           - Project overview")
    print("5. PROJECT_SUMMARY.md  - Technical details")
    
    print("\n🔧 Quick Help:\n")
    print("• First time? Choose option 1 (Complete Setup)")
    print("• Already set up? Choose option 3 (Real-Time Demo)")
    print("• Want to see results? Choose option 5 (View Results)")
    
    print("\n💡 Tips:\n")
    print("• Models need training once (option 2)")
    print("• Real-time demo can run unlimited times (option 3)")
    print("• Dashboard updates after each training (option 4)")
    
    input("\nPress Enter to return to menu...")

def run_cli_mode():
    """Run in CLI mode with interactive menu"""
    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        show_menu()
        
        choice = input(f"\n{Colors.BOLD}Choose an option (1-7): {Colors.END}").strip()
        
        if choice == '1':
            complete_setup()
            input("\nPress Enter to return to menu...")
        
        elif choice == '2':
            if not check_dependencies():
                print_error("Please install dependencies first")
                input("\nPress Enter to return to menu...")
                continue
            train_models()
            input("\nPress Enter to return to menu...")
        
        elif choice == '3':
            if not check_models():
                print_warning("Models not trained yet!")
                print("Please train models first (option 2)")
                input("\nPress Enter to return to menu...")
                continue
            run_realtime_demo()
            open_dashboard()
            input("\nPress Enter to return to menu...")
        
        elif choice == '4':
            open_dashboard()
            input("\nPress Enter to return to menu...")
        
        elif choice == '5':
            view_results()
        
        elif choice == '6':
            show_help()
        
        elif choice == '7':
            print_header("THANK YOU!")
            print(f"{Colors.GREEN}Thanks for using Predictive Maintenance System!{Colors.END}\n")
            return
        
        else:
            print_error("Invalid choice. Please enter 1-7.")
            time.sleep(2)

def run_gui_mode():
    """Launch web-based GUI"""
    print_header("LAUNCHING WEB INTERFACE")
    print_info("Starting FastAPI web server...")
    print_info("The web interface will open in your browser\n")
    
    try:
        # Check if web_dashboard.py exists
        web_dashboard_path = os.path.join("src", "web_dashboard.py")
        if not os.path.exists(web_dashboard_path):
            print_error("Web dashboard not found!")
            print_info("Please ensure web_dashboard.py is in the src/ directory")
            input("\nPress Enter to continue...")
            return
        
        # Launch FastAPI server
        subprocess.Popen([sys.executable, web_dashboard_path])
        
        time.sleep(2)
        
        # Open browser
        import webbrowser
        webbrowser.open("http://localhost:8000")
        
        print_success("Web interface launched!")
        print_info("Server running at: http://localhost:8000")
        print_warning("Press Ctrl+C in terminal to stop the server")
        
        input("\nPress Enter when done...")
        
    except Exception as e:
        print_error(f"Error launching web interface: {e}")
        input("\nPress Enter to continue...")

def select_mode():
    """Ask user to select CLI or GUI mode"""
    os.system('clear' if os.name == 'posix' else 'cls')
    
    print(f"{Colors.BOLD}{Colors.CYAN}")
    print("=" * 70)
    print("🏭  PREDICTIVE MAINTENANCE SYSTEM  🏭".center(70))
    print("=" * 70)
    print(f"{Colors.END}\n")
    
    print(f"{Colors.BOLD}Welcome! Please select your preferred interface:{Colors.END}\n")
    
    print("┌─────────────────────────────────────────────────────────────────┐")
    print("│                                                                 │")
    print(f"│  {Colors.BOLD}1️⃣  CLI Mode (Command Line){Colors.END}                                  │")
    print("│     → Interactive menu in terminal                             │")
    print("│     → Fast and lightweight                                     │")
    print("│     → Perfect for developers                                   │")
    print("│                                                                 │")
    print(f"│  {Colors.BOLD}2️⃣  GUI Mode (Web Interface){Colors.END}                                 │")
    print("│     → Modern web dashboard                                     │")
    print("│     → Point-and-click interface                                │")
    print("│     → Perfect for clients/non-technical users                  │")
    print("│                                                                 │")
    print("└─────────────────────────────────────────────────────────────────┘")
    
    print()
    choice = input(f"{Colors.BOLD}Select mode (1 or 2): {Colors.END}").strip()
    
    return choice

def main():
    """Main launcher function"""
    try:
        mode = select_mode()
        
        if mode == '1':
            print_info("Starting CLI mode...\n")
            time.sleep(1)
            run_cli_mode()
        elif mode == '2':
            run_gui_mode()
        else:
            print_error("Invalid mode selected")
            time.sleep(1)
            main()  # Ask again
            
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}Program interrupted{Colors.END}")
        print(f"{Colors.GREEN}Goodbye!{Colors.END}\n")

if __name__ == "__main__":
    main()
