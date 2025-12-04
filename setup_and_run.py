"""
FraudSight PRO - One-Click Setup Script
Automatically sets up data, trains model, and runs the dashboard
"""

import os
import sys
import subprocess

def print_banner():
    print("=" * 70)
    print("🛡️  FRAUDSIGHT PRO - SETUP WIZARD")
    print("=" * 70)
    print()

def check_dependencies():
    """Check if all required packages are installed"""
    print("📦 Checking dependencies...")
    required = ['pandas', 'numpy', 'streamlit', 'plotly', 'matplotlib', 'networkx', 'sklearn', 'joblib']
    missing = []
    
    for package in required:
        try:
            if package == 'sklearn':
                __import__('sklearn')
            else:
                __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Installing missing packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
        print("✅ All packages installed!")
    else:
        print("\n✅ All dependencies satisfied!")
    print()

def create_directories():
    """Create necessary directories"""
    print("📁 Creating directories...")
    dirs = ['./data', './models']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f"  ✅ {dir_path}")
    print()

def generate_training_data():
    """Generate clean transaction data for ML training"""
    print("📊 Generating training data...")
    try:
        from simulator import generate_clean_data
        generate_clean_data(20000)
        print("✅ Training data generated!")
    except Exception as e:
        print(f"❌ Error generating data: {e}")
        return False
    print()
    return True

def train_ml_model():
    """Train the ML model"""
    print("🤖 Training ML model...")
    try:
        from anomaly_model import train_and_save_model
        train_and_save_model()
        print("✅ ML model trained and saved!")
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return False
    print()
    return True

def setup_personas():
    """Setup user personas"""
    print("👥 Setting up user personas...")
    try:
        from simulator import USER_IDS
        from persona_engine import setup_user_personas
        setup_user_personas(USER_IDS)
        print("✅ User personas created!")
    except Exception as e:
        print(f"❌ Error setting up personas: {e}")
        return False
    print()
    return True

def run_dashboard():
    """Launch the Streamlit dashboard"""
    print("=" * 70)
    print("🚀 LAUNCHING FRAUDSIGHT PRO DASHBOARD")
    print("=" * 70)
    print()
    print("📌 Dashboard will open in your browser automatically")
    print("📌 Press Ctrl+C to stop the server")
    print()
    
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard.py"])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped. Thank you for using FraudSight PRO!")

def main():
    print_banner()
    
    # Step 1: Check dependencies
    check_dependencies()
    
    # Step 2: Create directories
    create_directories()
    
    # Step 3: Check if setup is needed
    need_setup = not os.path.exists('./data/clean_transactions.csv')
    need_model = not os.path.exists('./models/isolation_forest_model.pkl')
    need_personas = not os.path.exists('./data/user_personas.csv')
    
    if need_setup or need_model or need_personas:
        print("🔧 First-time setup required...\n")
        
        if need_setup:
            if not generate_training_data():
                print("❌ Setup failed at data generation stage")
                return
        
        if need_model:
            if not train_ml_model():
                print("❌ Setup failed at model training stage")
                return
        
        if need_personas:
            if not setup_personas():
                print("❌ Setup failed at persona setup stage")
                return
        
        print("✅ SETUP COMPLETE!")
        print()
    else:
        print("✅ System already configured!")
        print()
    
    # Step 4: Run the dashboard
    run_dashboard()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("\nPlease check:")
        print("  1. All Python files are in the same directory")
        print("  2. You have write permissions in this directory")
        print("  3. All dependencies are installed")
        sys.exit(1)