"""Setup script for the Spatial Optimization Conversational Agent"""
import os
import sys
from pathlib import Path

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_path = Path(".env")
    if not env_path.exists():
        print("Creating .env file...")
        with open(env_path, "w") as f:
            f.write("# Google Gemini API Key (required)\n")
            f.write("GEMINI_API_KEY=your_api_key_here\n\n")
            f.write("# Gurobi (optional - will fall back to PuLP if not available)\n")
            f.write("# GRB_LICENSE_FILE=/path/to/gurobi.lic\n")
        print("✓ Created .env file")
        print("  Please edit .env and add your GEMINI_API_KEY")
    else:
        print("✓ .env file already exists")

def create_streamlit_secrets():
    """Create Streamlit secrets directory and file"""
    secrets_dir = Path(".streamlit")
    secrets_dir.mkdir(exist_ok=True)
    
    secrets_path = secrets_dir / "secrets.toml"
    if not secrets_path.exists():
        print("Creating .streamlit/secrets.toml...")
        with open(secrets_path, "w") as f:
            f.write("# Google Gemini API Key\n")
            f.write('GEMINI_API_KEY = "your_api_key_here"\n')
        print("✓ Created .streamlit/secrets.toml")
        print("  Please edit this file and add your API key")
    else:
        print("✓ .streamlit/secrets.toml already exists")

def create_directories():
    """Create necessary directories"""
    dirs = ["data", "temp", "tests/test_data", "docs/problems"]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {dir_path}")

def generate_test_data():
    """Generate test data"""
    try:
        print("\nGenerating test data...")
        import geopandas as gpd
        from tests.generate_test_data import generate_test_data
        
        test_data_dir = Path("tests/test_data")
        generate_test_data(test_data_dir)
        print("✓ Test data generated successfully")
    except ImportError as e:
        print(f"⚠ Could not generate test data: {e}")
        print("  Install dependencies first: pip install -r requirements.txt")
        print("  Then run: python tests/generate_test_data.py")
    except Exception as e:
        print(f"⚠ Error generating test data: {e}")

def check_dependencies():
    """Check if key dependencies are installed"""
    print("\nChecking dependencies...")
    
    required = [
        "streamlit",
        "google-generativeai",
        "geopandas",
        "folium",
        "pulp"
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠ Missing dependencies: {', '.join(missing)}")
        print("  Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed")
        return True

def main():
    """Main setup function"""
    print("=" * 60)
    print("Spatial Optimization Conversational Agent - Setup")
    print("=" * 60)
    
    print("\n1. Creating configuration files...")
    create_env_file()
    create_streamlit_secrets()
    
    print("\n2. Creating directories...")
    create_directories()
    
    print("\n3. Checking dependencies...")
    deps_ok = check_dependencies()
    
    if deps_ok:
        print("\n4. Generating test data...")
        generate_test_data()
    else:
        print("\n4. Skipping test data generation (install dependencies first)")
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    
    print("\nNext steps:")
    print("1. Edit .env or .streamlit/secrets.toml with your GEMINI_API_KEY")
    print("2. Run: streamlit run app.py")
    print("3. Open http://localhost:8501 in your browser")
    print("\nFor help, see README.md or docs/architecture.md")
    print("=" * 60)

if __name__ == "__main__":
    main()

