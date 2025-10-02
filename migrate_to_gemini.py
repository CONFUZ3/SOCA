#!/usr/bin/env python3
"""
Migration script to help users switch from Claude to Gemini API
"""
import os
from pathlib import Path

def migrate_env_file():
    """Update .env file to use Gemini API key"""
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, "r") as f:
            content = f.read()
        
        # Replace Anthropic with Gemini
        if "ANTHROPIC_API_KEY" in content:
            content = content.replace("ANTHROPIC_API_KEY", "GEMINI_API_KEY")
            with open(env_path, "w") as f:
                f.write(content)
            print("✓ Updated .env file")
        else:
            print("ℹ .env file doesn't contain ANTHROPIC_API_KEY")
    else:
        print("ℹ No .env file found")

def migrate_streamlit_secrets():
    """Update Streamlit secrets file"""
    secrets_path = Path(".streamlit/secrets.toml")
    if secrets_path.exists():
        with open(secrets_path, "r") as f:
            content = f.read()
        
        if "ANTHROPIC_API_KEY" in content:
            content = content.replace("ANTHROPIC_API_KEY", "GEMINI_API_KEY")
            with open(secrets_path, "w") as f:
                f.write(content)
            print("✓ Updated .streamlit/secrets.toml")
        else:
            print("ℹ Streamlit secrets doesn't contain ANTHROPIC_API_KEY")
    else:
        print("ℹ No .streamlit/secrets.toml found")

def main():
    print("=" * 50)
    print("Migration: Claude → Gemini API")
    print("=" * 50)
    
    print("\n1. Updating configuration files...")
    migrate_env_file()
    migrate_streamlit_secrets()
    
    print("\n2. Next steps:")
    print("   - Get your Gemini API key from: https://aistudio.google.com/")
    print("   - Update your API key in .env or .streamlit/secrets.toml")
    print("   - Install new dependencies: pip install -r requirements.txt")
    print("   - Run: streamlit run app.py")
    
    print("\n" + "=" * 50)
    print("Migration Complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()
