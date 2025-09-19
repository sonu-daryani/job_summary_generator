#!/usr/bin/env python3
"""
Setup script for Job Summary Generator.
Helps users set up the environment and configuration.
"""
import os
import sys
import subprocess
from pathlib import Path

def create_env_file():
    """Create .env file with template values."""
    env_content = """# Hugging Face API Configuration
HUGGINGFACE_API_KEY=your_free_huggingface_api_key_here
HUGGINGFACE_API_URL=https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium

# Application Configuration
APP_NAME=Job Summary Generator
DEBUG=True
HOST=0.0.0.0
PORT=8000
"""
    
    env_file = Path(".env")
    if not env_file.exists():
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ Created .env file with template values")
        print("⚠️  Please update HUGGINGFACE_API_KEY with your actual API key")
    else:
        print("ℹ️  .env file already exists")

def install_dependencies():
    """Install required dependencies using uv."""
    try:
        print("📦 Installing dependencies with uv...")
        subprocess.check_call(["uv", "sync"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        print("💡 Make sure uv is installed: pip install uv")
        return False
    return True

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version {sys.version.split()[0]} is compatible")
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up Job Summary Generator...")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create .env file
    create_env_file()
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    print("=" * 50)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Get your free Hugging Face API key from https://huggingface.co/settings/tokens")
    print("2. Update the HUGGINGFACE_API_KEY in your .env file")
    print("3. Run the application with: python main.py")
    print("4. Open http://localhost:8000 in your browser")
    print("\nHappy summarizing! 🎉")

if __name__ == "__main__":
    main()
