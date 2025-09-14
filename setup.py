import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description}:")
        print(f"   {e.stderr}")
        return False

def setup_project():
    """Set up the complete project environment"""
    print("🚀 Setting up NutriMenu AI Project Environment")
    print("=" * 50)
    
    # Check if Python is available
    try:
        result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
        print(f"🐍 Python version: {result.stdout.strip()}")
    except Exception as e:
        print(f"❌ Python not found: {e}")
        return False
    
    # Create virtual environment
    venv_path = "venv"
    if not os.path.exists(venv_path):
        if not run_command(f"{sys.executable} -m venv {venv_path}", "Creating virtual environment"):
            return False
    else:
        print("✅ Virtual environment already exists!")
    
    # Determine activation script path
    if os.name == 'nt':  # Windows
        activate_script = os.path.join(venv_path, "Scripts", "activate")
        pip_path = os.path.join(venv_path, "Scripts", "pip")
        python_path = os.path.join(venv_path, "Scripts", "python")
    else:  # Unix/Linux/Mac
        activate_script = os.path.join(venv_path, "bin", "activate")
        pip_path = os.path.join(venv_path, "bin", "pip")
        python_path = os.path.join(venv_path, "bin", "python")
    
    # Install requirements
    if not run_command(f"{pip_path} install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command(f"{pip_path} install -r requirements.txt", "Installing requirements"):
        return False
    
    # Download NLTK data
    print("📥 Downloading NLTK data...")
    nltk_commands = [
        f"{python_path} -c \"import nltk; nltk.download('punkt')\"",
        f"{python_path} -c \"import nltk; nltk.download('stopwords')\"",
        f"{python_path} -c \"import nltk; nltk.download('averaged_perceptron_tagger')\"",
        f"{python_path} -c \"import nltk; nltk.download('wordnet')\""
    ]
    
    for cmd in nltk_commands:
        subprocess.run(cmd, shell=True, capture_output=True)
    
    print("✅ NLTK data downloaded!")
    
    # Create run scripts
    create_run_scripts(venv_path)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Activate the virtual environment:")
    if os.name == 'nt':
        print("      .\\venv\\Scripts\\activate")
    else:
        print("      source venv/bin/activate")
    print("   2. Run the demo:")
    print("      python demo.py")
    print("   3. Or use the run scripts:")
    if os.name == 'nt':
        print("      .\\run_demo.bat")
    else:
        print("      ./run_demo.sh")
    
    return True

def create_run_scripts(venv_path):
    """Create convenient run scripts"""
    
    # Windows batch script
    if os.name == 'nt':
        bat_content = f"""@echo off
echo 🍽️ Starting NutriMenu AI Demo...
call {venv_path}\\Scripts\\activate
python demo.py
pause
"""
        with open("run_demo.bat", "w") as f:
            f.write(bat_content)
        print("✅ Created run_demo.bat")
    
    # Unix shell script
    else:
        sh_content = f"""#!/bin/bash
echo "🍽️ Starting NutriMenu AI Demo..."
source {venv_path}/bin/activate
python demo.py
"""
        with open("run_demo.sh", "w") as f:
            f.write(sh_content)
        os.chmod("run_demo.sh", 0o755)  # Make executable
        print("✅ Created run_demo.sh")

if __name__ == "__main__":
    setup_project()