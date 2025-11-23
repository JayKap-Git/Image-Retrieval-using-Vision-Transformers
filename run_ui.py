#!/usr/bin/env python3
"""
Launch script for the Image Modeling System UI
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit UI for the Image Modeling System."""
    
    # Check if we're in the right directory
    if not os.path.exists("app.py"):
        print("Error: app.py not found. Please run this script from the project root directory.")
        sys.exit(1)
    
    # Check if required files exist
    required_files = [
        "data/classes.txt",
        "data/attributes.yaml", 
        "data/labels.csv",
        "outputs/tiny_vit_11m_224/best.pt",
        "outputs/tiny_vit_11m_224/retrieval.pt",
        "outputs/swin_tiny_patch4_window7_224/best.pt",
        "outputs/swin_tiny_patch4_window7_224/retrieval.pt"
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print("Warning: The following required files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        print("\nPlease ensure all model checkpoints and data files are present.")
        print("You may need to train the models first using the main pipeline.")
    
    print("Starting Image Modeling System UI...")
    print("The UI will open in your default web browser.")
    print("Press Ctrl+C to stop the server.")
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\nShutting down the UI server...")
    except Exception as e:
        print(f"Error launching UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
