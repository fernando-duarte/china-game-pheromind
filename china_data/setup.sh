#!/bin/bash
# Setup script for China Economic Data Analysis
# This script creates a virtual environment, installs dependencies, and runs the data scripts

# Exit on error
set -e

echo "=== China Economic Data Analysis Setup ==="
echo "This script will:"
echo "1. Create a Python virtual environment"
echo "2. Install required dependencies"
echo "3. Run the data downloader and processor scripts"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed. Please install Python 3 and try again."
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p output

# Create and activate virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete!"
echo ""

# Run the scripts automatically
echo "Running China Economic Data Downloader..."
python china_data_downloader.py

echo "Running China Economic Data Processor..."
python china_data_processor.py

echo "All done! Output files are in the 'output' directory."

# Deactivate virtual environment
deactivate

echo ""
echo "=== Setup Complete ==="
