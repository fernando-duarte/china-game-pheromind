#!/bin/bash
# Setup script for China Economic Data Analysis
# This script creates a virtual environment, installs dependencies, and runs the data scripts

# Exit on error
set -e

# Define colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to check if a command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Function to check Python version
check_python_version() {
    local cmd="$1"
    local version=$($cmd -c 'import sys; print(sys.version_info[0])')
    echo "$version"
}

# Determine which Python command to use
PYTHON_CMD=""
if command_exists python3; then
    PYTHON_VERSION=$(check_python_version "python3")
    if [ "$PYTHON_VERSION" -eq 3 ]; then
        PYTHON_CMD="python3"
        echo "Using python3 command."
    fi
fi

if [ -z "$PYTHON_CMD" ] && command_exists python; then
    PYTHON_VERSION=$(check_python_version "python")
    if [ "$PYTHON_VERSION" -eq 3 ]; then
        PYTHON_CMD="python"
        echo "Using python command (Python 3)."
    fi
fi

if [ -z "$PYTHON_CMD" ]; then
    echo -e "${RED}Error: Python 3 is not available. Please install Python 3 and try again.${NC}"
    echo "The script requires Python 3, but neither 'python3' nor 'python' commands are available with Python 3."
    exit 1
fi

# Parse command line arguments
DEV_MODE=false
SKIP_RUN=false
ALPHA=""
CAPITAL_OUTPUT_RATIO=""
OUTPUT_FILE=""
PROCESSOR_ARGS=""

for arg in "$@"
do
    case $arg in
        --dev)
        DEV_MODE=true
        shift
        ;;
        --skip-run)
        SKIP_RUN=true
        shift
        ;;
        -a=*|--alpha=*)
        ALPHA="${arg#*=}"
        PROCESSOR_ARGS="$PROCESSOR_ARGS -a $ALPHA"
        shift
        ;;
        -k=*|--capital-output-ratio=*)
        CAPITAL_OUTPUT_RATIO="${arg#*=}"
        PROCESSOR_ARGS="$PROCESSOR_ARGS -k $CAPITAL_OUTPUT_RATIO"
        shift
        ;;
        -o=*|--output-file=*)
        OUTPUT_FILE="${arg#*=}"
        PROCESSOR_ARGS="$PROCESSOR_ARGS -o $OUTPUT_FILE"
        shift
        ;;
        --help)
        echo "Usage: ./setup.sh [OPTIONS]"
        echo "Options:"
        echo "  --dev                           Install development dependencies"
        echo "  --skip-run                      Skip running the data scripts"
        echo "  -a=VALUE, --alpha=VALUE         Capital share parameter for TFP calculation (default: 0.33)"
        echo "  -k=VALUE, --capital-output-ratio=VALUE  Capital-to-output ratio for base year (default: 3.0)"
        echo "  -o=NAME, --output-file=NAME     Base name for output files (default: china_data_processed)"
        echo "  --help                          Show this help message"
        exit 0
        ;;
    esac
done

echo -e "${GREEN}=== China Economic Data Analysis Setup ===${NC}"
echo "This script will:"
echo "1. Create a Python virtual environment"
echo "2. Install required dependencies"
if [ "$DEV_MODE" = true ]; then
    echo "   (Including development dependencies)"
fi
if [ "$SKIP_RUN" = false ]; then
    echo "3. Run the data downloader and processor scripts"
    if [ -n "$PROCESSOR_ARGS" ]; then
        echo "   With custom processor arguments: $PROCESSOR_ARGS"
        if [ -n "$ALPHA" ]; then
            echo "   - Alpha: $ALPHA"
        fi
        if [ -n "$CAPITAL_OUTPUT_RATIO" ]; then
            echo "   - Capital-output ratio: $CAPITAL_OUTPUT_RATIO"
        fi
        if [ -n "$OUTPUT_FILE" ]; then
            echo "   - Output file base name: $OUTPUT_FILE"
        fi
    fi
fi

# Python 3 check is now handled at the beginning of the script

# Create output directory if it doesn't exist
mkdir -p output

# Check if we're already in a virtual environment
ALREADY_IN_VENV=false
if [ -n "$VIRTUAL_ENV" ]; then
    ALREADY_IN_VENV=true
    ORIGINAL_VENV="$VIRTUAL_ENV"
    echo -e "\n${YELLOW}Already in a virtual environment: ${ORIGINAL_VENV}${NC}"
    echo "Will use the current virtual environment for installation."
else
    # Create and activate virtual environment
    echo -e "\n${YELLOW}Creating virtual environment...${NC}"
    if [ -d "venv" ]; then
        echo "Removing existing virtual environment."
        rm -rf venv
    fi
    echo "Creating new virtual environment..."
    $PYTHON_CMD -m venv venv
    source venv/bin/activate
fi

# Upgrade pip and setuptools
echo -e "\n${YELLOW}Upgrading pip and setuptools...${NC}"
$PYTHON_CMD -m pip install --upgrade pip > /dev/null
# Install setuptools with distutils support
$PYTHON_CMD -m pip install 'setuptools>=67.0.0' > /dev/null

# Install dependencies
echo -e "\n${YELLOW}Installing dependencies...${NC}"
if [ "$DEV_MODE" = true ]; then
    echo "Installing development dependencies..."
    $PYTHON_CMD -m pip install -r dev-requirements.txt
else
    $PYTHON_CMD -m pip install -r requirements.txt
fi

echo -e "\n${GREEN}Setup complete!${NC}"

# Run the scripts automatically
if [ "$SKIP_RUN" = false ]; then
    echo -e "\n${YELLOW}Running China Economic Data Downloader...${NC}"
    $PYTHON_CMD china_data_downloader.py

    echo -e "\n${YELLOW}Running China Economic Data Processor...${NC}"
    if [ -n "$PROCESSOR_ARGS" ]; then
        echo "Using custom processor arguments: $PROCESSOR_ARGS"
        $PYTHON_CMD china_data_processor.py $PROCESSOR_ARGS
    else
        $PYTHON_CMD china_data_processor.py
    fi

    echo -e "\n${GREEN}All done! Output files are in the 'output' directory.${NC}"
fi

# Only deactivate if we activated it ourselves
if [ "$ALREADY_IN_VENV" = false ]; then
    # Deactivate virtual environment
    deactivate
    echo "Local virtual environment deactivated."
else
    echo "Keeping your original virtual environment active."
fi

echo -e "\n${GREEN}=== Setup Complete ===${NC}"
