#!/bin/bash
# Setup script for China Economic Data Analysis
# This script creates a virtual environment, installs dependencies, and runs the data scripts

# Exit on error
set -e

# Determine the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

# If run from root, SCRIPT_DIR will be .../china_data, PROJECT_ROOT will be ...
# If run from inside china_data, SCRIPT_DIR will be .../china_data, PROJECT_ROOT will be ...
# If run from inside root, SCRIPT_DIR will be .../china_data, PROJECT_ROOT will be ...

# Set working directory to china_data for all relative paths
cd "$SCRIPT_DIR"

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

# Function to run pytest
run_pytest() {
    echo -e "\n${YELLOW}Running tests...${NC}"
    # Temporarily add parent directory to PYTHONPATH so `import china_data` works
    (export PYTHONPATH="..:$PYTHONPATH"; $PYTHON_CMD -m pytest tests)
    # Consider adding error handling for pytest if needed:
    # if [ $? -ne 0 ]; then
    #     echo -e "${RED}Pytest failed!${NC}"
    #     cleanup_and_exit "Exiting due to test failures." 1
    # fi
}

# Function for cleanup and exit
cleanup_and_exit() {
    local message_prefix="$1"
    local exit_code="${2:-0}" # Default exit code 0

    if [ "$ALREADY_IN_VENV" = false ] && [ -n "$VIRTUAL_ENV" ]; then # Check if venv is active and script created it
        deactivate
        echo "Local virtual environment deactivated."
    elif [ "$ALREADY_IN_VENV" = true ]; then
        echo "Keeping your original virtual environment active."
    fi

    if [ -n "$message_prefix" ]; then
        echo -e "\n${GREEN}${message_prefix}${NC}"
    fi
    echo -e "\n${GREEN}=== Script Finished ===${NC}"
    exit $exit_code
}

# Parse command line arguments
DEV_MODE=false
ONLY_RUN_TESTS=false
ALPHA=""
CAPITAL_OUTPUT_RATIO=""
OUTPUT_FILE=""
PROCESSOR_ARGS=""
END_YEAR="2025"

for arg in "$@"
do
    case $arg in
        --dev)
        DEV_MODE=true
        shift
        ;;
        --test)
        ONLY_RUN_TESTS=true
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
        --end-year=*)
        END_YEAR="${arg#*=}"
        PROCESSOR_ARGS="$PROCESSOR_ARGS --end-year=$END_YEAR"
        shift
        ;;
        --help)
        echo "Usage: ./setup.sh [OPTIONS]"
        echo "Options:"
        echo "  --dev                           Install development dependencies (and run tests after main scripts)"
        echo "  --test                          Install development dependencies and run tests only"
        echo "  -a=VALUE, --alpha=VALUE         Capital share parameter for TFP calculation (default: 0.33)"
        echo "  -k=VALUE, --capital-output-ratio=VALUE  Capital-to-output ratio for base year (default: 3.0)"
        echo "  -o=NAME, --output-file=NAME     Base name for output files (default: china_data_processed)"
        echo "  --end-year=YYYY                 Last year to process (default: 2025, ignored if --test is used)"
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
if [ "$DEV_MODE" = true ] || [ "$ONLY_RUN_TESTS" = true ]; then
    echo "Installing development dependencies..."
    $PYTHON_CMD -m pip install -r dev-requirements.txt
else
    echo "Installing standard dependencies..."
    $PYTHON_CMD -m pip install -r requirements.txt
fi

echo -e "\n${GREEN}Dependency setup complete!${NC}"

if [ "$ONLY_RUN_TESTS" = true ]; then
    run_pytest
    cleanup_and_exit "Exiting after running tests only." 0
fi

# If not ONLY_RUN_TESTS, proceed with data scripts
echo -e "\n${YELLOW}Running China Economic Data Downloader...${NC}"
$PYTHON_CMD china_data_downloader.py --end-year=$END_YEAR

echo -e "\n${YELLOW}Running China Economic Data Processor...${NC}"
if [ -n "$PROCESSOR_ARGS" ]; then
    echo "Using custom processor arguments: $PROCESSOR_ARGS"
    $PYTHON_CMD china_data_processor.py $PROCESSOR_ARGS --end-year=$END_YEAR
else
    $PYTHON_CMD china_data_processor.py --end-year=$END_YEAR
fi

echo -e "\n${GREEN}Data processing complete! Output files are in the 'output' directory.${NC}"

if [ "$DEV_MODE" = true ]; then
    # ONLY_RUN_TESTS is false if we reach here, so this runs tests after data scripts for --dev
    run_pytest
fi

cleanup_and_exit "Main script execution finished." 0
