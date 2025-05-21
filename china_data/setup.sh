#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Determine if we're in the project root or in the china_data directory
PROJECT_ROOT="$SCRIPT_DIR"
if [[ "$(basename "$SCRIPT_DIR")" == "china_data" ]]; then
    # We're in the china_data directory
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
fi

command_exists(){ command -v "$1" &>/dev/null; }
PYTHON_CMD=""
if command_exists python3; then PYTHON_CMD=python3; fi
if [ -z "$PYTHON_CMD" ] && command_exists python; then PYTHON_CMD=python; fi
[ -z "$PYTHON_CMD" ] && { echo "Python 3 required"; exit 1; }

DEV=false
TEST_ONLY=false
PROCESSOR_ARGS=""
END_YEAR=2025
for arg in "$@"; do
    case $arg in
        --dev) DEV=true;;
        --test) TEST_ONLY=true;;
        -a=*|--alpha=*) PROCESSOR_ARGS+=" -a ${arg#*=}";;
        -k=*|--capital-output-ratio=*) PROCESSOR_ARGS+=" -k ${arg#*=}";;
        -o=*|--output-file=*) PROCESSOR_ARGS+=" -o ${arg#*=}";;
        --end-year=*) END_YEAR="${arg#*=}";;
        --help) echo "Usage: ./setup.sh [--dev|--test] [-a=VAL] [-k=VAL] [-o=NAME] [--end-year=YYYY]"; exit 0;;
    esac
done

mkdir -p output
ALREADY_IN_VENV=false
if [ -z "$VIRTUAL_ENV" ]; then
    $PYTHON_CMD -m venv venv
    source venv/bin/activate
else
    ALREADY_IN_VENV=true
fi

$PYTHON_CMD -m pip install --upgrade pip >/dev/null
$PYTHON_CMD -m pip install 'setuptools>=67.0.0' >/dev/null
if $DEV || $TEST_ONLY; then
    $PYTHON_CMD -m pip install -r dev-requirements.txt
else
    $PYTHON_CMD -m pip install -r requirements.txt
fi

run_tests(){
    (export PYTHONPATH="..:$PYTHONPATH"; $PYTHON_CMD -m pytest tests)
}

if $TEST_ONLY; then
    run_tests
    exit 0
fi

# Set up Python path to include the project root
cd "$PROJECT_ROOT"
export PYTHONPATH="$PYTHONPATH:$(pwd)"
cd "$SCRIPT_DIR"

# Determine how to run the Python modules based on our location
if [[ "$(basename "$SCRIPT_DIR")" == "china_data" ]]; then
    # We're in the china_data directory, so we need to use the china_data module prefix
    $PYTHON_CMD -m china_data.china_data_downloader --end-year=$END_YEAR
    $PYTHON_CMD -m china_data.china_data_processor $PROCESSOR_ARGS --end-year=$END_YEAR
else
    # We're already at the project root, so we can run the modules directly
    $PYTHON_CMD -m china_data_downloader --end-year=$END_YEAR
    $PYTHON_CMD -m china_data_processor $PROCESSOR_ARGS --end-year=$END_YEAR
fi

if $DEV; then run_tests; fi

if ! $ALREADY_IN_VENV; then deactivate; fi
