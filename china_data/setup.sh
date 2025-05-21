#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

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

$PYTHON_CMD china_data_downloader.py --end-year=$END_YEAR
$PYTHON_CMD china_data_processor.py $PROCESSOR_ARGS --end-year=$END_YEAR

if $DEV; then run_tests; fi

if ! $ALREADY_IN_VENV; then deactivate; fi
