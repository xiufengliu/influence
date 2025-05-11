#!/bin/bash
# Script to clean just the results directory, preserving completed experiment sets.

echo "Cleaning results directory selectively..."

RESULTS_BASE_DIR="./data/results/test_tnnls_sequential"

if [ ! -d "$RESULTS_BASE_DIR" ]; then
    echo "Results base directory $RESULTS_BASE_DIR does not exist. Nothing to clean."
    # Ensure the results directory and .gitkeep file exist for future runs
    mkdir -p data/results
    touch data/results/.gitkeep
    exit 0
fi

# Iterate over each influence method's subdirectory
for method_dir in "$RESULTS_BASE_DIR"/*/; do
    if [ -d "$method_dir" ]; then # Check if it's a directory
        method_name=$(basename "$method_dir")
        echo "Checking directory: $method_dir"
        if [ -f "${method_dir}all_results.json" ]; then
            echo "  Found all_results.json in $method_dir. Skipping cleanup for $method_name."
        else
            echo "  No all_results.json in $method_dir. Removing directory for $method_name."
            rm -rf "$method_dir"
        fi
    fi
done

# Ensure the top-level results directory and .gitkeep file exist
# This is important if all subdirectories were removed or if it's the first run
mkdir -p data/results
touch data/results/.gitkeep 
# Also ensure the specific test directory exists if it was cleared or not present
mkdir -p "$RESULTS_BASE_DIR"


echo "Results directory cleaned selectively!"
echo "You can now re-run the program. Completed experiments (if any) have been preserved."
