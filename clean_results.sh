#!/bin/bash
# Script to clean just the results directory

echo "Cleaning results directory..."

# Remove all files and subfolders in the results directory but keep .gitkeep
# The -mindepth 1 ensures we don't try to delete data/results itself.
# -path './data/results/.gitkeep' -prune ensures .gitkeep is not deleted if it exists at the top level of data/results.
# -o -exec rm -rf {} + removes everything else.
find ./data/results -mindepth 1 -path './data/results/.gitkeep' -prune -o -exec rm -rf {} +

# Ensure the results directory and .gitkeep file exist
mkdir -p data/results
touch data/results/.gitkeep

echo "Results directory cleaned!"
echo "You can now re-run the program with a clean slate."
