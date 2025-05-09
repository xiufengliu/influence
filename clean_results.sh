#!/bin/bash
# Script to clean just the results directory

echo "Cleaning results directory..."

# Remove all files in the results directory but keep the directory structure
find ./data/results -type f -not -name ".gitkeep" -delete

# Create .gitkeep files to preserve directory structure
mkdir -p data/results
touch data/results/.gitkeep

echo "Results directory cleaned!"
echo "You can now re-run the program with a clean slate."
