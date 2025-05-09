#!/bin/bash
# Script to clean the project before pushing to GitHub

echo "Cleaning project for GitHub..."

# Remove Python cache files
echo "Removing Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name "*.pyd" -delete
find . -type f -name ".coverage" -delete
find . -type d -name ".pytest_cache" -exec rm -rf {} +
find . -type d -name ".coverage" -exec rm -rf {} +

# Remove Jupyter notebook checkpoints
echo "Removing Jupyter notebook checkpoints..."
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +

# Remove IDE files
echo "Removing IDE files..."
find . -type d -name ".idea" -exec rm -rf {} +
find . -type d -name ".vscode" -exec rm -rf {} +
find . -type f -name "*.swp" -delete
find . -type f -name "*.swo" -delete

# Remove OS specific files
echo "Removing OS specific files..."
find . -type f -name ".DS_Store" -delete
find . -type f -name "Thumbs.db" -delete

# Remove log files
echo "Removing log files..."
find . -type f -name "*.log" -delete

# Clean data/results directory (but keep the directory structure)
echo "Cleaning results directory..."
find ./data/results -type f -not -name ".gitkeep" -delete

# Create a separate script for cleaning just the results
if [ ! -f clean_results.sh ]; then
    echo "Creating clean_results.sh script..."
    cat > clean_results.sh << 'EOF'
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
EOF
    chmod +x clean_results.sh
    echo "Created clean_results.sh script for cleaning just the results directory"
fi

# Create .gitkeep files to preserve directory structure
echo "Creating .gitkeep files to preserve directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/results
touch data/raw/.gitkeep
touch data/processed/.gitkeep
touch data/results/.gitkeep

# Convert PNG files to PDF (already done in the code)
echo "PNG to PDF conversion already handled in the code."

# Check for large files (>10MB)
echo "Checking for large files (>10MB)..."
find . -type f -size +10M | while read file; do
    echo "Warning: Large file found: $file"
    echo "Consider adding this file to .gitignore or using Git LFS"
done

# Update .gitignore if needed
if ! grep -q "data/results/" .gitignore; then
    echo "Updating .gitignore..."
    echo "data/results/" >> .gitignore
fi

if ! grep -q "*.pdf" .gitignore; then
    echo "Adding *.pdf to .gitignore..."
    echo "*.pdf" >> .gitignore
fi

# Create a clean requirements.txt if it doesn't exist
if [ ! -f requirements.txt ]; then
    echo "Creating requirements.txt..."
    pip freeze > requirements.txt
fi

# Remove redundant test files
echo "Removing redundant test files..."
rm -f test_contextual_coherence.py
rm -f test_run_experiment.py
rm -f test_run_experiments_simple.py
rm -f test_import.py

echo "Keeping essential test files:"
echo "- tests/test_framework.py (unit tests)"
echo "- test_tnnls_real_datasets.py (main integration test)"
echo "- test_real_datasets.py (data loading test)"

echo "Project cleaning completed!"
echo "You can now safely push your code to GitHub."
