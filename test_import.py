"""
Test script to verify imports.
"""

try:
    from experiments.tnnls.run_experiments import run_experiments
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {e}")
