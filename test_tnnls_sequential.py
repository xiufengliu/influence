"""
Test script to run TNNLS experiments sequentially, one influence method at a time.
This approach minimizes memory usage by running only one influence method at a time.
"""

import os
import sys
from pathlib import Path
from experiments.tnnls.run_experiments import run_experiments

# Define experiment parameters
datasets = ["energy_data", "steel_industry"]
all_experiments = [
    "clustering_quality",
    "temporal_analysis",
    "contextual_coherence",
    "ablation_studies",
    "case_studies",
    "computational_analysis"
]
all_influence_methods = ["spearman", "lime", "shap"]  # Order from least to most memory-intensive
clustering_algorithms = ["kmeans", "hierarchical"]
n_clusters_list = [3, 5]
random_seeds = [42]
output_dir = Path("data/results/test_tnnls_sequential")
# Use conservative parallel processing
safe_n_jobs = 1  # Use a single job to minimize memory usage
verbose = True

# Allow command-line override of influence method
if len(sys.argv) > 1:
    requested_method = sys.argv[1].lower()
    if requested_method in all_influence_methods:
        all_influence_methods = [requested_method]
        print(f"Running with only {requested_method} influence method as requested")
    else:
        print(f"Warning: Unknown influence method '{requested_method}'. Using all methods.")

# Run experiments one by one, one influence method at a time
print("Starting TNNLS experiments sequentially...")
print(f"Using {safe_n_jobs} parallel job to minimize memory usage")

# Track overall results
all_results = {}

# Run each influence method separately
for influence_method in all_influence_methods:
    print(f"\n\n{'='*80}")
    print(f"RUNNING WITH INFLUENCE METHOD: {influence_method.upper()}")
    print(f"{'='*80}\n")
    
    # Run each experiment type separately for this influence method
    for experiment in all_experiments:
        try:
            print(f"\n{'='*60}")
            print(f"Running {experiment} experiment with {influence_method} influence...")
            print(f"{'='*60}\n")
            
            # Run just this experiment with just this influence method
            result = run_experiments(
                datasets=datasets,
                experiments=[experiment],
                influence_methods=[influence_method],
                clustering_algorithms=clustering_algorithms,
                n_clusters_list=n_clusters_list,
                random_seeds=random_seeds,
                output_dir=output_dir / influence_method,
                n_jobs=safe_n_jobs,
                verbose=verbose
            )
            
            # Store results
            if influence_method not in all_results:
                all_results[influence_method] = {}
            all_results[influence_method][experiment] = result
            
            print(f"{experiment} experiment with {influence_method} completed successfully!")
            
        except Exception as e:
            import traceback
            print(f"{experiment} experiment with {influence_method} failed: {e}")
            traceback.print_exc()
            print(f"Continuing with next experiment...")
    
    # Force garbage collection after each influence method
    import gc
    gc.collect()

# Report overall results
print("\n\n" + "="*80)
print("TNNLS experiments summary:")
print("="*80)

for influence_method in all_influence_methods:
    print(f"\nResults for {influence_method.upper()} influence method:")
    if influence_method in all_results:
        for experiment in all_experiments:
            if experiment in all_results[influence_method]:
                print(f"  ✅ {experiment}: Completed successfully")
            else:
                print(f"  ❌ {experiment}: Failed")
    else:
        print(f"  ❌ All experiments failed")

print(f"\nResults saved to {output_dir}")
