#!/usr/bin/env python3
"""
Setup verification script for Dynamic Influence-Based Clustering.

This script verifies that all components are working correctly and 
runs a simple test to ensure the framework is properly configured.
"""

import sys
import warnings
import subprocess
from pathlib import Path

warnings.filterwarnings('ignore')

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'numpy', 'pandas', 'scikit-learn', 'xgboost',
        'shap', 'lime', 'matplotlib', 'seaborn',
        'torch', 'captum', 'joblib', 'tqdm'
    ]
    
    optional_packages = ['tslearn']
    
    missing_required = []
    missing_optional = []
    
    print("Checking required dependencies...")
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_required.append(package)
            print(f"✗ {package} (REQUIRED)")
    
    print("\nChecking optional dependencies...")
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_optional.append(package)
            print(f"⚠ {package} (OPTIONAL - time series baselines will use fallbacks)")
    
    if missing_required:
        print(f"\n❌ Missing required packages: {missing_required}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    if missing_optional:
        print(f"\n⚠️  Missing optional packages: {missing_optional}")
        print("Install with: pip install tslearn>=0.5.2")
    
    print("\n✅ All required dependencies are available!")
    return True

def check_module_imports():
    """Test importing key modules."""
    print("\nTesting module imports...")
    
    modules_to_test = [
        ('src.clustering.dynamic_kmeans', 'DynamicKMeansClustering'),
        ('src.clustering.kmeans', 'KMeansClustering'),
        ('src.clustering.timeseries_baselines', 'KShapeClustering'),
        ('src.influence.shap_influence', 'ShapInfluence'),
        ('src.influence.spearman_influence', 'SpearmanInfluence'),
        ('src.models.gradient_boost', 'GradientBoostModel'),
        ('src.utils.metrics', 'ClusteringEvaluator'),
        ('src.preprocessing.data_loader', 'DataLoader'),
    ]
    
    success = True
    for module_name, class_name in modules_to_test:
        try:
            module = __import__(module_name, fromlist=[class_name])
            cls = getattr(module, class_name)
            print(f"✓ {module_name}.{class_name}")
        except Exception as e:
            print(f"✗ {module_name}.{class_name}: {e}")
            success = False
    
    return success

def run_simple_test():
    """Run a simple test to verify the framework works."""
    print("\nRunning simple framework test...")
    
    try:
        import numpy as np
        from src.clustering.dynamic_kmeans import DynamicKMeansClustering
        from src.influence.spearman_influence import SpearmanInfluence
        
        # Generate simple test data
        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(100) * 0.1
        timestamps = np.arange(100)
        contexts = np.random.choice(['A', 'B'], 100)
        
        # Test influence extraction
        influence_extractor = SpearmanInfluence()
        influence_vectors = influence_extractor.compute_influence(X, y)
        
        # Test clustering
        clustering = DynamicKMeansClustering(n_clusters=3, alpha=1.0, beta=1.0, gamma=1.0)
        labels = clustering.fit_predict(influence_vectors, timestamps, contexts)
        
        print("✓ Simple test completed successfully!")
        print(f"  - Generated {len(influence_vectors)} influence vectors")
        print(f"  - Created {len(np.unique(labels))} clusters")
        return True
        
    except Exception as e:
        print(f"✗ Simple test failed: {e}")
        return False

def create_sample_config():
    """Create a sample configuration file if it doesn't exist."""
    config_path = Path("config_sample.py")
    
    if not config_path.exists():
        print(f"\nCreating sample configuration file: {config_path}")
        
        sample_config = '''"""
Sample configuration for Dynamic Influence-Based Clustering.
Copy this to config.py and modify as needed.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Model parameters
MODEL_PARAMS = {
    "gradient_boost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": 42
    },
    "lstm": {
        "hidden_dim": 50,
        "n_layers": 2,
        "output_dim": 1,
        "epochs": 50,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "random_state": 42
    }
}

# Clustering parameters
CLUSTERING_PARAMS = {
    "dynamic_kmeans": {
        "alpha": 1.0,    # Cohesion weight
        "beta": 1.0,     # Temporal weight  
        "gamma": 1.0,    # Contextual weight
        "J_penalty": 1.0,
        "max_iter": 300,
        "tol": 1e-4
    }
}

# Logging parameters
LOGGING_PARAMS = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
}
'''
        
        with open(config_path, 'w') as f:
            f.write(sample_config)
        
        print(f"✓ Created {config_path}")

def main():
    """Main setup verification function."""
    print("Dynamic Influence-Based Clustering - Setup Verification")
    print("=" * 60)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    if not deps_ok:
        print("\n❌ Setup incomplete. Please install missing dependencies.")
        sys.exit(1)
    
    # Check module imports
    imports_ok = check_module_imports()
    
    if not imports_ok:
        print("\n❌ Module import errors detected. Please check the installation.")
        sys.exit(1)
    
    # Run simple test
    test_ok = run_simple_test()
    
    if not test_ok:
        print("\n❌ Framework test failed. Please check the configuration.")
        sys.exit(1)
    
    # Create sample config
    create_sample_config()
    
    print("\n" + "=" * 60)
    print("✅ Setup verification completed successfully!")
    print("\nNext steps:")
    print("1. Review and customize config.py for your datasets")
    print("2. Place your data files in the data/raw/ directory")
    print("3. Run experiments using run_comprehensive_experiments.py")
    print("4. For quick start: python main.py --dataset your_dataset --influence spearman")

if __name__ == "__main__":
    main()
