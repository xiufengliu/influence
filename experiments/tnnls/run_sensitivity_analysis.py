
import os
import sys
import argparse
import logging
import json
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import config
from src.utils.logger import setup_logger
from src.preprocessing.data_loader import DataLoader
from src.models.gradient_boost import GradientBoostModel
from src.influence.shap_influence import ShapInfluence
from src.clustering.dynamic_kmeans import DynamicKMeansClustering
from src.utils.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score


def parse_args():
    parser = argparse.ArgumentParser(description="Run sensitivity analysis for Dynamic Influence-Based Clustering.")
    parser.add_argument("--dataset", type=str, default="energy_data", help="Dataset to use.")
    parser.add_argument("--output_dir", type=str, default="data/results/sensitivity_analysis", help="Directory to save results.")
    return parser.parse_args()

def run_sensitivity_analysis(dataset_name, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("sensitivity_analysis", "INFO", output_dir / "sensitivity_analysis.log")

    logger.info(f"Loading dataset: {dataset_name}")
    data_loader = DataLoader(dataset_name)
    X, y, timestamps, contexts, entity_ids = data_loader.load_data()

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, timestamps_train, timestamps_test, contexts_train, contexts_test, entity_ids_train, entity_ids_test = train_test_split(
        X, y, timestamps, contexts, entity_ids, test_size=0.3, random_state=42, shuffle=False
    )

    logger.info("Training predictive model...")
    model = GradientBoostModel()
    model.fit(X_train, y_train)

    logger.info("Generating influence scores...")
    influence_generator = ShapInfluence()
    Z = influence_generator.generate_influence(model, X_test)

    alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    beta_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    gamma_values = [0.1, 0.5, 1.0, 2.0, 5.0]

    results = []

    for alpha in alpha_values:
        for beta in beta_values:
            for gamma in gamma_values:
                logger.info(f"Running clustering with alpha={alpha}, beta={beta}, gamma={gamma}")
                try:
                    clustering = DynamicKMeansClustering(n_clusters=3, alpha=alpha, beta=beta, gamma=gamma)
                    clustering.fit(Z, timestamps_test, contexts_test, entity_ids_test)
                    labels = clustering.labels_

                    silhouette = silhouette_score(Z, labels)
                    davies_bouldin = davies_bouldin_score(Z, labels)
                    calinski_harabasz = calinski_harabasz_score(Z, labels)

                    results.append({
                        "alpha": alpha,
                        "beta": beta,
                        "gamma": gamma,
                        "silhouette": silhouette,
                        "davies_bouldin": davies_bouldin,
                        "calinski_harabasz": calinski_harabasz
                    })
                except Exception as e:
                    logger.error(f"Error with alpha={alpha}, beta={beta}, gamma={gamma}: {e}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "sensitivity_results.csv", index=False)

    logger.info("Visualizing results...")
    for metric in ["silhouette", "davies_bouldin", "calinski_harabasz"]:
        plt.figure(figsize=(12, 10))
        pivot = results_df.pivot_table(index="beta", columns="gamma", values=metric)
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
        plt.title(f"{metric.replace('_', ' ').title()} vs. Beta and Gamma (Alpha=1.0)")
        plt.savefig(output_dir / f"{metric}_beta_gamma_heatmap.png")
        plt.close()

if __name__ == "__main__":
    args = parse_args()
    run_sensitivity_analysis(args.dataset, args.output_dir)
