"""
Case studies for TNNLS submission.

This module implements case studies to demonstrate the practical utility of the
Dynamic Influence-Based Clustering Framework in real-world scenarios.
"""

import os
import sys
import logging
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

import config
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.preprocessor import Preprocessor
from src.models.gradient_boost import GradientBoostModel
from src.influence.shap_influence import ShapInfluence
from src.clustering.kmeans import KMeansClustering
from src.temporal.transition_matrix import TransitionMatrix
from src.temporal.anomaly_detection import AnomalyDetection
from src.utils.logger import setup_logger
from src.utils.visualization import visualize_clusters, visualize_transitions, visualize_temporal_evolution


def run_pattern_discovery_case_study(dataset_name, influence_method, output_dir):
    """
    Run pattern discovery case study.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to use.
    influence_method : str
        Influence method to use (shap, lime, or spearman).
    output_dir : str or Path
        Directory to save results.

    Returns
    -------
    dict
        Dictionary containing case study results.
    """
    # Set up logging
    logger = logging.getLogger(f"case_study_pattern_{dataset_name}")

    # Import clustering algorithms here to avoid circular imports
    from src.clustering.hierarchical import HierarchicalClustering

    try:
        # Load and preprocess data
        data_loader = DataLoader(dataset_name=dataset_name)
        # Get preprocessed data directly from the data loader
        X, y, t, c = data_loader.load_data(preprocess=True)

        # Train predictive model
        model_params = config.MODEL_PARAMS["gradient_boost"].copy()
        model = GradientBoostModel(**model_params)
        model.fit(X, y)

        # Generate influence space using the specified method
        if influence_method == "shap":
            influence_params = config.INFLUENCE_PARAMS["shap"].copy()
            influence_generator = ShapInfluence(**influence_params)
        elif influence_method == "lime":
            from src.influence.lime_influence import LimeInfluence
            influence_params = config.INFLUENCE_PARAMS["lime"].copy()
            influence_generator = LimeInfluence(**influence_params)
        else:  # spearman
            from src.influence.spearman_influence import SpearmanInfluence
            influence_params = config.INFLUENCE_PARAMS["spearman"].copy()
            influence_generator = SpearmanInfluence(**influence_params)

        logger.info(f"Generating influence space using {influence_method}...")
        Z = influence_generator.generate_influence(model, X)
        logger.info(f"Influence space generated successfully")

        # Perform clustering with different numbers of clusters
        n_clusters_list = [3, 5, 7]
        clustering_results = {}

        for n_clusters in n_clusters_list:
            # Create clustering algorithm
            clustering = KMeansClustering(n_clusters=n_clusters)

            # Fit and predict
            clusters = clustering.fit_predict(Z)

            # Store results
            clustering_results[n_clusters] = {
                "clusters": clusters,
                "cluster_centers": clustering.get_cluster_centers()
            }

        # Analyze cluster characteristics
        cluster_analysis = {}

        for n_clusters, result in clustering_results.items():
            clusters = result["clusters"]
            cluster_centers = result["cluster_centers"]

            # Analyze each cluster
            cluster_stats = []

            for i in range(n_clusters):
                # Get instances in this cluster
                cluster_mask = clusters == i
                cluster_X = X[cluster_mask]
                cluster_y = y[cluster_mask]
                cluster_Z = Z[cluster_mask]

                # Calculate statistics
                cluster_stat = {
                    "cluster_id": i,
                    "size": np.sum(cluster_mask),
                    "percentage": 100 * np.sum(cluster_mask) / len(clusters),
                    "mean_target": np.mean(cluster_y),
                    "std_target": np.std(cluster_y),
                    "mean_features": np.mean(cluster_X, axis=0).tolist(),
                    "mean_influence": np.mean(cluster_Z, axis=0).tolist(),
                    "top_influence_features": np.argsort(-np.abs(np.mean(cluster_Z, axis=0)))[:5].tolist()
                }

                cluster_stats.append(cluster_stat)

            cluster_analysis[n_clusters] = cluster_stats

        # Save visualizations
        if output_dir:
            vis_dir = Path(output_dir) / "pattern_discovery"
            vis_dir.mkdir(exist_ok=True)

            # Visualize clusters for each n_clusters
            for n_clusters, result in clustering_results.items():
                clusters = result["clusters"]

                # Visualize clusters with PCA
                visualize_clusters(
                    Z, clusters,
                    output_path=vis_dir / f"{dataset_name}_clusters_{n_clusters}_pca.png",
                    method='pca'
                )

                # Visualize clusters with t-SNE
                visualize_clusters(
                    Z, clusters,
                    output_path=vis_dir / f"{dataset_name}_clusters_{n_clusters}_tsne.png",
                    method='tsne'
                )

                # Create feature importance heatmap for each cluster
                plt.figure(figsize=(15, 10))
                cluster_influences = np.zeros((n_clusters, Z.shape[1]))

                for i in range(n_clusters):
                    cluster_mask = clusters == i
                    cluster_influences[i] = np.mean(Z[cluster_mask], axis=0)

                sns.heatmap(
                    cluster_influences,
                    cmap="coolwarm",
                    center=0,
                    annot=False,
                    fmt=".2f"
                )
                plt.title(f"Cluster Influence Profiles ({n_clusters} clusters)")
                plt.xlabel("Feature")
                plt.ylabel("Cluster")
                plt.tight_layout()
                plt.savefig(vis_dir / f"{dataset_name}_cluster_influences_{n_clusters}.pdf", format='pdf')
                plt.close()

        # Return results
        return {
            "dataset": dataset_name,
            "clustering_results": clustering_results,
            "cluster_analysis": cluster_analysis
        }

    except Exception as e:
        logger.error(f"Error in case study: {e}", exc_info=True)

        # Return error result
        return {
            "dataset": dataset_name,
            "error": str(e)
        }


def run_transition_analysis_case_study(dataset_name, influence_method, output_dir):
    """
    Run transition analysis case study.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to use.
    influence_method : str
        Influence method to use (shap, lime, or spearman).
    output_dir : str or Path
        Directory to save results.

    Returns
    -------
    dict
        Dictionary containing case study results.
    """
    # Set up logging
    logger = logging.getLogger(f"case_study_transition_{dataset_name}")

    try:
        # Load and preprocess data
        data_loader = DataLoader(dataset_name=dataset_name)
        # Get preprocessed data directly from the data loader
        X, y, t, c = data_loader.load_data(preprocess=True)

        # Train predictive model
        model_params = config.MODEL_PARAMS["gradient_boost"].copy()
        model = GradientBoostModel(**model_params)
        model.fit(X, y)

        # Generate influence space using the specified method
        if influence_method == "shap":
            influence_params = config.INFLUENCE_PARAMS["shap"].copy()
            influence_generator = ShapInfluence(**influence_params)
        elif influence_method == "lime":
            from src.influence.lime_influence import LimeInfluence
            influence_params = config.INFLUENCE_PARAMS["lime"].copy()
            influence_generator = LimeInfluence(**influence_params)
        else:  # spearman
            from src.influence.spearman_influence import SpearmanInfluence
            influence_params = config.INFLUENCE_PARAMS["spearman"].copy()
            influence_generator = SpearmanInfluence(**influence_params)

        logger.info(f"Generating influence space using {influence_method}...")
        Z = influence_generator.generate_influence(model, X)
        logger.info(f"Influence space generated successfully")

        # Perform clustering
        n_clusters = 5  # Use 5 clusters for detailed transition analysis
        clustering = KMeansClustering(n_clusters=n_clusters)
        clusters = clustering.fit_predict(Z)

        # Compute transition matrix
        transition_matrix = TransitionMatrix()
        P = transition_matrix.compute(clusters, t)

        # Compute stationary distribution
        stationary_distribution = transition_matrix.stationary_distribution

        # Compute cluster stability
        cluster_stability = transition_matrix.get_cluster_stability()

        # Compute expected time to each cluster
        expected_times = {}
        for i in range(n_clusters):
            expected_times[i] = transition_matrix.get_expected_time_to_cluster(i).tolist()

        # Compute n-step transition matrices
        n_steps = [2, 3, 5, 10]
        n_step_transitions = {}

        for n in n_steps:
            n_step_transitions[n] = transition_matrix.get_n_step_transition(n).tolist()

        # Save visualizations
        if output_dir:
            vis_dir = Path(output_dir) / "transition_analysis"
            vis_dir.mkdir(exist_ok=True)

            # Visualize transition matrix
            visualize_transitions(
                P,
                output_path=vis_dir / f"{dataset_name}_transition_matrix.pdf"
            )

            # Visualize temporal evolution
            visualize_temporal_evolution(
                clusters, t,
                output_path=vis_dir / f"{dataset_name}_temporal_evolution.pdf"
            )

            # Visualize stationary distribution
            plt.figure(figsize=(10, 6))
            plt.bar(range(n_clusters), stationary_distribution)
            plt.title("Stationary Distribution")
            plt.xlabel("Cluster")
            plt.ylabel("Probability")
            plt.xticks(range(n_clusters))
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(vis_dir / f"{dataset_name}_stationary_distribution.pdf", format='pdf')
            plt.close()

            # Visualize cluster stability
            plt.figure(figsize=(10, 6))
            plt.bar(range(n_clusters), cluster_stability)
            plt.title("Cluster Stability (Self-Transition Probability)")
            plt.xlabel("Cluster")
            plt.ylabel("Stability")
            plt.xticks(range(n_clusters))
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(vis_dir / f"{dataset_name}_cluster_stability.pdf", format='pdf')
            plt.close()

            # Create network diagram of transitions
            plt.figure(figsize=(10, 8))

            # Create positions for nodes in a circle
            pos = {}
            for i in range(n_clusters):
                angle = 2 * np.pi * i / n_clusters
                pos[i] = (np.cos(angle), np.sin(angle))

            # Draw nodes
            for i in range(n_clusters):
                plt.plot(pos[i][0], pos[i][1], 'o', markersize=20 * stationary_distribution[i] * 5,
                         color=plt.cm.tab10(i % 10))
                plt.text(pos[i][0], pos[i][1], str(i),
                         horizontalalignment='center', verticalalignment='center')

            # Draw edges
            for i in range(n_clusters):
                for j in range(n_clusters):
                    if P[i, j] > 0.1:  # Only draw significant transitions
                        plt.arrow(pos[i][0], pos[i][1],
                                  0.8 * (pos[j][0] - pos[i][0]),
                                  0.8 * (pos[j][1] - pos[i][1]),
                                  head_width=0.05, head_length=0.1,
                                  fc=plt.cm.Blues(P[i, j]), ec=plt.cm.Blues(P[i, j]),
                                  alpha=P[i, j], linewidth=P[i, j] * 3)

            plt.title(f"Transition Network for {dataset_name}")
            plt.axis('equal')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(vis_dir / f"{dataset_name}_transition_network.pdf", format='pdf')
            plt.close()

        # Return results
        return {
            "dataset": dataset_name,
            "n_clusters": n_clusters,
            "transition_matrix": P.tolist(),
            "stationary_distribution": stationary_distribution.tolist(),
            "cluster_stability": cluster_stability.tolist(),
            "expected_times": expected_times,
            "n_step_transitions": n_step_transitions
        }

    except Exception as e:
        logger.error(f"Error in case study: {e}", exc_info=True)

        # Return error result
        return {
            "dataset": dataset_name,
            "error": str(e)
        }


def run_anomaly_detection_case_study(dataset_name, influence_method, output_dir):
    """
    Run anomaly detection case study.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to use.
    influence_method : str
        Influence method to use (shap, lime, or spearman).
    output_dir : str or Path
        Directory to save results.

    Returns
    -------
    dict
        Dictionary containing case study results.
    """
    # Set up logging
    logger = logging.getLogger(f"case_study_anomaly_{dataset_name}")

    try:
        # Load and preprocess data
        data_loader = DataLoader(dataset_name=dataset_name)
        # Get preprocessed data directly from the data loader
        X, y, t, c = data_loader.load_data(preprocess=True)

        # Train predictive model
        model_params = config.MODEL_PARAMS["gradient_boost"].copy()
        model = GradientBoostModel(**model_params)
        model.fit(X, y)

        # Generate influence space using the specified method
        if influence_method == "shap":
            influence_params = config.INFLUENCE_PARAMS["shap"].copy()
            influence_generator = ShapInfluence(**influence_params)
        elif influence_method == "lime":
            from src.influence.lime_influence import LimeInfluence
            influence_params = config.INFLUENCE_PARAMS["lime"].copy()
            influence_generator = LimeInfluence(**influence_params)
        else:  # spearman
            from src.influence.spearman_influence import SpearmanInfluence
            influence_params = config.INFLUENCE_PARAMS["spearman"].copy()
            influence_generator = SpearmanInfluence(**influence_params)

        logger.info(f"Generating influence space using {influence_method}...")
        Z = influence_generator.generate_influence(model, X)
        logger.info(f"Influence space generated successfully")

        # Perform clustering
        n_clusters = 5
        clustering = KMeansClustering(n_clusters=n_clusters)
        clusters = clustering.fit_predict(Z)

        # Compute transition matrix
        transition_matrix = TransitionMatrix()
        P = transition_matrix.compute(clusters, t)

        # Detect anomalies with different thresholds
        thresholds = [0.05, 0.1, 0.2]
        anomaly_results = {}

        for threshold in thresholds:
            # Detect anomalies
            anomaly_detector = AnomalyDetection()
            anomalies = anomaly_detector.detect(clusters, P, t, threshold=threshold)

            # Store results
            if len(anomalies) > 0:
                anomaly_results[threshold] = anomalies.to_dict(orient="records")
            else:
                anomaly_results[threshold] = []

        # Detect contextual anomalies
        contextual_anomalies = anomaly_detector.detect_contextual_anomalies(
            clusters, c, P, t, threshold=0.1
        )

        if len(contextual_anomalies) > 0:
            contextual_anomaly_results = contextual_anomalies.to_dict(orient="records")
        else:
            contextual_anomaly_results = []

        # Save visualizations
        if output_dir:
            vis_dir = Path(output_dir) / "anomaly_detection"
            vis_dir.mkdir(exist_ok=True)

            # Visualize anomalies for each threshold
            for threshold in thresholds:
                anomalies = pd.DataFrame(anomaly_results[threshold])

                if len(anomalies) > 0:
                    plt.figure(figsize=(15, 8))

                    # Plot cluster evolution
                    plt.scatter(
                        pd.to_datetime(t),
                        clusters,
                        c=clusters,
                        cmap='tab10',
                        alpha=0.7,
                        s=50,
                        label='Clusters'
                    )

                    # Highlight anomalies
                    anomaly_times = pd.to_datetime(anomalies['timestamp'])
                    anomaly_clusters = anomalies['from_cluster']
                    plt.scatter(
                        anomaly_times,
                        anomaly_clusters,
                        c='red',
                        marker='x',
                        s=100,
                        label='Anomalies'
                    )

                    plt.title(f'Temporal Evolution with Anomalies (threshold={threshold})')
                    plt.xlabel('Time')
                    plt.ylabel('Cluster')
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    plt.savefig(vis_dir / f"{dataset_name}_anomalies_{threshold}.png", dpi=300)
                    plt.close()

            # Visualize contextual anomalies
            if len(contextual_anomaly_results) > 0:
                contextual_anomalies = pd.DataFrame(contextual_anomaly_results)

                plt.figure(figsize=(15, 8))

                # Plot cluster evolution
                plt.scatter(
                    pd.to_datetime(t),
                    clusters,
                    c=clusters,
                    cmap='tab10',
                    alpha=0.7,
                    s=50,
                    label='Clusters'
                )

                # Highlight contextual anomalies
                anomaly_times = pd.to_datetime(contextual_anomalies['timestamp'])
                anomaly_clusters = contextual_anomalies['from_cluster']
                plt.scatter(
                    anomaly_times,
                    anomaly_clusters,
                    c='purple',
                    marker='*',
                    s=150,
                    label='Contextual Anomalies'
                )

                plt.title(f'Temporal Evolution with Contextual Anomalies')
                plt.xlabel('Time')
                plt.ylabel('Cluster')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.tight_layout()
                plt.savefig(vis_dir / f"{dataset_name}_contextual_anomalies.pdf", format='pdf')
                plt.close()

        # Return results
        return {
            "dataset": dataset_name,
            "n_clusters": n_clusters,
            "anomaly_results": anomaly_results,
            "contextual_anomaly_results": contextual_anomaly_results
        }

    except Exception as e:
        logger.error(f"Error in case study: {e}", exc_info=True)

        # Return error result
        return {
            "dataset": dataset_name,
            "error": str(e)
        }


def run_case_studies(datasets, influence_methods, clustering_algorithms,
                    n_clusters_list, random_seeds, output_dir, n_jobs=-1, verbose=False):
    """
    Run comprehensive case studies.

    Parameters
    ----------
    datasets : list
        List of datasets to use.
    influence_methods : list
        List of influence methods to evaluate.
    clustering_algorithms : list
        List of clustering algorithms to evaluate.
    n_clusters_list : list
        List of number of clusters to evaluate.
    random_seeds : list
        List of random seeds for reproducibility and statistical analysis.
    output_dir : str or Path
        Directory to save results.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 for all available cores).
    verbose : bool, default=False
        Enable verbose output.

    Returns
    -------
    dict
        Dictionary containing case study results.
    """
    # Set up logging
    logger = setup_logger("case_studies", "INFO")
    logger.info("Starting case studies...")

    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run pattern discovery case studies
    logger.info("Running pattern discovery case studies...")
    pattern_results = {}

    # Use only the first influence method to avoid memory issues
    influence_method = influence_methods[0]
    logger.info(f"Using influence method: {influence_method}")

    for dataset in datasets:
        logger.info(f"Running pattern discovery case study for {dataset} with {influence_method}...")
        pattern_result = run_pattern_discovery_case_study(dataset, influence_method, output_dir)
        pattern_results[dataset] = pattern_result

    # Run transition analysis case studies
    logger.info("Running transition analysis case studies...")
    transition_results = {}

    for dataset in datasets:
        logger.info(f"Running transition analysis case study for {dataset} with {influence_method}...")
        transition_result = run_transition_analysis_case_study(dataset, influence_method, output_dir)
        transition_results[dataset] = transition_result

    # Run anomaly detection case studies
    logger.info("Running anomaly detection case studies...")
    anomaly_results = {}

    for dataset in datasets:
        logger.info(f"Running anomaly detection case study for {dataset} with {influence_method}...")
        anomaly_result = run_anomaly_detection_case_study(dataset, influence_method, output_dir)
        anomaly_results[dataset] = anomaly_result

    # Create summary report
    logger.info("Creating summary report...")
    create_summary_report(pattern_results, transition_results, anomaly_results, output_dir)

    logger.info(f"Case studies completed. Results saved to {output_dir}")

    return {
        "pattern_discovery": pattern_results,
        "transition_analysis": transition_results,
        "anomaly_detection": anomaly_results
    }


def create_summary_report(pattern_results, transition_results, anomaly_results, output_dir):
    """
    Create summary report of case studies.

    Parameters
    ----------
    pattern_results : dict
        Pattern discovery case study results.
    transition_results : dict
        Transition analysis case study results.
    anomaly_results : dict
        Anomaly detection case study results.
    output_dir : str or Path
        Directory to save report.
    """
    # Create report directory
    report_dir = Path(output_dir) / "summary_report"
    report_dir.mkdir(exist_ok=True)

    # Create HTML report
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Dynamic Influence-Based Clustering Framework: Case Studies</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #2c3e50; }
            h2 { color: #3498db; }
            h3 { color: #2980b9; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            tr:nth-child(even) { background-color: #f9f9f9; }
            .section { margin-bottom: 30px; }
            .figure { margin: 20px 0; text-align: center; }
            .figure img { max-width: 100%; }
            .caption { font-style: italic; margin-top: 5px; }
        </style>
    </head>
    <body>
        <h1>Dynamic Influence-Based Clustering Framework: Case Studies</h1>

        <div class="section">
            <h2>1. Pattern Discovery</h2>
    """

    # Add pattern discovery results
    for dataset, result in pattern_results.items():
        if "error" in result:
            html_content += f"<h3>{dataset}: Error</h3><p>{result['error']}</p>"
            continue

        html_content += f"<h3>{dataset}</h3>"

        # Add cluster visualizations
        html_content += "<div class='figure'>"
        html_content += f"<img src='../pattern_discovery/{dataset}_clusters_5_pca.png' alt='Clusters PCA'>"
        html_content += "<p class='caption'>Clusters in PCA-reduced influence space</p>"
        html_content += "</div>"

        html_content += "<div class='figure'>"
        html_content += f"<img src='../pattern_discovery/{dataset}_cluster_influences_5.png' alt='Cluster Influences'>"
        html_content += "<p class='caption'>Cluster influence profiles</p>"
        html_content += "</div>"

    html_content += """
        </div>

        <div class="section">
            <h2>2. Transition Analysis</h2>
    """

    # Add transition analysis results
    for dataset, result in transition_results.items():
        if "error" in result:
            html_content += f"<h3>{dataset}: Error</h3><p>{result['error']}</p>"
            continue

        html_content += f"<h3>{dataset}</h3>"

        # Add transition visualizations
        html_content += "<div class='figure'>"
        html_content += f"<img src='../transition_analysis/{dataset}_transition_matrix.png' alt='Transition Matrix'>"
        html_content += "<p class='caption'>Transition matrix between clusters</p>"
        html_content += "</div>"

        html_content += "<div class='figure'>"
        html_content += f"<img src='../transition_analysis/{dataset}_transition_network.png' alt='Transition Network'>"
        html_content += "<p class='caption'>Transition network visualization</p>"
        html_content += "</div>"

        html_content += "<div class='figure'>"
        html_content += f"<img src='../transition_analysis/{dataset}_temporal_evolution.png' alt='Temporal Evolution'>"
        html_content += "<p class='caption'>Temporal evolution of cluster assignments</p>"
        html_content += "</div>"

    html_content += """
        </div>

        <div class="section">
            <h2>3. Anomaly Detection</h2>
    """

    # Add anomaly detection results
    for dataset, result in anomaly_results.items():
        if "error" in result:
            html_content += f"<h3>{dataset}: Error</h3><p>{result['error']}</p>"
            continue

        html_content += f"<h3>{dataset}</h3>"

        # Add anomaly visualizations
        html_content += "<div class='figure'>"
        html_content += f"<img src='../anomaly_detection/{dataset}_anomalies_0.1.png' alt='Anomalies'>"
        html_content += "<p class='caption'>Detected anomalies (threshold=0.1)</p>"
        html_content += "</div>"

        # Add contextual anomaly visualizations if available
        if os.path.exists(output_dir / "anomaly_detection" / f"{dataset}_contextual_anomalies.png"):
            html_content += "<div class='figure'>"
            html_content += f"<img src='../anomaly_detection/{dataset}_contextual_anomalies.png' alt='Contextual Anomalies'>"
            html_content += "<p class='caption'>Detected contextual anomalies</p>"
            html_content += "</div>"

    html_content += """
        </div>
    </body>
    </html>
    """

    # Save HTML report
    with open(report_dir / "case_studies_report.html", "w") as f:
        f.write(html_content)