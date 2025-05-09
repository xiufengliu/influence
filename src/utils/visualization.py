"""
Visualization utilities for the Dynamic Influence-Based Clustering Framework.
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def visualize_clusters(Z, labels, output_path=None, method='pca', figsize=(10, 8)):
    """
    Visualize clusters in the influence space.

    Parameters
    ----------
    Z : numpy.ndarray
        Influence space matrix.
    labels : numpy.ndarray
        Cluster labels.
    output_path : str or Path, default=None
        Path to save the visualization. If None, displays the plot.
    method : str, default='pca'
        Dimensionality reduction method. Options: 'pca', 'tsne'.
    figsize : tuple, default=(10, 8)
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object or None if an error occurs.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Visualizing clusters using {method}...")

    try:
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Apply dimensionality reduction
        if Z.shape[1] > 2:
            # Check if we have only one cluster and using t-SNE (which can cause segfaults)
            n_unique = len(np.unique(labels))
            if n_unique <= 1 and method == 'tsne':
                logger.warning("Only one cluster found. Using PCA instead of t-SNE to avoid segmentation fault.")
                method = 'pca'
                reducer = PCA(n_components=2, random_state=42)
            elif method == 'pca':
                reducer = PCA(n_components=2, random_state=42)
            elif method == 'tsne':
                # Use a safe perplexity value to avoid division by zero
                # More conservative perplexity calculation
                perplexity = min(30, max(5, len(Z) // 10))
                try:
                    reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, method='exact')
                except Exception as e:
                    logger.warning(f"Error initializing t-SNE: {e}. Using PCA instead.")
                    reducer = PCA(n_components=2, random_state=42)
            else:
                logger.warning(f"Unknown method: {method}. Using PCA instead.")
                reducer = PCA(n_components=2, random_state=42)

            try:
                Z_2d = reducer.fit_transform(Z)
            except Exception as e:
                logger.warning(f"Error in dimensionality reduction: {e}. Using PCA instead.")
                reducer = PCA(n_components=2, random_state=42)
                Z_2d = reducer.fit_transform(Z)
        else:
            Z_2d = Z

        # Plot clusters
        unique_labels = np.unique(labels)
        n_unique = len(unique_labels)

        # Use a colormap that can handle the number of clusters
        if n_unique <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, n_unique))
        else:
            colors = plt.cm.viridis(np.linspace(0, 1, n_unique))

        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(
                Z_2d[mask, 0], Z_2d[mask, 1],
                c=[colors[i]], label=f'Cluster {label}',
                alpha=0.7, edgecolors='w', linewidth=0.5
            )

        # Add legend and labels (only if not too many clusters)
        if n_unique <= 20:
            ax.legend()
        ax.set_title(f'Cluster Visualization ({method.upper()})')
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Tight layout
        plt.tight_layout()

        # Save or display
        if output_path is not None:
            # Create directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Change file extension to pdf if it's png
            output_path_str = str(output_path)
            if output_path_str.endswith('.png'):
                output_path = output_path_str.replace('.png', '.pdf')

            plt.savefig(output_path, format='pdf', bbox_inches='tight')
            logger.info(f"Cluster visualization saved to {output_path}")
        else:
            plt.show()

        return fig

    except Exception as e:
        logger.error(f"Error in cluster visualization: {e}")
        plt.close()  # Make sure to close any open figures
        return None


def visualize_transitions(transition_matrix, output_path=None, figsize=(10, 8)):
    """
    Visualize the transition matrix as a heatmap.

    Parameters
    ----------
    transition_matrix : numpy.ndarray
        Transition matrix.
    output_path : str or Path, default=None
        Path to save the visualization. If None, displays the plot.
    figsize : tuple, default=(10, 8)
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object or None if an error occurs.
    """
    logger = logging.getLogger(__name__)
    logger.info("Visualizing transition matrix...")

    try:
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Check matrix size for appropriate formatting
        n_clusters = transition_matrix.shape[0]
        if n_clusters > 10:
            # For large matrices, don't show annotations
            sns.heatmap(
                transition_matrix,
                annot=False,
                cmap='YlGnBu',
                linewidths=0.5,
                ax=ax
            )
        else:
            # For smaller matrices, show annotations
            sns.heatmap(
                transition_matrix,
                annot=True,
                cmap='YlGnBu',
                fmt='.2f',
                linewidths=0.5,
                ax=ax
            )

        # Add labels
        ax.set_title('Cluster Transition Matrix')
        ax.set_xlabel('To Cluster')
        ax.set_ylabel('From Cluster')

        # Tight layout
        plt.tight_layout()

        # Save or display
        if output_path is not None:
            # Create directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Change file extension to pdf if it's png
            output_path_str = str(output_path)
            if output_path_str.endswith('.png'):
                output_path = output_path_str.replace('.png', '.pdf')

            plt.savefig(output_path, format='pdf', bbox_inches='tight')
            logger.info(f"Transition matrix visualization saved to {output_path}")
        else:
            plt.show()

        return fig

    except Exception as e:
        logger.error(f"Error in transition matrix visualization: {e}")
        plt.close()  # Make sure to close any open figures
        return None


def visualize_influence_distribution(Z, feature_names=None, output_path=None, figsize=(12, 10)):
    """
    Visualize the distribution of influence scores for each feature.

    Parameters
    ----------
    Z : numpy.ndarray
        Influence space matrix.
    feature_names : list, default=None
        List of feature names.
    output_path : str or Path, default=None
        Path to save the visualization. If None, displays the plot.
    figsize : tuple, default=(12, 10)
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    logger = logging.getLogger(__name__)
    logger.info("Visualizing influence distribution...")

    # Set feature names if not provided
    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(Z.shape[1])]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Create boxplot
    sns.boxplot(data=Z, ax=ax)

    # Add labels
    ax.set_title('Distribution of Influence Scores')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Influence Score')
    ax.set_xticklabels(feature_names, rotation=45, ha='right')

    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)

    # Tight layout
    plt.tight_layout()

    # Save or display
    if output_path is not None:
        # Create directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Influence distribution visualization saved to {output_path}")
    else:
        plt.show()

    return fig


def visualize_temporal_evolution(clusters, timestamps, output_path=None, figsize=(12, 6)):
    """
    Visualize the temporal evolution of cluster assignments.

    Parameters
    ----------
    clusters : numpy.ndarray
        Cluster assignments.
    timestamps : numpy.ndarray or pandas.Series
        Timestamps for each instance.
    output_path : str or Path, default=None
        Path to save the visualization. If None, displays the plot.
    figsize : tuple, default=(12, 6)
        Figure size.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object or None if an error occurs.
    """
    import pandas as pd

    logger = logging.getLogger(__name__)
    logger.info("Visualizing temporal evolution of clusters...")

    try:
        # Convert timestamps to pandas Series if not already
        if not isinstance(timestamps, pd.Series):
            timestamps = pd.Series(timestamps)

        # Create DataFrame
        df = pd.DataFrame({'cluster': clusters, 'timestamp': timestamps})

        # Sort by timestamp
        df = df.sort_values('timestamp')

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Get unique clusters
        unique_clusters = df['cluster'].unique()
        n_clusters = len(unique_clusters)

        # Choose appropriate palette based on number of clusters
        if n_clusters <= 10:
            palette = 'tab10'
        else:
            palette = 'viridis'

        # Plot cluster evolution
        sns.scatterplot(
            x='timestamp',
            y='cluster',
            data=df,
            hue='cluster',
            palette=palette,
            s=50,
            alpha=0.7,
            ax=ax
        )

        # Add labels
        ax.set_title('Temporal Evolution of Cluster Assignments')
        ax.set_xlabel('Time')
        ax.set_ylabel('Cluster')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.7)

        # Remove legend title and limit legend if too many clusters
        if n_clusters <= 20:
            handles, labels = ax.get_legend_handles_labels()
            ax.legend(handles, labels, title=None)
        else:
            ax.get_legend().remove()

        # Tight layout
        plt.tight_layout()

        # Save or display
        if output_path is not None:
            # Create directory if it doesn't exist
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            # Change file extension to pdf if it's png
            output_path_str = str(output_path)
            if output_path_str.endswith('.png'):
                output_path = output_path_str.replace('.png', '.pdf')

            plt.savefig(output_path, format='pdf', bbox_inches='tight')
            logger.info(f"Temporal evolution visualization saved to {output_path}")
        else:
            plt.show()

        return fig

    except Exception as e:
        logger.error(f"Error in temporal evolution visualization: {e}")
        plt.close()  # Make sure to close any open figures
        return None
