"""PCA embeddings and cosine similarity search for fighters.

Reduces the full feature matrix to a lower-dimensional embedding space
using StandardScaler + PCA (D-09). Provides cosine similarity search
to find the most similar fighters to a given query (D-11).
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


def compute_embeddings(
    feature_matrix: np.ndarray,
    n_components: int = 8,
) -> np.ndarray:
    """Reduce a feature matrix to lower-dimensional embeddings via PCA.

    Steps:
      1. Replace NaN with 0.0 (Pitfall 4 / T-07-04).
      2. StandardScaler for zero-mean, unit-variance normalization.
      3. PCA with min(n_components, n_fighters, n_features) components.

    Args:
        feature_matrix: Array of shape (n_fighters, n_features).
        n_components: Target number of PCA dimensions. Default 8 (D-09).

    Returns:
        Embeddings array of shape (n_fighters, actual_components) where
        actual_components = min(n_components, n_fighters, n_features).
    """
    # Step 1: Impute NaN to 0.0 before scaling (RESEARCH Pitfall 4 / T-07-04)
    clean_matrix = np.nan_to_num(feature_matrix, nan=0.0)

    # Step 2: StandardScaler for zero-mean, unit-variance
    scaled = StandardScaler().fit_transform(clean_matrix)

    # Step 3: PCA with safe component count
    n_fighters, n_features = scaled.shape
    actual_components = min(n_components, n_fighters, n_features)
    pca = PCA(n_components=actual_components)
    embeddings: np.ndarray = pca.fit_transform(scaled)

    return embeddings


def find_similar_fighters(
    embeddings: np.ndarray,
    fighter_index: int,
    top_n: int = 10,
) -> list[tuple[int, float]]:
    """Find the most similar fighters by cosine similarity on embeddings.

    Args:
        embeddings: Array of shape (n_fighters, n_components) from
            compute_embeddings.
        fighter_index: Row index of the query fighter in the matrix.
        top_n: Maximum number of similar fighters to return.

    Returns:
        List of (matrix_index, similarity_score) tuples sorted by
        descending similarity. Self is excluded. Length is
        min(top_n, n_fighters - 1).
    """
    query = embeddings[fighter_index].reshape(1, -1)
    similarities = cosine_similarity(query, embeddings)[0]

    # Build (index, similarity) pairs, excluding self
    pairs = [
        (idx, float(sim))
        for idx, sim in enumerate(similarities)
        if idx != fighter_index
    ]

    # Sort by descending similarity
    pairs.sort(key=lambda x: x[1], reverse=True)

    # Limit to top_n
    return pairs[:top_n]
