"""Shared code for comparing domains by the cosine similarity of their embedding centroids.

The embedding notebooks import from here so every model is compared the same way.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def compute_centroids(embeddings: np.ndarray, domains: pd.Series) -> pd.DataFrame:
    """Mean embedding per domain, with one row per domain.

    ``domains`` gives the domain of each row in ``embeddings``.
    """
    return pd.DataFrame(embeddings).groupby(np.asarray(domains)).mean()


def centroid_similarities(centroids: pd.DataFrame) -> pd.DataFrame:
    """Cosine similarity between every pair of domain centroids."""
    return pd.DataFrame(
        cosine_similarity(centroids), index=centroids.index, columns=centroids.index
    )


def most_and_least_similar(similarities: pd.DataFrame) -> pd.DataFrame:
    """The most and least similar other domain for each domain, with their similarities."""
    others = similarities.mask(np.eye(len(similarities), dtype=bool))
    return pd.DataFrame(
        {
            "most_similar": others.idxmax(axis=1),
            "most_similarity": others.max(axis=1),
            "least_similar": others.idxmin(axis=1),
            "least_similarity": others.min(axis=1),
        }
    )
