"""Unit tests for PCA embeddings and cosine similarity search."""

from __future__ import annotations

import numpy as np
import pytest

from ufc_prediction.features.embeddings import compute_embeddings, find_similar_fighters


class TestComputeEmbeddings:
    """Tests for compute_embeddings covering shape, NaN handling, and edge cases."""

    def test_standard_shape_reduction(self) -> None:
        """(20, 15) matrix with n_components=8 produces (20, 8) output."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((20, 15))
        result = compute_embeddings(matrix, n_components=8)
        assert result.shape == (20, 8)

    def test_fewer_fighters_than_components(self) -> None:
        """(5, 15) matrix with n_components=8 produces (5, 5) output (min rule)."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((5, 15))
        result = compute_embeddings(matrix, n_components=8)
        assert result.shape == (5, 5)

    def test_fewer_features_than_components(self) -> None:
        """(3, 2) matrix with n_components=8 produces (3, 2) output (min rule)."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((3, 2))
        result = compute_embeddings(matrix, n_components=8)
        assert result.shape == (3, 2)

    def test_handles_nan_values(self) -> None:
        """NaN values in input are imputed to 0.0 before PCA (Pitfall 4)."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((10, 5))
        matrix[0, 0] = np.nan
        matrix[3, 2] = np.nan
        matrix[7, 4] = np.nan
        # Should not raise
        result = compute_embeddings(matrix, n_components=3)
        assert result.shape == (10, 3)
        assert not np.any(np.isnan(result))

    def test_output_has_zero_mean_per_component(self) -> None:
        """PCA output should have approximately zero mean per component."""
        rng = np.random.default_rng(42)
        matrix = rng.standard_normal((50, 10))
        result = compute_embeddings(matrix, n_components=5)
        column_means = result.mean(axis=0)
        np.testing.assert_allclose(column_means, 0.0, atol=1e-10)


class TestFindSimilarFighters:
    """Tests for find_similar_fighters covering ranking, exclusion, and edge cases."""

    def test_excludes_self_from_results(self) -> None:
        """The query fighter should not appear in the results."""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((10, 5))
        results = find_similar_fighters(embeddings, fighter_index=0, top_n=5)
        indices = [idx for idx, _ in results]
        assert 0 not in indices

    def test_results_sorted_by_descending_similarity(self) -> None:
        """Results should be sorted from most to least similar."""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((10, 5))
        results = find_similar_fighters(embeddings, fighter_index=0, top_n=9)
        similarities = [sim for _, sim in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_identical_vectors_have_high_similarity(self) -> None:
        """Two identical fighter vectors should have similarity close to 1.0."""
        embeddings = np.array(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        )
        results = find_similar_fighters(embeddings, fighter_index=0, top_n=2)
        # First result should be fighter 1 (identical vector)
        top_idx, top_sim = results[0]
        assert top_idx == 1
        assert top_sim == pytest.approx(1.0, abs=1e-6)

    def test_orthogonal_vectors_have_low_similarity(self) -> None:
        """Orthogonal vectors should have similarity close to 0.0."""
        embeddings = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        results = find_similar_fighters(embeddings, fighter_index=0, top_n=2)
        for _, sim in results:
            assert sim == pytest.approx(0.0, abs=1e-6)

    def test_top_n_limits_results(self) -> None:
        """Results length should be min(top_n, n_fighters - 1)."""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((10, 5))
        results = find_similar_fighters(embeddings, fighter_index=0, top_n=3)
        assert len(results) == 3

    def test_top_n_exceeds_available_fighters(self) -> None:
        """When top_n > n_fighters - 1, return all others."""
        rng = np.random.default_rng(42)
        embeddings = rng.standard_normal((5, 3))
        results = find_similar_fighters(embeddings, fighter_index=0, top_n=100)
        assert len(results) == 4  # 5 fighters - 1 (self)
