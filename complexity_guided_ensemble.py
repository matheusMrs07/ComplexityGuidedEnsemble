"""
Complexity-Guided Ensemble Learning.

This module implements an ensemble learning approach that:
1. Uses Complexity Guided Sampler for bag generation
2. Systematically varies complexity focus across ensemble members (μ parameter)

Fully supports multiclass classification with any number of classes.

"""

from typing import Optional, List, Tuple, Literal
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from joblib import Parallel, delayed

# Import from our multiclass complexity sampler
from complexity_sampler import (
    ComplexityGuidedSampler,
    SamplerConfig,
    ComplexityType,
    ArrayLike,
)


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

VotingStrategy = Literal["soft", "hard"]


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class EnsembleConfig:
    """Configuration for the Complexity-Guided Ensemble.

    Attributes:
        n_estimators: Number of base classifiers in the ensemble
        base_estimator: Base classifier class to use
        complexity_type: Type of complexity metric ('overlap', 'error_rate', 'neighborhood')
        sigma: Standard deviation for Gaussian weighting (fixed across ensemble)
        k_neighbors: Number of neighbors for synthetic sample generation
        voting: Voting strategy ('soft' for probability averaging, 'hard' for majority)
        random_state: Seed for reproducibility
        n_jobs: Number of parallel jobs (-1 for all cores)
        cv_folds: Cross-validation folds for evaluation
        verbose: Print progress information
        hardness_function: Custom hardness function from pyhard
    """

    n_estimators: int = 10
    base_estimator: ClassifierMixin = DecisionTreeClassifier
    complexity_type: ComplexityType = "overlap"
    sigma: float = 0.2
    k_neighbors: int = 5
    voting: VotingStrategy = "soft"
    random_state: Optional[int] = None
    n_jobs: int = -1
    cv_folds: int = 3
    verbose: int = 0
    hardness_function: Optional[str] = None


# ============================================================================
# MAIN ENSEMBLE CLASS
# ============================================================================


class ComplexityGuidedEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier using complexity-guided resampling - Multiclass Version.

    This ensemble creates diverse base classifiers by:
    1. Systematically varying the complexity focus (μ parameter) across members
    2. Using IHWR (Instance Hardness Weighted Resampling) for bag generation

    Supports any number of classes (binary and multiclass).

    Parameters
    ----------
    config : EnsembleConfig, optional
        Configuration object for the ensemble
    **kwargs : dict
        Configuration parameters (used if config is None)

    Attributes
    ----------
    estimators_ : list
        List of fitted base estimators
    mu_values_ : array
        Mu values used for each estimator
    classes_ : array
        Unique class labels
    n_classes_ : int
        Number of classes

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(
    ...     n_samples=1000, n_classes=3, n_clusters_per_class=1,
    ...     weights=[0.7, 0.2, 0.1], random_state=42
    ... )
    >>>
    >>> ensemble = ComplexityGuidedEnsemble(
    ...     n_estimators=10,
    ...     random_state=42
    ... )
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)
    """

    def __init__(
        self,
        config: Optional[EnsembleConfig] = None,
        store_subsets: bool = True,
        **kwargs,
    ):
        self.config = config or EnsembleConfig(**kwargs)
        self.store_subsets = store_subsets

        self._validate_config()

        # Initialize attributes
        self.estimators_: List[ClassifierMixin] = []
        self.mu_values_: Optional[ArrayLike] = None
        self.subsets_: List[Tuple[ArrayLike, ArrayLike]] = []
        self.classes_: Optional[ArrayLike] = None
        self.n_classes_: Optional[int] = None
        self.sampler_: Optional[ComplexityGuidedSampler] = None

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")

        if not (0 < self.config.sigma <= 1):
            raise ValueError("sigma must be in (0, 1]")

    def _generate_mu_values(self) -> ArrayLike:
        """Generate mu values systematically distributed from 0 to 1."""
        if self.config.n_estimators == 1:
            return np.array([0.5])
        return np.linspace(0, 1, self.config.n_estimators)

    def _initialize_sampler(self, X: ArrayLike, y: ArrayLike) -> None:
        """Initialize the complexity-guided sampler."""
        sampler_config = SamplerConfig(
            complex_type=self.config.complexity_type,
            hardness_function=self.config.hardness_function,
            random_state=self.config.random_state,
            cv_folds=self.config.cv_folds,
        )

        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, index=np.arange(X.shape[0]))
        else:
            X_df = X

        if not isinstance(y, pd.Series):
            y_series = pd.Series(y)
        else:
            y_series = y

        data = pd.concat([X_df, y_series.rename("target")], axis=1)

        self.sampler_ = ComplexityGuidedSampler(
            config=sampler_config, data=data, target_col="target"
        )

    def _create_subset(
        self, X: ArrayLike, y: ArrayLike, mu: float
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Create subset using IHWR."""
        X_resampled, y_resampled = self.sampler_.fit_resample(
            X, y, mu=mu, sigma=self.config.sigma, k_neighbors=self.config.k_neighbors
        )

        return X_resampled, y_resampled

    def _fit_single_estimator(
        self, X: ArrayLike, y: ArrayLike, mu: float, estimator_idx: int
    ) -> ClassifierMixin:
        """Fit a single base estimator on a complexity-guided subset."""
        if self.config.verbose > 0:
            print(
                f"  Training estimator {estimator_idx + 1}/{self.config.n_estimators} "
                f"(μ={mu:.3f})"
            )

        X_subset, y_subset = self._create_subset(X, y, mu)

        if self.store_subsets:
            self.subsets_.append((X_subset.copy(), y_subset.copy()))

        base_estimator = clone(
            self.config.base_estimator(random_state=self.config.random_state)
        )
        base_estimator.fit(X_subset, y_subset)

        if self.config.verbose > 0:
            unique, counts = np.unique(y_subset.astype(int), return_counts=True)
            dist_str = ", ".join([f"C{c}: {n}" for c, n in zip(unique, counts)])
            print(f"    Subset size: {len(X_subset)}, Class distribution: {dist_str}")

        return base_estimator

    def fit(self, X: ArrayLike, y: ArrayLike) -> "ComplexityGuidedEnsemble":
        """Fit the ensemble on training data."""
        if self.config.verbose > 0:
            print(f"\n{'='*70}")
            print(f"Training Complexity-Guided Ensemble (Multiclass)")
            print(f"{'='*70}")
            print(f"Configuration:")
            print(f"  - Base estimator: {self.config.base_estimator.__name__}")
            print(f"  - Number of estimators: {self.config.n_estimators}")
            print(f"  - Complexity type: {self.config.complexity_type}")
            print(f"  - Sigma: {self.config.sigma}")
            print(f"  - K neighbors: {self.config.k_neighbors}")
            print(f"{'='*70}\n")

        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        if self.config.verbose > 0:
            print(f"Number of classes: {self.n_classes_}")
            unique, counts = np.unique(y, return_counts=True)
            for cls, count in zip(unique, counts):
                print(f"  Class {cls}: {count} samples")
            print()

        self._initialize_sampler(X, y)
        self.mu_values_ = self._generate_mu_values()

        if self.config.verbose > 0:
            print(f"Generated μ values: {self.mu_values_}")
            print()

        if self.config.n_jobs == 1:
            self.estimators_ = []
            for idx, mu in enumerate(self.mu_values_):
                estimator = self._fit_single_estimator(X, y, mu, idx)
                self.estimators_.append(estimator)
        else:
            self.estimators_ = Parallel(n_jobs=self.config.n_jobs, verbose=0)(
                delayed(self._fit_single_estimator)(X, y, mu, idx)
                for idx, mu in enumerate(self.mu_values_)
            )

        if self.config.verbose > 0:
            print(f"\n{'='*70}")
            print(f"Ensemble training completed!")
            print(f"{'='*70}\n")

        return self

    def predict(self, X: ArrayLike) -> ArrayLike:
        """Predict class labels for samples."""
        if not self.estimators_:
            raise ValueError("Ensemble must be fitted before prediction")

        X = np.asarray(X)

        if self.config.voting == "soft":
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]
        else:
            # Hard voting: majority vote
            predictions = np.array([est.predict(X) for est in self.estimators_])

            # For each sample, find the most common prediction
            final_predictions = []
            for i in range(X.shape[0]):
                sample_preds = predictions[:, i]
                # Count votes for each class
                unique, counts = np.unique(sample_preds, return_counts=True)
                winner = unique[np.argmax(counts)]
                final_predictions.append(winner)

            return np.array(final_predictions)

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """Predict class probabilities for samples."""
        if not self.estimators_:
            raise ValueError("Ensemble must be fitted before prediction")

        X = np.asarray(X)

        if not hasattr(self.estimators_[0], "predict_proba"):
            raise AttributeError(
                f"{self.config.base_estimator.__name__} does not support predict_proba. "
                "Use voting='hard' or a classifier with probability support."
            )

        # Collect probabilities from all estimators
        all_probas = []

        for est in self.estimators_:
            proba = est.predict_proba(X)

            # Ensure probabilities align with self.classes_
            # Some estimators may not have seen all classes during training
            est_classes = est.classes_
            aligned_proba = np.zeros((X.shape[0], self.n_classes_))

            for i, cls in enumerate(est_classes):
                cls_idx = np.where(self.classes_ == cls)[0]
                if len(cls_idx) > 0:
                    aligned_proba[:, cls_idx[0]] = proba[:, i]

            all_probas.append(aligned_proba)

        # Average probabilities across all estimators
        avg_proba = np.mean(all_probas, axis=0)

        # Normalize to ensure probabilities sum to 1
        row_sums = avg_proba.sum(axis=1, keepdims=True)
        avg_proba = avg_proba / (row_sums + 1e-10)

        return avg_proba

    def get_ensemble_diversity(self) -> float:
        """Calculate diversity among ensemble members."""
        if not self.store_subsets:
            raise ValueError("store_subsets must be True to calculate diversity")

        n_estimators = len(self.estimators_)
        if n_estimators < 2:
            return 0.0

        diversities = []
        for i in range(n_estimators):
            for j in range(i + 1, n_estimators):
                X_i, _ = self.subsets_[i]
                X_j, _ = self.subsets_[j]

                X_i_hashes = {hash(tuple(row)) for row in X_i}
                X_j_hashes = {hash(tuple(row)) for row in X_j}
                overlap = len(X_i_hashes & X_j_hashes)
                union = len(X_i_hashes | X_j_hashes)

                diversity = 1 - (overlap / (union + 1e-10))
                diversities.append(diversity)

        return np.mean(diversities)

    def get_complexity_distribution(self) -> dict:
        """Get statistics about complexity distribution across ensemble."""
        if self.mu_values_ is None:
            raise ValueError("Ensemble must be fitted first")

        return {
            "mu_values": self.mu_values_,
            "mu_mean": self.mu_values_.mean(),
            "mu_std": self.mu_values_.std(),
            "mu_range": (self.mu_values_.min(), self.mu_values_.max()),
            "n_estimators": len(self.mu_values_),
            "n_classes": self.n_classes_,
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def compare_ensemble_configurations(
    X: ArrayLike,
    y: ArrayLike,
    n_estimators: int = 10,
    configurations: Optional[List[dict]] = None,
    cv_folds: int = 3,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """Compare different ensemble configurations."""
    if configurations is None:
        configurations = [
            {
                "name": "Default Configuration",
                "complexity_type": "overlap",
                "sigma": 0.2,
            },
            {
                "name": "Error Rate Complexity",
                "complexity_type": "error_rate",
                "sigma": 0.2,
            },
            {
                "name": "Neighborhood Complexity",
                "complexity_type": "neighborhood",
                "sigma": 0.2,
            },
            {
                "name": "High Sigma",
                "complexity_type": "overlap",
                "sigma": 0.5,
            },
        ]

    results = []

    for config in configurations:
        name = config.pop("name")

        ensemble = ComplexityGuidedEnsemble(
            n_estimators=n_estimators, random_state=random_state, **config
        )

        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        scores = cross_val_score(
            ensemble, X, y, cv=cv, scoring="f1_weighted", n_jobs=-1
        )

        results.append(
            {
                "Configuration": name,
                "F1 Score": scores.mean(),
                "F1 Std": scores.std(),
                **config,
            }
        )

    return pd.DataFrame(results)
