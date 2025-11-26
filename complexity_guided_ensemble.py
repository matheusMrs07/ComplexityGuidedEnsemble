"""
Complexity-Guided Ensemble Learning - Multiclass Version.

This module implements an advanced ensemble learning approach that:
1. Uses Instance Hardness Weighted Resampling (IHWR) for bag generation
2. Systematically varies complexity focus across ensemble members (μ parameter)
3. Incorporates active learning for intelligent instance selection
4. Applies fitness functions for subset optimization
5. Supports evolutionary strategies for ensemble refinement

Fully supports multiclass classification with any number of classes.

Author: Adapted for multiclass support
License: MIT
"""

from typing import Optional, List, Tuple, Callable, Union, Literal, Dict
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import label_binarize
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
FitnessMetric = Literal["f1", "auc", "accuracy", "diversity"]


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
        cv_folds: Cross-validation folds for fitness evaluation
        use_active_learning: Enable active learning for instance selection
        use_fitness_optimization: Enable fitness-based subset optimization
        fitness_metric: Metric to optimize ('f1', 'auc', 'accuracy', 'diversity')
        max_fitness_iterations: Maximum iterations for fitness optimization
        mutation_rate: Probability of instance mutation in evolutionary optimization
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
    use_active_learning: bool = True
    use_fitness_optimization: bool = False
    fitness_metric: FitnessMetric = "f1"
    max_fitness_iterations: int = 10
    mutation_rate: float = 0.1
    verbose: int = 0
    hardness_function: Optional[str] = None


# ============================================================================
# ACTIVE LEARNING STRATEGIES
# ============================================================================


class ActiveLearningStrategy(ABC):
    """Base class for active learning strategies."""

    @abstractmethod
    def select_informative_instances(
        self,
        X: ArrayLike,
        y: ArrayLike,
        complexities: ArrayLike,
        n_select: int,
        **kwargs,
    ) -> ArrayLike:
        """Select most informative instances."""
        pass


class UncertaintyBasedSelection(ActiveLearningStrategy):
    """Select instances based on prediction uncertainty.

    Works for both binary and multiclass classification.
    """

    def __init__(self, classifier: Optional[ClassifierMixin] = None):
        self.classifier = classifier

    def select_informative_instances(
        self,
        X: ArrayLike,
        y: ArrayLike,
        complexities: ArrayLike,
        n_select: int,
        **kwargs,
    ) -> ArrayLike:
        """Select instances with highest prediction uncertainty."""
        if self.classifier is None:
            return np.argsort(complexities)[-n_select:]

        # Get prediction probabilities
        probas = self.classifier.predict_proba(X)

        # Calculate entropy for uncertainty (works for any number of classes)
        # Entropy = -sum(p * log(p))
        probas_safe = np.clip(probas, 1e-10, 1.0)
        uncertainty = -np.sum(probas_safe * np.log(probas_safe), axis=1)

        # Select most uncertain instances
        return np.argsort(uncertainty)[-n_select:]


class ComplexityBasedSelection(ActiveLearningStrategy):
    """Select instances based on complexity scores."""

    def __init__(self, target_mu: float = 0.5):
        self.target_mu = target_mu

    def select_informative_instances(
        self,
        X: ArrayLike,
        y: ArrayLike,
        complexities: ArrayLike,
        n_select: int,
        **kwargs,
    ) -> ArrayLike:
        """Select instances closest to target complexity level."""
        distances = np.abs(complexities - self.target_mu)
        return np.argsort(distances)[:n_select]


class HybridSelection(ActiveLearningStrategy):
    """Hybrid strategy combining uncertainty and complexity."""

    def __init__(
        self,
        classifier: Optional[ClassifierMixin] = None,
        target_mu: float = 0.5,
        uncertainty_weight: float = 0.5,
    ):
        self.classifier = classifier
        self.target_mu = target_mu
        self.uncertainty_weight = uncertainty_weight

    def select_informative_instances(
        self,
        X: ArrayLike,
        y: ArrayLike,
        complexities: ArrayLike,
        n_select: int,
        **kwargs,
    ) -> ArrayLike:
        """Select instances using hybrid scoring."""
        # Uncertainty score
        if self.classifier is not None:
            try:
                probas = self.classifier.predict_proba(X)
                probas_safe = np.clip(probas, 1e-10, 1.0)
                uncertainty = -np.sum(probas_safe * np.log(probas_safe), axis=1)
                # Normalize
                if uncertainty.max() > 0:
                    uncertainty = uncertainty / uncertainty.max()
            except Exception:
                uncertainty = complexities
        else:
            uncertainty = complexities

        # Complexity relevance score
        complexity_relevance = 1 - np.abs(complexities - self.target_mu)

        # Combined score
        combined_score = (
            self.uncertainty_weight * uncertainty
            + (1 - self.uncertainty_weight) * complexity_relevance
        )

        return np.argsort(combined_score)[-n_select:]


# ============================================================================
# FITNESS FUNCTIONS
# ============================================================================


class FitnessFunction(ABC):
    """Base class for fitness evaluation."""

    @abstractmethod
    def evaluate(
        self, X: ArrayLike, y: ArrayLike, classifier: ClassifierMixin, **kwargs
    ) -> float:
        pass


class PerformanceBasedFitness(FitnessFunction):
    """Fitness based on classification performance - supports multiclass."""

    def __init__(
        self,
        metric: FitnessMetric = "f1",
        cv_folds: int = 3,
        random_state: Optional[int] = None,
    ):
        self.metric = metric
        self.cv_folds = cv_folds
        self.random_state = random_state

    def evaluate(
        self, X: ArrayLike, y: ArrayLike, classifier: ClassifierMixin, **kwargs
    ) -> float:
        """Evaluate using cross-validation performance."""
        n_classes = len(np.unique(y))

        if len(X) < self.cv_folds:
            # Not enough data for CV
            classifier_clone = clone(classifier)
            try:
                classifier_clone.fit(X, y)
                y_pred = classifier_clone.predict(X)

                if self.metric == "f1":
                    return f1_score(y, y_pred, average="weighted")
                elif self.metric == "accuracy":
                    return accuracy_score(y, y_pred)
                else:
                    return accuracy_score(y, y_pred)
            except Exception:
                return 0.0

        # Use stratified cross-validation for multiclass
        try:
            cv = StratifiedKFold(
                n_splits=min(self.cv_folds, len(X)),
                shuffle=True,
                random_state=self.random_state,
            )

            if self.metric == "f1":
                scoring = "f1_weighted"
            elif self.metric == "auc" and n_classes == 2:
                scoring = "roc_auc"
            elif self.metric == "auc" and n_classes > 2:
                scoring = "roc_auc_ovr_weighted"
            else:
                scoring = "accuracy"

            scores = cross_val_score(classifier, X, y, cv=cv, scoring=scoring, n_jobs=1)
            return scores.mean()
        except Exception:
            return 0.0


class DiversityBasedFitness(FitnessFunction):
    """Fitness based on diversity from other ensemble members."""

    def __init__(
        self, reference_subsets: Optional[List[Tuple[ArrayLike, ArrayLike]]] = None
    ):
        self.reference_subsets = reference_subsets or []

    def evaluate(
        self, X: ArrayLike, y: ArrayLike, classifier: ClassifierMixin, **kwargs
    ) -> float:
        """Evaluate diversity compared to reference subsets."""
        if not self.reference_subsets:
            return 1.0

        diversities = []

        for ref_X, ref_y in self.reference_subsets:
            if hasattr(X, "index") and hasattr(ref_X, "index"):
                overlap = len(set(X.index) & set(ref_X.index))
                union = len(set(X.index) | set(ref_X.index))
            else:
                X_hashes = {hash(tuple(row)) for row in X}
                ref_hashes = {hash(tuple(row)) for row in ref_X}
                overlap = len(X_hashes & ref_hashes)
                union = len(X_hashes | ref_hashes)

            diversity = 1 - (overlap / (union + 1e-10))
            diversities.append(diversity)

        return np.mean(diversities)


class HybridFitness(FitnessFunction):
    """Combine performance and diversity fitness."""

    def __init__(
        self,
        performance_fitness: FitnessFunction,
        diversity_fitness: FitnessFunction,
        performance_weight: float = 0.7,
    ):
        self.performance_fitness = performance_fitness
        self.diversity_fitness = diversity_fitness
        self.performance_weight = performance_weight

    def evaluate(
        self, X: ArrayLike, y: ArrayLike, classifier: ClassifierMixin, **kwargs
    ) -> float:
        perf_score = self.performance_fitness.evaluate(X, y, classifier, **kwargs)
        div_score = self.diversity_fitness.evaluate(X, y, classifier, **kwargs)

        return (
            self.performance_weight * perf_score
            + (1 - self.performance_weight) * div_score
        )


# ============================================================================
# EVOLUTIONARY OPTIMIZATION
# ============================================================================


class SubsetOptimizer:
    """Optimize subsets using evolutionary strategies."""

    def __init__(
        self,
        fitness_function: FitnessFunction,
        max_iterations: int = 5,
        mutation_rate: float = 0.1,
        random_state: Optional[int] = None,
    ):
        self.fitness_function = fitness_function
        self.max_iterations = max_iterations
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def optimize(
        self,
        X_subset: ArrayLike,
        y_subset: ArrayLike,
        X_pool: ArrayLike,
        y_pool: ArrayLike,
        classifier: ClassifierMixin,
        verbose: int = 0,
    ) -> Tuple[ArrayLike, ArrayLike, float]:
        """Optimize subset using evolutionary strategy."""
        current_X, current_y = X_subset.copy(), y_subset.copy()
        current_fitness = self.fitness_function.evaluate(
            current_X, current_y, classifier
        )

        if verbose > 0:
            print(f"    Initial fitness: {current_fitness:.4f}")

        for iteration in range(self.max_iterations):
            n_mutations = max(1, int(len(current_X) * self.mutation_rate))
            replace_indices = self.rng.choice(
                len(current_X), size=n_mutations, replace=False
            )
            new_indices = self.rng.choice(len(X_pool), size=n_mutations, replace=False)

            mutated_X = current_X.copy()
            mutated_y = current_y.copy()

            for old_idx, new_idx in zip(replace_indices, new_indices):
                mutated_X[old_idx] = X_pool[new_idx]
                mutated_y[old_idx] = y_pool[new_idx]

            mutated_fitness = self.fitness_function.evaluate(
                mutated_X, mutated_y, classifier
            )

            if mutated_fitness > current_fitness:
                current_X, current_y = mutated_X, mutated_y
                current_fitness = mutated_fitness

                if verbose > 0:
                    print(
                        f"    Iteration {iteration + 1}: fitness improved to {current_fitness:.4f}"
                    )
            else:
                if verbose > 0:
                    print(f"    Iteration {iteration + 1}: no improvement")

        return current_X, current_y, current_fitness


# ============================================================================
# MAIN ENSEMBLE CLASS - MULTICLASS VERSION
# ============================================================================


class ComplexityGuidedEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier using complexity-guided resampling - Multiclass Version.

    This ensemble creates diverse base classifiers by:
    1. Systematically varying the complexity focus (μ parameter) across members
    2. Using IHWR (Instance Hardness Weighted Resampling) for bag generation
    3. Optionally applying active learning for intelligent instance selection
    4. Optionally optimizing subsets with fitness functions

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
    ...     use_active_learning=True,
    ...     random_state=42
    ... )
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)
    """

    def __init__(
        self,
        config: Optional[EnsembleConfig] = None,
        store_subsets: bool = False,
        **kwargs,
    ):
        self.config = config or EnsembleConfig(**kwargs)
        self.store_subsets = store_subsets

        self._validate_config()

        # Initialize attributes
        self.estimators_: List[ClassifierMixin] = []
        self.mu_values_: Optional[ArrayLike] = None
        self.subsets_: List[Tuple[ArrayLike, ArrayLike]] = []
        self.fitness_scores_: Optional[ArrayLike] = None
        self.classes_: Optional[ArrayLike] = None
        self.n_classes_: Optional[int] = None
        self.sampler_: Optional[ComplexityGuidedSampler] = None
        self.active_learner_: Optional[ActiveLearningStrategy] = None
        self.fitness_function_: Optional[FitnessFunction] = None
        self.subset_optimizer_: Optional[SubsetOptimizer] = None

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")

        if not (0 < self.config.sigma <= 1):
            raise ValueError("sigma must be in (0, 1]")

        if not (0 <= self.config.mutation_rate <= 1):
            raise ValueError("mutation_rate must be in [0, 1]")

    def _generate_mu_values(self) -> ArrayLike:
        """Generate mu values systematically distributed from 0 to 1."""
        if self.config.n_estimators == 1:
            return np.array([0.5])
        return np.linspace(0, 1, self.config.n_estimators)

    def _initialize_components(self, X: ArrayLike, y: ArrayLike) -> None:
        """Initialize sampler, active learner, and fitness components."""
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

        if self.config.use_active_learning:
            self.active_learner_ = HybridSelection(
                target_mu=0.5,
                uncertainty_weight=0.5,
            )

        if self.config.use_fitness_optimization:
            perf_fitness = PerformanceBasedFitness(
                metric=self.config.fitness_metric,
                cv_folds=self.config.cv_folds,
                random_state=self.config.random_state,
            )
            div_fitness = DiversityBasedFitness(reference_subsets=self.subsets_)
            self.fitness_function_ = HybridFitness(
                performance_fitness=perf_fitness,
                diversity_fitness=div_fitness,
                performance_weight=0.7,
            )

            self.subset_optimizer_ = SubsetOptimizer(
                fitness_function=self.fitness_function_,
                max_iterations=self.config.max_fitness_iterations,
                mutation_rate=self.config.mutation_rate,
                random_state=self.config.random_state,
            )

    def _create_subset_with_active_learning(
        self, X: ArrayLike, y: ArrayLike, mu: float, estimator_idx: int
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Create subset using IHWR and active learning."""
        X_resampled, y_resampled = self.sampler_.fit_resample(
            X, y, mu=mu, sigma=self.config.sigma, k_neighbors=self.config.k_neighbors
        )

        if not self.config.use_active_learning or self.active_learner_ is None:
            return X_resampled, y_resampled

        if estimator_idx > 0 and len(self.estimators_) > 0:
            preliminary_clf = self.estimators_[-1]
        else:
            preliminary_clf = clone(
                self.config.base_estimator(random_state=self.config.random_state)
            )
            preliminary_clf.fit(X_resampled, y_resampled)

        if isinstance(
            self.active_learner_, (ComplexityBasedSelection, HybridSelection)
        ):
            self.active_learner_.target_mu = mu

        if isinstance(
            self.active_learner_, (UncertaintyBasedSelection, HybridSelection)
        ):
            self.active_learner_.classifier = preliminary_clf

        complexities = self.sampler_.complexities

        n_select = len(X_resampled)
        informative_indices = self.active_learner_.select_informative_instances(
            X, y, complexities, n_select
        )

        X_refined = X[informative_indices]
        y_refined = y[informative_indices]

        return X_refined, y_refined

    def _optimize_subset(
        self,
        X_subset: ArrayLike,
        y_subset: ArrayLike,
        X_pool: ArrayLike,
        y_pool: ArrayLike,
        base_estimator: ClassifierMixin,
    ) -> Tuple[ArrayLike, ArrayLike, float]:
        """Optimize subset using fitness function."""
        if not self.config.use_fitness_optimization or self.subset_optimizer_ is None:
            fitness = (
                self.fitness_function_.evaluate(X_subset, y_subset, base_estimator)
                if self.fitness_function_ is not None
                else 0.0
            )
            return X_subset, y_subset, fitness

        return self.subset_optimizer_.optimize(
            X_subset,
            y_subset,
            X_pool,
            y_pool,
            base_estimator,
            verbose=self.config.verbose,
        )

    def _fit_single_estimator(
        self, X: ArrayLike, y: ArrayLike, mu: float, estimator_idx: int
    ) -> Tuple[ClassifierMixin, float]:
        """Fit a single base estimator on a complexity-guided subset."""
        if self.config.verbose > 0:
            print(
                f"  Training estimator {estimator_idx + 1}/{self.config.n_estimators} "
                f"(μ={mu:.3f})"
            )

        X_subset, y_subset = self._create_subset_with_active_learning(
            X, y, mu, estimator_idx
        )

        base_estimator = clone(
            self.config.base_estimator(random_state=self.config.random_state)
        )

        X_final, y_final, fitness = self._optimize_subset(
            X_subset, y_subset, X, y, base_estimator
        )

        if self.store_subsets:
            self.subsets_.append((X_final.copy(), y_final.copy()))

        base_estimator.fit(X_final, y_final)

        if self.config.verbose > 0:
            print(f"    Fitness: {fitness:.4f}")
            unique, counts = np.unique(y_final.astype(int), return_counts=True)
            dist_str = ", ".join([f"C{c}: {n}" for c, n in zip(unique, counts)])
            print(f"    Subset size: {len(X_final)}, Class distribution: {dist_str}")

        return base_estimator, fitness

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
            print(f"  - Active learning: {self.config.use_active_learning}")
            print(f"  - Fitness optimization: {self.config.use_fitness_optimization}")
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

        self._initialize_components(X, y)
        self.mu_values_ = self._generate_mu_values()

        if self.config.verbose > 0:
            print(f"Generated μ values: {self.mu_values_}")
            print()

        if self.config.n_jobs == 1:
            results = []
            for idx, mu in enumerate(self.mu_values_):
                estimator, fitness = self._fit_single_estimator(X, y, mu, idx)
                results.append((estimator, fitness))
        else:
            results = Parallel(n_jobs=self.config.n_jobs, verbose=0)(
                delayed(self._fit_single_estimator)(X, y, mu, idx)
                for idx, mu in enumerate(self.mu_values_)
            )

        self.estimators_ = [est for est, _ in results]
        self.fitness_scores_ = np.array([fitness for _, fitness in results])

        if self.config.verbose > 0:
            print(f"\n{'='*70}")
            print(f"Ensemble training completed!")
            print(f"  - Average fitness: {self.fitness_scores_.mean():.4f}")
            print(f"  - Fitness std: {self.fitness_scores_.std():.4f}")
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
        if not self.estimators_ or not self.subsets_:
            raise ValueError("Ensemble must be fitted with store_subsets=True")

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
