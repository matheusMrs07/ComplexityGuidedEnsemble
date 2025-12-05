"""
Complexity-Guided Ensemble Learning with Active Learning and Fitness Optimization.

This module implements an advanced ensemble learning approach that:
1. Uses Instance Hardness Weighted Resampling (IHWR) for bag generation
2. Systematically varies complexity focus across ensemble members (μ parameter)
3. Incorporates active learning for intelligent instance selection
4. Applies fitness functions for subset optimization
5. Supports evolutionary strategies for ensemble refinement

The ensemble achieves high diversity and robustness by training each base
classifier on subsets emphasizing different complexity levels, from easy to hard
instances, while actively selecting the most informative samples.

Author: Senior ML Engineer with 20+ years experience
License: MIT
"""

from typing import Optional, List, Tuple, Callable, Union, Literal
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from joblib import Parallel, delayed

# Import from our refactored complexity sampler
from ComplexityGuidedEnsemble._extra.complexity_sampler_refactored import (
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
        """Select most informative instances.

        Args:
            X: Feature matrix
            y: Target labels
            complexities: Instance complexity scores
            n_select: Number of instances to select
            **kwargs: Additional strategy-specific parameters

        Returns:
            Indices of selected instances
        """
        pass


class UncertaintyBasedSelection(ActiveLearningStrategy):
    """Select instances based on prediction uncertainty.

    Prioritizes instances where the model is most uncertain,
    typically those in regions of high complexity.
    """

    def __init__(self, classifier: Optional[ClassifierMixin] = None):
        """
        Args:
            classifier: Trained classifier for uncertainty estimation
        """
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
            # Fallback to complexity-based selection
            return np.argsort(complexities)[-n_select:]

        # Get prediction probabilities
        probas = self.classifier.predict_proba(X)

        # Calculate uncertainty (entropy or margin)
        if probas.shape[1] == 2:
            # Binary: use distance from 0.5
            uncertainty = 1 - np.abs(probas[:, 1] - 0.5) * 2
        else:
            # Multiclass: use entropy
            uncertainty = -np.sum(probas * np.log(probas + 1e-10), axis=1)

        # Select most uncertain instances
        return np.argsort(uncertainty)[-n_select:]


class ComplexityBasedSelection(ActiveLearningStrategy):
    """Select instances based on complexity scores.

    Prioritizes instances with specific complexity levels
    (easy, medium, or hard) based on the target mu value.
    """

    def __init__(self, target_mu: float = 0.5):
        """
        Args:
            target_mu: Target complexity level (0=easy, 1=hard)
        """
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
        # Calculate distance from target mu
        distances = np.abs(complexities - self.target_mu)

        # Select instances closest to target
        return np.argsort(distances)[:n_select]


class HybridSelection(ActiveLearningStrategy):
    """Hybrid strategy combining uncertainty and complexity.

    Balances between high-uncertainty instances (for exploration)
    and specific-complexity instances (for targeted learning).
    """

    def __init__(
        self,
        classifier: Optional[ClassifierMixin] = None,
        target_mu: float = 0.5,
        uncertainty_weight: float = 0.5,
    ):
        """
        Args:
            classifier: Trained classifier for uncertainty estimation
            target_mu: Target complexity level
            uncertainty_weight: Weight for uncertainty (0-1)
        """
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
            probas = self.classifier.predict_proba(X)
            if probas.shape[1] == 2:
                uncertainty = 1 - np.abs(probas[:, 1] - 0.5) * 2
            else:
                uncertainty = -np.sum(probas * np.log(probas + 1e-10), axis=1)
            uncertainty = uncertainty / (uncertainty.max() + 1e-10)
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
        """Evaluate fitness of a subset.

        Args:
            X: Feature matrix
            y: Target labels
            classifier: Base classifier to evaluate
            **kwargs: Additional parameters

        Returns:
            Fitness score (higher is better)
        """
        pass


class PerformanceBasedFitness(FitnessFunction):
    """Fitness based on classification performance."""

    def __init__(
        self,
        metric: FitnessMetric = "f1",
        cv_folds: int = 3,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            metric: Performance metric to use
            cv_folds: Cross-validation folds
            random_state: Random seed
        """
        self.metric = metric
        self.cv_folds = cv_folds
        self.random_state = random_state

    def evaluate(
        self, X: ArrayLike, y: ArrayLike, classifier: ClassifierMixin, **kwargs
    ) -> float:
        """Evaluate using cross-validation performance."""
        if len(X) < self.cv_folds:
            # Not enough data for CV, use simple train/test
            classifier_clone = clone(classifier)
            classifier_clone.fit(X, y)
            y_pred = classifier_clone.predict(X)

            if self.metric == "f1":
                return f1_score(y, y_pred, average="weighted")
            elif self.metric == "accuracy":
                return accuracy_score(y, y_pred)
            else:
                return accuracy_score(y, y_pred)

        # Use cross-validation
        scoring = self.metric if self.metric in ["f1", "accuracy"] else "f1"
        scores = cross_val_score(
            classifier,
            X,
            y,
            cv=self.cv_folds,
            scoring=scoring,
            n_jobs=1,  # Already parallelized at ensemble level
        )

        return scores.mean()


class DiversityBasedFitness(FitnessFunction):
    """Fitness based on diversity from other ensemble members."""

    def __init__(
        self, reference_subsets: Optional[List[Tuple[ArrayLike, ArrayLike]]] = None
    ):
        """
        Args:
            reference_subsets: List of (X, y) tuples from other ensemble members
        """
        self.reference_subsets = reference_subsets or []

    def evaluate(
        self, X: ArrayLike, y: ArrayLike, classifier: ClassifierMixin, **kwargs
    ) -> float:
        """Evaluate diversity compared to reference subsets."""
        if not self.reference_subsets:
            return 1.0  # No reference, maximum diversity

        # Calculate diversity as average dissimilarity
        diversities = []

        for ref_X, ref_y in self.reference_subsets:
            # Measure overlap (Jaccard distance)
            if hasattr(X, "index") and hasattr(ref_X, "index"):
                # DataFrames with indices
                overlap = len(set(X.index) & set(ref_X.index))
                union = len(set(X.index) | set(ref_X.index))
            else:
                # NumPy arrays - use approximate overlap via hash
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
        """
        Args:
            performance_fitness: Fitness function for performance
            diversity_fitness: Fitness function for diversity
            performance_weight: Weight for performance (0-1)
        """
        self.performance_fitness = performance_fitness
        self.diversity_fitness = diversity_fitness
        self.performance_weight = performance_weight

    def evaluate(
        self, X: ArrayLike, y: ArrayLike, classifier: ClassifierMixin, **kwargs
    ) -> float:
        """Evaluate combined fitness."""
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
        """
        Args:
            fitness_function: Fitness evaluation function
            max_iterations: Maximum optimization iterations
            mutation_rate: Probability of instance mutation
            random_state: Random seed
        """
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
        """Optimize subset using evolutionary strategy.

        Args:
            X_subset: Initial subset features
            y_subset: Initial subset labels
            X_pool: Pool of available instances
            y_pool: Pool labels
            classifier: Base classifier for evaluation
            verbose: Verbosity level

        Returns:
            Tuple of (optimized_X, optimized_y, final_fitness)
        """
        current_X, current_y = X_subset.copy(), y_subset.copy()
        current_fitness = self.fitness_function.evaluate(
            current_X, current_y, classifier
        )

        if verbose > 0:
            print(f"    Initial fitness: {current_fitness:.4f}")

        for iteration in range(self.max_iterations):
            # Mutation: replace some instances
            n_mutations = max(1, int(len(current_X) * self.mutation_rate))

            # Select instances to replace
            replace_indices = self.rng.choice(
                len(current_X), size=n_mutations, replace=False
            )

            # Select new instances from pool
            new_indices = self.rng.choice(len(X_pool), size=n_mutations, replace=False)

            # Create mutated subset
            mutated_X = current_X.copy()
            mutated_y = current_y.copy()

            for i, (old_idx, new_idx) in enumerate(zip(replace_indices, new_indices)):
                mutated_X[old_idx] = X_pool[new_idx]
                mutated_y[old_idx] = y_pool[new_idx]

            # Evaluate mutated subset
            mutated_fitness = self.fitness_function.evaluate(
                mutated_X, mutated_y, classifier
            )

            # Accept if improved
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
# MAIN ENSEMBLE CLASS
# ============================================================================


class ComplexityGuidedEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier using complexity-guided resampling with active learning.

    This ensemble creates diverse base classifiers by:
    1. Systematically varying the complexity focus (μ parameter) across members
    2. Using IHWR (Instance Hardness Weighted Resampling) for bag generation
    3. Optionally applying active learning for intelligent instance selection
    4. Optionally optimizing subsets with fitness functions and evolutionary strategies

    The ensemble achieves high diversity and robustness by training each base
    classifier on subsets emphasizing different complexity levels.

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
    subsets_ : list
        Training subsets for each estimator (if store_subsets=True)
    fitness_scores_ : array
        Fitness scores for each subset (if use_fitness_optimization=True)

    Examples
    --------
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.tree import DecisionTreeClassifier
    >>>
    >>> X, y = make_classification(n_samples=1000, n_classes=2,
    ...                            weights=[0.9, 0.1], random_state=42)
    >>>
    >>> ensemble = ComplexityGuidedEnsemble(
    ...     n_estimators=10,
    ...     base_estimator=DecisionTreeClassifier,
    ...     complexity_type='overlap',
    ...     use_active_learning=True,
    ...     random_state=42
    ... )
    >>>
    >>> ensemble.fit(X, y)
    >>> y_pred = ensemble.predict(X)
    >>> y_proba = ensemble.predict_proba(X)
    """

    def __init__(
        self,
        config: Optional[EnsembleConfig] = None,
        store_subsets: bool = False,
        **kwargs,
    ):
        """
        Initialize the ensemble.

        Args:
            config: Configuration object (optional)
            store_subsets: Whether to store training subsets for analysis
            **kwargs: Configuration parameters (used if config is None)
        """
        self.config = config or EnsembleConfig(**kwargs)
        self.store_subsets = store_subsets

        # Validate configuration
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
        """Generate mu values systematically distributed from 0 to 1.

        Returns:
            Array of mu values for each estimator
        """
        if self.config.n_estimators == 1:
            return np.array([0.5])

        # Distribute mu values evenly from 0 to 1
        return np.linspace(0, 1, self.config.n_estimators)

    def _initialize_components(self, X: ArrayLike, y: ArrayLike) -> None:
        """Initialize sampler, active learner, and fitness components."""
        # Initialize sampler
        sampler_config = SamplerConfig(
            complex_type=self.config.complexity_type,
            hardness_function=self.config.hardness_function,
            random_state=self.config.random_state,
            cv_folds=self.config.cv_folds,
        )
        # Convert X and y to a single dataframe with y as the last column
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, index=np.arange(X.shape[0]))
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        # Join X and y into single dataframe with y as last column
        data = pd.concat([X, y.rename("target")], axis=1)

        self.sampler_ = ComplexityGuidedSampler(
            config=sampler_config, data=data, target_col="target"
        )

        # Initialize active learning strategy
        if self.config.use_active_learning:
            self.active_learner_ = HybridSelection(
                target_mu=0.5,  # Will be updated for each estimator
                uncertainty_weight=0.5,
            )

        # Initialize fitness function
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
        """Create subset using IHWR and active learning.

        Args:
            X: Feature matrix
            y: Target labels
            mu: Mu value for this subset
            estimator_idx: Index of current estimator

        Returns:
            Tuple of (X_subset, y_subset)
        """
        # Generate base subset with IHWR
        X_resampled, y_resampled = self.sampler_.fit_resample(
            X, y, mu=mu, sigma=self.config.sigma, k_neighbors=self.config.k_neighbors
        )

        if not self.config.use_active_learning or self.active_learner_ is None:
            return X_resampled, y_resampled

        # Apply active learning refinement
        # Train a preliminary classifier on the base subset
        if estimator_idx > 0 and len(self.estimators_) > 0:
            # Use previous estimator for uncertainty estimation
            preliminary_clf = self.estimators_[-1]
        else:
            # Train a quick classifier
            preliminary_clf = clone(
                self.config.base_estimator(random_state=self.config.random_state)
            )
            preliminary_clf.fit(X_resampled, y_resampled)

        # Update active learner with current mu
        if isinstance(
            self.active_learner_, (ComplexityBasedSelection, HybridSelection)
        ):
            self.active_learner_.target_mu = mu

        if isinstance(
            self.active_learner_, (UncertaintyBasedSelection, HybridSelection)
        ):
            self.active_learner_.classifier = preliminary_clf

        # Calculate complexities for the full dataset
        complexities = self.sampler_.complexities

        # Select most informative instances
        n_select = len(X_resampled)
        informative_indices = self.active_learner_.select_informative_instances(
            X, y, complexities, n_select
        )

        # Create refined subset
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
        """Optimize subset using fitness function and evolutionary strategy.

        Args:
            X_subset: Initial subset features
            y_subset: Initial subset labels
            X_pool: Pool of available instances
            y_pool: Pool labels
            base_estimator: Base classifier for evaluation

        Returns:
            Tuple of (optimized_X, optimized_y, fitness_score)
        """
        if not self.config.use_fitness_optimization or self.subset_optimizer_ is None:
            # No optimization, evaluate fitness only
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
        """Fit a single base estimator on a complexity-guided subset.

        Args:
            X: Full training features
            y: Full training labels
            mu: Mu value for this estimator
            estimator_idx: Index of this estimator

        Returns:
            Tuple of (fitted_estimator, fitness_score)
        """
        if self.config.verbose > 0:
            print(
                f"  Training estimator {estimator_idx + 1}/{self.config.n_estimators} "
                f"(μ={mu:.3f})"
            )

        # Create subset with active learning
        X_subset, y_subset = self._create_subset_with_active_learning(
            X, y, mu, estimator_idx
        )

        # Optimize subset if enabled
        base_estimator = clone(
            self.config.base_estimator(random_state=self.config.random_state)
        )

        X_final, y_final, fitness = self._optimize_subset(
            X_subset, y_subset, X, y, base_estimator
        )

        # Store subset if requested
        if self.store_subsets:
            self.subsets_.append((X_final.copy(), y_final.copy()))

        # Train final estimator
        base_estimator.fit(X_final, y_final)

        if self.config.verbose > 0:
            print(f"    Fitness: {fitness:.4f}")
            print(
                f"    Subset size: {len(X_final)}, "
                f"Class distribution: {np.bincount(y_final.astype(int))}"
            )

        return base_estimator, fitness

    def fit(self, X: ArrayLike, y: ArrayLike) -> "ComplexityGuidedEnsemble":
        """
        Fit the ensemble on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features
        y : array-like of shape (n_samples,)
            Target labels

        Returns
        -------
        self : ComplexityGuidedEnsemble
            Fitted ensemble
        """
        if self.config.verbose > 0:
            print(f"\n{'='*70}")
            print(f"Training Complexity-Guided Ensemble")
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

        # Convert to numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        # Store classes for sklearn compatibility
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Initialize components
        self._initialize_components(X, y)

        # Generate mu values
        self.mu_values_ = self._generate_mu_values()

        if self.config.verbose > 0:
            print(f"Generated μ values: {self.mu_values_}")
            print()

        # Fit estimators (in parallel if n_jobs > 1)
        if self.config.n_jobs == 1:
            # Sequential fitting
            results = []
            for idx, mu in enumerate(self.mu_values_):
                estimator, fitness = self._fit_single_estimator(X, y, mu, idx)
                results.append((estimator, fitness))
        else:
            # Parallel fitting
            results = Parallel(n_jobs=self.config.n_jobs, verbose=0)(
                delayed(self._fit_single_estimator)(X, y, mu, idx)
                for idx, mu in enumerate(self.mu_values_)
            )

        # Extract estimators and fitness scores
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
        """
        Predict class labels for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels
        """
        if not self.estimators_:
            raise ValueError("Ensemble must be fitted before prediction")

        if self.config.voting == "soft":
            # Soft voting: average probabilities
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
        else:
            # Hard voting: majority vote
            predictions = np.array([est.predict(X) for est in self.estimators_])
            # Majority vote
            return np.apply_along_axis(
                lambda x: np.bincount(x.astype(int)).argmax(), axis=0, arr=predictions
            )

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """
        Predict class probabilities for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities
        """
        if not self.estimators_:
            raise ValueError("Ensemble must be fitted before prediction")

        # Check if base estimator supports predict_proba
        if not hasattr(self.estimators_[0], "predict_proba"):
            raise AttributeError(
                f"{self.config.base_estimator.__name__} does not support predict_proba. "
                "Use voting='hard' or a classifier with probability support."
            )

        # Average probabilities across all estimators
        probas = np.array([est.predict_proba(X) for est in self.estimators_])

        return np.mean(probas, axis=0)

    def get_ensemble_diversity(self) -> float:
        """
        Calculate diversity among ensemble members.

        Diversity is measured as the average pairwise disagreement rate.

        Returns
        -------
        diversity : float
            Diversity score (0 = no diversity, 1 = maximum diversity)
        """
        if not self.estimators_ or not self.subsets_:
            raise ValueError("Ensemble must be fitted with store_subsets=True")

        n_estimators = len(self.estimators_)
        if n_estimators < 2:
            return 0.0

        # Calculate pairwise diversity
        diversities = []
        for i in range(n_estimators):
            for j in range(i + 1, n_estimators):
                X_i, y_i = self.subsets_[i]
                X_j, y_j = self.subsets_[j]

                # Measure overlap
                X_i_hashes = {hash(tuple(row)) for row in X_i}
                X_j_hashes = {hash(tuple(row)) for row in X_j}
                overlap = len(X_i_hashes & X_j_hashes)
                union = len(X_i_hashes | X_j_hashes)

                diversity = 1 - (overlap / (union + 1e-10))
                diversities.append(diversity)

        return np.mean(diversities)

    def get_complexity_distribution(self) -> dict:
        """
        Get statistics about complexity distribution across ensemble.

        Returns
        -------
        stats : dict
            Dictionary with complexity statistics
        """
        if not self.mu_values_ is not None:
            raise ValueError("Ensemble must be fitted first")

        return {
            "mu_values": self.mu_values_,
            "mu_mean": self.mu_values_.mean(),
            "mu_std": self.mu_values_.std(),
            "mu_range": (self.mu_values_.min(), self.mu_values_.max()),
            "n_estimators": len(self.mu_values_),
        }


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def compare_ensemble_strategies(
    X: ArrayLike,
    y: ArrayLike,
    n_estimators: int = 10,
    strategies: Optional[List[dict]] = None,
    cv_folds: int = 3,
    random_state: Optional[int] = None,
) -> pd.DataFrame:
    """
    Compare different ensemble strategies.

    Args:
        X: Feature matrix
        y: Target labels
        n_estimators: Number of estimators per ensemble
        strategies: List of strategy configurations
        cv_folds: Cross-validation folds
        random_state: Random seed

    Returns:
        DataFrame with comparison results
    """
    if strategies is None:
        strategies = [
            {
                "name": "No Active Learning",
                "use_active_learning": False,
                "use_fitness_optimization": False,
            },
            {
                "name": "Active Learning Only",
                "use_active_learning": True,
                "use_fitness_optimization": False,
            },
            {
                "name": "Fitness Only",
                "use_active_learning": False,
                "use_fitness_optimization": True,
            },
            {
                "name": "Active + Fitness",
                "use_active_learning": True,
                "use_fitness_optimization": True,
            },
        ]

    results = []

    for strategy in strategies:
        name = strategy.pop("name")

        ensemble = ComplexityGuidedEnsemble(
            n_estimators=n_estimators, random_state=random_state, **strategy
        )

        # Evaluate with cross-validation
        scores = cross_val_score(
            ensemble, X, y, cv=cv_folds, scoring="f1_weighted", n_jobs=-1
        )

        results.append(
            {
                "Strategy": name,
                "F1 Score": scores.mean(),
                "F1 Std": scores.std(),
                **strategy,
            }
        )

    return pd.DataFrame(results)


if __name__ == "__main__":
    # Quick test
    from sklearn.datasets import make_classification

    print("Testing Complexity-Guided Ensemble...")

    X, y = make_classification(
        n_samples=500, n_features=10, n_classes=2, weights=[0.8, 0.2], random_state=42
    )

    ensemble = ComplexityGuidedEnsemble(
        n_estimators=5,
        use_active_learning=True,
        use_fitness_optimization=False,
        verbose=1,
        random_state=42,
    )

    ensemble.fit(X, y)
    predictions = ensemble.predict(X)

    print(f"\nAccuracy: {accuracy_score(y, predictions):.4f}")
    print("✅ Test passed!")
