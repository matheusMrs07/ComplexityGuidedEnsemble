"""
Hybrid Resampling Method Based on Instance Complexity.

This module implements a complexity-guided hybrid resampling approach that combines
undersampling and oversampling strategies to balance highly imbalanced datasets
without losing relevant information or generating low-quality synthetic samples.

The algorithm adaptively balances classes by considering instance-level complexity
metrics, weighted by a Gaussian distribution.
"""

from typing import Optional, Callable, Literal, Tuple
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from pyhard.measures import ClassificationMeasures


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

ComplexityType = Literal["error_rate", "overlap", "neighborhood"]
ArrayLike = np.ndarray


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class SamplerConfig:
    """Configuration for the complexity-based sampler.

    Attributes:
        sampling_strategy: Strategy for resampling ('auto', 'all', etc.)
        random_state: Seed for reproducibility
        complex_type: Type of complexity metric to use
        hardness_function: Custom hardness function name from pyhard
        oversample_strategy: Custom oversampling strategy class
        cv_folds: Number of cross-validation folds for error rate
        n_jobs: Number of parallel jobs (-1 for all cores)
    """

    sampling_strategy: str = "auto"
    random_state: Optional[int] = None
    complex_type: ComplexityType = "overlap"
    hardness_function: Optional[str] = None
    oversample_strategy: Optional[Callable] = None
    cv_folds: int = 5
    n_jobs: Optional[int] = None


# ============================================================================
# COMPLEXITY CALCULATORS
# ============================================================================


class ComplexityCalculator:
    """Base class for instance complexity calculation strategies."""

    @staticmethod
    def normalize(values: ArrayLike) -> ArrayLike:
        """Normalize values to [0, 1] range.

        Args:
            values: Array to normalize

        Returns:
            Normalized array
        """
        max_val = values.max()
        if max_val == 0:
            warnings.warn(
                "All complexity values are zero. Returning uniform distribution."
            )
            return np.ones_like(values) / len(values)
        return values / max_val


class ErrorRateComplexity(ComplexityCalculator):
    """Calculate complexity based on classification error rate using cross-validation."""

    def __init__(
        self,
        base_classifier: ClassifierMixin = DecisionTreeClassifier,
        cv: int = 5,
        random_state: Optional[int] = None,
    ):
        """
        Args:
            base_classifier: Classifier class to use for error estimation
            cv: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        self.base_classifier = base_classifier
        self.cv = cv
        self.random_state = random_state

    def calculate(self, X: ArrayLike, y: ArrayLike) -> ArrayLike:
        """Calculate error-based complexity for each instance.

        Uses cross-validation predictions instead of leave-one-out for efficiency.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)

        Returns:
            Array of complexity values (n_samples,)
        """
        clf = self.base_classifier(random_state=self.random_state)

        # Use cross-validation instead of LOO for better performance
        y_proba = cross_val_predict(
            clf, X, y, cv=self.cv, method="predict_proba", n_jobs=-1
        )

        # Calculate error as |y_true - P(y=0)|
        complexities = np.abs(y - y_proba[:, 0])

        return complexities


class OverlapComplexity(ComplexityCalculator):
    """Calculate complexity based on feature space overlap between classes."""

    def calculate(self, X: ArrayLike, y: ArrayLike) -> ArrayLike:
        """Calculate overlap-based complexity for each instance.

        Measures how close each instance is to the overlap region between classes.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)

        Returns:
            Array of complexity values (n_samples,)
        """
        # Normalize features to [0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_normalized = scaler.fit_transform(X)

        # Calculate bounds for each class
        unique_classes = np.unique(y)
        class_bounds = self._calculate_class_bounds(X_normalized, y, unique_classes)

        # Find overlap region center
        overlap_center = self._calculate_overlap_center(class_bounds)

        # Calculate distance to overlap center for each instance
        complexities = np.linalg.norm(X_normalized - overlap_center, axis=1)

        return self.normalize(complexities)

    @staticmethod
    def _calculate_class_bounds(X: ArrayLike, y: ArrayLike, classes: ArrayLike) -> dict:
        """Calculate min/max bounds for each class."""
        bounds = {}
        for cls in classes:
            class_mask = y == cls
            subset = X[class_mask]
            bounds[cls] = {
                "min": np.min(subset, axis=0),
                "max": np.max(subset, axis=0),
            }
        return bounds

    @staticmethod
    def _calculate_overlap_center(class_bounds: dict) -> ArrayLike:
        """Calculate center of overlap region between classes."""
        mins = np.array([v["min"] for v in class_bounds.values()])
        maxs = np.array([v["max"] for v in class_bounds.values()])

        # Second smallest min and second largest max define overlap region
        overlap_min = np.partition(mins, 1, axis=0)[1]
        overlap_max = np.partition(maxs, -2, axis=0)[-2]

        return (overlap_min + overlap_max) / 2


class NeighborhoodComplexity(ComplexityCalculator):
    """Calculate complexity based on neighborhood homogeneity."""

    def calculate(self, X: ArrayLike, y: ArrayLike) -> ArrayLike:
        """Calculate neighborhood-based complexity for each instance.

        Measures how many same-class neighbors each instance has.
        Higher values indicate instances in homogeneous regions (lower complexity).

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)

        Returns:
            Array of complexity values (n_samples,)
        """
        n_samples = len(X)
        knn = NearestNeighbors(n_neighbors=n_samples, metric="euclidean")
        knn.fit(X)

        complexities = np.zeros(n_samples)

        for i, instance in enumerate(X):
            _, indices = knn.kneighbors([instance])

            # Count consecutive same-class neighbors
            same_class_count = self._count_consecutive_same_class(indices[0], y, y[i])
            complexities[i] = same_class_count

        return self.normalize(complexities)

    @staticmethod
    def _count_consecutive_same_class(
        neighbor_indices: ArrayLike, y: ArrayLike, target_class: int
    ) -> int:
        """Count consecutive neighbors of the same class."""
        count = 0
        for idx in neighbor_indices:
            if y[idx] == target_class:
                count += 1
            else:
                break
        return count


# ============================================================================
# COMPLEXITY FACTORY
# ============================================================================


class ComplexityFactory:
    """Factory for creating complexity calculators."""

    _calculators = {
        "error_rate": ErrorRateComplexity,
        "overlap": OverlapComplexity,
        "neighborhood": NeighborhoodComplexity,
    }

    @classmethod
    def create(cls, complexity_type: ComplexityType, **kwargs) -> ComplexityCalculator:
        """Create a complexity calculator instance.

        Args:
            complexity_type: Type of complexity to calculate
            **kwargs: Additional arguments for the calculator

        Returns:
            ComplexityCalculator instance

        Raises:
            ValueError: If complexity_type is not recognized
        """
        if complexity_type not in cls._calculators:
            valid_types = ", ".join(cls._calculators.keys())
            raise ValueError(
                f"Unknown complexity type: {complexity_type}. "
                f"Valid types: {valid_types}"
            )

        calculator_class = cls._calculators[complexity_type]

        # Only ErrorRateComplexity accepts kwargs
        if complexity_type == "error_rate":
            return calculator_class(**kwargs)
        else:
            return calculator_class()


# ============================================================================
# WEIGHT FUNCTIONS
# ============================================================================


def gaussian_probability(x: ArrayLike, mu: float, sigma: float) -> ArrayLike:
    """Calculate Gaussian probability density.

    Args:
        x: Input values
        mu: Mean of the Gaussian distribution
        sigma: Standard deviation of the Gaussian distribution

    Returns:
        Probability density values
    """
    coefficient = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * np.power((x - mu) / sigma, 2)
    return coefficient * np.exp(exponent)


# ============================================================================
# SAMPLING STRATEGIES
# ============================================================================


class SyntheticSampleGenerator:
    """Generate synthetic samples using k-NN averaging."""

    def __init__(self, k: int = 3):
        """
        Args:
            k: Number of neighbors to use for averaging
        """
        self.k = k

    def generate(self, instance: ArrayLike, data: ArrayLike) -> ArrayLike:
        """Generate a synthetic sample based on k nearest neighbors.

        Args:
            instance: Reference instance
            data: Dataset to find neighbors from

        Returns:
            Synthetic sample (mean of k-NN)
        """
        k = min(self.k, len(data))

        knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        knn.fit(data)

        _, indices = knn.kneighbors([instance])

        return np.mean(data[indices[0]], axis=0)


class ResamplingStrategy:
    """Handles undersampling and oversampling operations."""

    def __init__(self, random_state: Optional[int] = None, k_neighbors: int = 3):
        """
        Args:
            random_state: Random seed for reproducibility
            k_neighbors: Number of neighbors for synthetic sample generation
        """
        self.random_state = random_state
        self.generator = SyntheticSampleGenerator(k=k_neighbors)

    def undersample(
        self, X: ArrayLike, n_samples: int, weights: ArrayLike
    ) -> ArrayLike:
        """Undersample majority class using weighted sampling.

        Args:
            X: Feature matrix
            n_samples: Number of samples to select
            weights: Sample weights (higher weight = more likely to select)

        Returns:
            Undersampled feature matrix
        """
        df = pd.DataFrame(X)
        sampled = df.sample(
            n=n_samples, weights=weights, random_state=self.random_state, replace=False
        )
        return sampled.to_numpy()

    def oversample(self, X: ArrayLike, n_samples: int, weights: ArrayLike) -> ArrayLike:
        """Oversample minority class using synthetic sample generation.

        Args:
            X: Feature matrix
            n_samples: Target number of samples
            weights: Sample weights for selecting seed instances

        Returns:
            Oversampled feature matrix
        """
        current_samples = X.copy()
        remaining = n_samples - len(current_samples)

        # Generate samples in batches
        while remaining > 0:
            batch_size = min(remaining, len(X))

            # Select seed instances using weights
            df = pd.DataFrame(X)
            seeds = df.sample(
                n=batch_size,
                weights=weights,
                random_state=self.random_state,
                replace=True,
            )

            # Generate synthetic samples
            synthetic = np.array(
                [
                    self.generator.generate(seed.values, current_samples)
                    for _, seed in seeds.iterrows()
                ]
            )

            current_samples = np.vstack([current_samples, synthetic])
            remaining = n_samples - len(current_samples)

        return current_samples[:n_samples]


# ============================================================================
# MAIN SAMPLER CLASS
# ============================================================================


class ComplexityGuidedSampler:
    """
    Hybrid resampling method guided by instance complexity.

    This sampler balances imbalanced datasets by:
    1. Calculating complexity metrics for each instance
    2. Weighting instances using a Gaussian distribution
    3. Undersampling majority class (removing easy/hard instances)
    4. Oversampling minority class (generating synthetic samples)

    Example:
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(
        ...     n_samples=1000, n_classes=2, weights=[0.9, 0.1],
        ...     random_state=42
        ... )
        >>>
        >>> sampler = ComplexityGuidedSampler(
        ...     complex_type="overlap",
        ...     random_state=42
        ... )
        >>> X_resampled, y_resampled = sampler.fit_resample(
        ...     X, y, mu=0.5, sigma=0.2, k_neighbors=5
        ... )
    """

    def __init__(
        self,
        data: Optional[pd.DataFrame] = None,
        target_col: Optional[str] = None,
        config: Optional[SamplerConfig] = None,
        **kwargs,
    ):
        """
        Initialize the sampler.

        Args:
            data: DataFrame with features and target (optional)
            target_col: Name of the target column in data (optional)
            config: Configuration object (optional)
            **kwargs: Configuration parameters (used if config is None)
        """
        self.config = config or SamplerConfig(**kwargs)
        self._validate_config()

        # Initialize data structures
        self.X_train: Optional[ArrayLike] = None
        self.y_train: Optional[ArrayLike] = None
        self.complexities: Optional[ArrayLike] = None
        self.n_samples: Optional[int] = None
        self.majority_mask: Optional[ArrayLike] = None
        self.minority_mask: Optional[ArrayLike] = None

        # Initialize components
        self._complexity_calculator: Optional[ComplexityCalculator] = None
        self._resampling_strategy: Optional[ResamplingStrategy] = None
        self._measures_calculator: Optional[ClassificationMeasures] = None

        # Load data if provided
        if data is not None:
            if target_col is None:
                target_col = data.columns[-1]  # Use last column as default target
                warnings.warn(
                    f"No target_col specified, using last column '{target_col}' as target"
                )
            self._load_data(data, target_col)

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.config.complex_type not in ["error_rate", "overlap", "neighborhood"]:
            raise ValueError(f"Invalid complex_type: {self.config.complex_type}")

        if self.config.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")

    def _load_data(self, data: pd.DataFrame, target_col: str) -> None:
        """Load and prepare data from DataFrame.

        Workaround: rename columns to 0..n-1 before calling ClassificationMeasures.
        Isso evita o KeyError proveniente de um acesso incorreto a feature.dtypes[0]
        na implementação atual de pyhard.measures.gower_distance.
        """
        # copia para não alterar o DataFrame original
        df = data.copy()

        # renomear colunas para inteiros sequenciais (0..n-1)
        orig_cols = list(df.columns)
        new_cols = list(range(len(orig_cols)))
        mapping = dict(zip(orig_cols, new_cols))
        df = df.rename(columns=mapping)

        # resolver target_col para o novo nome (inteiro)
        if isinstance(target_col, int):
            target_col_renamed = target_col
        else:
            if target_col not in mapping:
                raise ValueError(
                    f"target_col '{target_col}' não encontrado nas colunas do DataFrame"
                )
            target_col_renamed = mapping[target_col]

        # inicializar o calculador de medidas com o DataFrame renomeado
        self._measures_calculator = ClassificationMeasures(df, target_col_renamed)

        # extrair X e y já preparados pelo ClassificationMeasures
        self.X_train = self._measures_calculator.X.to_numpy()
        self.y_train = self._measures_calculator.y.to_numpy()

        self._validate_data()

    # def _load_data(self, data: pd.DataFrame, target_col: str) -> None:
    #     """Load and prepare data from DataFrame."""
    #     self._measures_calculator = ClassificationMeasures(data, target_col)
    #     self.X_train = self._measures_calculator.X.to_numpy()
    #     self.y_train = self._measures_calculator.y.to_numpy()

    #     self._validate_data()

    def _validate_data(self) -> None:
        """Validate loaded data."""
        # if len(np.unique(self.y_train)) != 2:
        #     raise ValueError("This sampler only supports binary classification")

        if len(self.X_train) < 10:
            raise ValueError("Dataset too small (minimum 10 samples required)")

    def _calculate_complexities(self) -> ArrayLike:
        """Calculate instance complexities using the configured method.

        Returns:
            Array of complexity values
        """
        # Use custom hardness function if provided
        if self.config.hardness_function and self._measures_calculator:
            try:
                return getattr(
                    self._measures_calculator, self.config.hardness_function
                )()
            except AttributeError:
                warnings.warn(
                    f"Hardness function '{self.config.hardness_function}' not found. "
                    f"Falling back to {self.config.complex_type}."
                )

        # Use built-in complexity calculator
        if self._complexity_calculator is None:
            self._complexity_calculator = ComplexityFactory.create(
                self.config.complex_type,
                cv=self.config.cv_folds,
                random_state=self.config.random_state,
            )

        return self._complexity_calculator.calculate(self.X_train, self.y_train)

    def _calculate_target_samples(self) -> int:
        """Calculate target number of samples per class after balancing.

        Returns:
            Target number of samples per class
        """
        unique_classes, class_counts = np.unique(self.y_train, return_counts=True)
        return int(np.mean(class_counts))

    def _identify_majority_minority(self) -> Tuple[ArrayLike, ArrayLike]:
        """Identify majority and minority class masks.

        Returns:
            Tuple of (majority_mask, minority_mask)
        """
        unique_classes, class_counts = np.unique(self.y_train, return_counts=True)

        majority_class = unique_classes[np.argmax(class_counts)]
        minority_class = unique_classes[np.argmin(class_counts)]

        majority_mask = self.y_train == majority_class
        minority_mask = self.y_train == minority_class

        return majority_mask, minority_mask

    def fit(
        self, X: Optional[ArrayLike] = None, y: Optional[ArrayLike] = None
    ) -> "ComplexityGuidedSampler":
        """
        Fit the sampler to the data.

        Calculates complexities and identifies majority/minority classes.

        Args:
            X: Feature matrix (optional if data was provided in __init__)
            y: Target labels (optional if data was provided in __init__)

        Returns:
            self
        """
        # Load data if provided
        if X is not None and y is not None:
            self.X_train = np.asarray(X)
            self.y_train = np.asarray(y)
            self._validate_data()
        elif self.X_train is None or self.y_train is None:
            raise ValueError("No data available. Provide data in __init__ or fit()")

        # Calculate complexities
        self.complexities = self._calculate_complexities()

        # Identify classes
        self.majority_mask, self.minority_mask = self._identify_majority_minority()

        # Calculate target sample size
        self.n_samples = self._calculate_target_samples()

        return self

    def resample(
        self, mu: float, sigma: float, k_neighbors: int = 3
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Resample the data using complexity-guided strategy.

        Args:
            mu: Mean of Gaussian weight distribution (0-1)
                - mu=0: favor easy instances
                - mu=1: favor hard instances
                - mu=0.5: uniform sampling
            sigma: Standard deviation of Gaussian (controls spread)
            k_neighbors: Number of neighbors for synthetic sample generation

        Returns:
            Tuple of (X_resampled, y_resampled)

        Raises:
            ValueError: If sampler hasn't been fitted
        """
        if self.complexities is None:
            raise ValueError(
                "Sampler must be fitted before resampling. Call fit() first."
            )

        # Initialize resampling strategy
        self._resampling_strategy = ResamplingStrategy(
            random_state=self.config.random_state, k_neighbors=k_neighbors
        )

        # Calculate sample weights using Gaussian
        weights = gaussian_probability(self.complexities, mu, sigma)

        # Undersample majority class
        X_majority_resampled = self._resampling_strategy.undersample(
            X=self.X_train[self.majority_mask],
            n_samples=self.n_samples,
            weights=weights[self.majority_mask],
        )

        # Get majority class label
        majority_label = self.y_train[self.majority_mask][0]

        # Handle minority class oversampling
        if self.config.oversample_strategy is not None:
            # Use external oversampling strategy (e.g., SMOTE)
            X_combined = np.vstack(
                [X_majority_resampled, self.X_train[self.minority_mask]]
            )

            minority_label = self.y_train[self.minority_mask][0]
            y_combined = np.concatenate(
                [
                    np.full(len(X_majority_resampled), majority_label),
                    np.full(np.sum(self.minority_mask), minority_label),
                ]
            )

            oversampler = self.config.oversample_strategy()
            return oversampler.fit_resample(X_combined, y_combined)

        # Use built-in oversampling
        X_minority_resampled = self._resampling_strategy.oversample(
            X=self.X_train[self.minority_mask],
            n_samples=self.n_samples,
            weights=weights[self.minority_mask],
        )

        # Combine resampled data
        X_resampled = np.vstack([X_majority_resampled, X_minority_resampled])

        minority_label = self.y_train[self.minority_mask][0]
        y_resampled = np.concatenate(
            [
                np.full(len(X_majority_resampled), majority_label),
                np.full(len(X_minority_resampled), minority_label),
            ]
        )

        return X_resampled, y_resampled

    def fit_resample(
        self,
        X: Optional[ArrayLike] = None,
        y: Optional[ArrayLike] = None,
        mu: float = 0.5,
        sigma: float = 0.2,
        k_neighbors: int = 3,
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Fit the sampler and resample the data in one step.

        Args:
            X: Feature matrix (optional if data was provided in __init__)
            y: Target labels (optional if data was provided in __init__)
            mu: Mean of Gaussian weight distribution
            sigma: Standard deviation of Gaussian
            k_neighbors: Number of neighbors for synthetic sample generation

        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        self.fit(X, y)
        return self.resample(mu, sigma, k_neighbors)


# ============================================================================
# LEGACY COMPATIBILITY (DEPRECATED)
# ============================================================================


class Sampler(ComplexityGuidedSampler):
    """
    Legacy class name for backward compatibility.

    .. deprecated::
        Use ComplexityGuidedSampler instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Sampler is deprecated. Use ComplexityGuidedSampler instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


# ============================================================================
# UTILITY FUNCTIONS (DEPRECATED)
# ============================================================================


def get_complexities(
    X: ArrayLike,
    y: ArrayLike,
    complex_type: Optional[ComplexityType] = None,
    base_classifier: ClassifierMixin = DecisionTreeClassifier,
) -> ArrayLike:
    """
    Calculate instance complexities (legacy function).

    .. deprecated::
        Use ComplexityFactory.create() and calculator.calculate() instead.
    """
    warnings.warn(
        "get_complexities is deprecated. Use ComplexityFactory instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    if complex_type is None:
        complex_type = "overlap"

    calculator = ComplexityFactory.create(complex_type)
    return calculator.calculate(X, y)
