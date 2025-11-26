"""
Hybrid Resampling Method Based on Instance Complexity - Multiclass Version.

This module implements a complexity-guided hybrid resampling approach that combines
undersampling and oversampling strategies to balance highly imbalanced datasets
without losing relevant information or generating low-quality synthetic samples.

Supports multiclass classification by treating each class independently and
balancing towards the mean class size.

Author: Adapted for multiclass support
License: MIT
"""

from typing import Optional, Callable, Literal, Tuple, Dict, List
from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
from sklearn.base import BaseEstimator, ClassifierMixin

# Optional pyhard import - gracefully handle if not available
try:
    from pyhard.measures import ClassificationMeasures

    PYHARD_AVAILABLE = True
except ImportError:
    PYHARD_AVAILABLE = False
    ClassificationMeasures = None


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
        sampling_strategy: Strategy for resampling ('auto', 'all', 'mean', 'median')
        random_state: Seed for reproducibility
        complex_type: Type of complexity metric to use
        hardness_function: Custom hardness function name from pyhard
        oversample_strategy: Custom oversampling strategy class
        cv_folds: Number of cross-validation folds for error rate
        n_jobs: Number of parallel jobs (-1 for all cores)
        target_ratio: Target ratio for balancing (1.0 = fully balanced)
    """

    sampling_strategy: str = "auto"
    random_state: Optional[int] = None
    complex_type: ComplexityType = "overlap"
    hardness_function: Optional[str] = None
    oversample_strategy: Optional[Callable] = None
    cv_folds: int = 5
    n_jobs: Optional[int] = None
    target_ratio: float = 1.0


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
        min_val = values.min()
        max_val = values.max()

        if max_val == min_val:
            warnings.warn(
                "All complexity values are identical. Returning uniform distribution."
            )
            return np.ones_like(values) / len(values)

        return (values - min_val) / (max_val - min_val)


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

        Uses cross-validation predictions. For multiclass, uses 1 - P(correct class).

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)

        Returns:
            Array of complexity values (n_samples,)
        """
        clf = self.base_classifier(random_state=self.random_state)

        # Adjust cv if dataset is too small
        n_samples = len(X)
        cv = min(self.cv, n_samples)

        # Use cross-validation to get probability predictions
        try:
            y_proba = cross_val_predict(
                clf, X, y, cv=cv, method="predict_proba", n_jobs=-1
            )
        except ValueError:
            # Fallback if CV fails (e.g., too few samples per class)
            clf.fit(X, y)
            y_proba = clf.predict_proba(X)

        # Get the classes from the classifier
        classes = clf.classes_ if hasattr(clf, "classes_") else np.unique(y)

        # Calculate complexity as 1 - P(true class)
        complexities = np.zeros(len(y))
        for i, (true_label, proba) in enumerate(zip(y, y_proba)):
            # Find index of true class
            class_idx = np.where(classes == true_label)[0]
            if len(class_idx) > 0:
                complexities[i] = 1 - proba[class_idx[0]]
            else:
                complexities[i] = 1.0  # Maximum complexity if class not found

        return complexities


class OverlapComplexity(ComplexityCalculator):
    """Calculate complexity based on feature space overlap between classes."""

    def calculate(self, X: ArrayLike, y: ArrayLike) -> ArrayLike:
        """Calculate overlap-based complexity for each instance.

        For multiclass: measures distance to the centroid of other classes.
        Higher values = further from other classes = easier instances.
        We invert this so higher = harder.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)

        Returns:
            Array of complexity values (n_samples,)
        """
        # Normalize features to [0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        X_normalized = scaler.fit_transform(X)

        unique_classes = np.unique(y)
        n_classes = len(unique_classes)

        # Calculate centroid for each class
        centroids = {}
        for cls in unique_classes:
            class_mask = y == cls
            centroids[cls] = np.mean(X_normalized[class_mask], axis=0)

        # Calculate complexity based on distance to other class centroids
        complexities = np.zeros(len(X))

        for i, (instance, label) in enumerate(zip(X_normalized, y)):
            # Calculate distances to all other class centroids
            other_distances = []
            own_distance = np.linalg.norm(instance - centroids[label])

            for cls, centroid in centroids.items():
                if cls != label:
                    dist = np.linalg.norm(instance - centroid)
                    other_distances.append(dist)

            if other_distances:
                # Complexity = how close to other classes relative to own class
                min_other_dist = min(other_distances)
                # Higher complexity when close to other classes
                complexities[i] = (
                    1 / (1 + min_other_dist) if min_other_dist > 0 else 1.0
                )
            else:
                complexities[i] = 0.0

        return self.normalize(complexities)


class NeighborhoodComplexity(ComplexityCalculator):
    """Calculate complexity based on neighborhood homogeneity."""

    def __init__(self, k_neighbors: int = 5):
        """
        Args:
            k_neighbors: Number of neighbors to consider
        """
        self.k_neighbors = k_neighbors

    def calculate(self, X: ArrayLike, y: ArrayLike) -> ArrayLike:
        """Calculate neighborhood-based complexity for each instance.

        Measures the proportion of different-class neighbors.
        Higher values indicate instances in heterogeneous regions (higher complexity).

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target labels (n_samples,)

        Returns:
            Array of complexity values (n_samples,)
        """
        n_samples = len(X)
        k = min(
            self.k_neighbors + 1, n_samples
        )  # +1 because instance is its own neighbor

        knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        knn.fit(X)

        complexities = np.zeros(n_samples)

        for i, instance in enumerate(X):
            _, indices = knn.kneighbors([instance])

            # Exclude the instance itself
            neighbor_indices = indices[0][1:] if len(indices[0]) > 1 else indices[0]

            if len(neighbor_indices) > 0:
                # Count neighbors with different class
                different_class_count = np.sum(y[neighbor_indices] != y[i])
                complexities[i] = different_class_count / len(neighbor_indices)
            else:
                complexities[i] = 0.0

        return complexities  # Already in [0, 1]


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

    def __init__(self, k: int = 3, random_state: Optional[int] = None):
        """
        Args:
            k: Number of neighbors to use for averaging
            random_state: Random seed for reproducibility
        """
        self.k = k
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def generate(self, instance: ArrayLike, data: ArrayLike) -> ArrayLike:
        """Generate a synthetic sample based on k nearest neighbors.

        Uses SMOTE-like interpolation between instance and a random neighbor.

        Args:
            instance: Reference instance
            data: Dataset to find neighbors from

        Returns:
            Synthetic sample
        """
        k = min(self.k, len(data))

        knn = NearestNeighbors(n_neighbors=k, metric="euclidean")
        knn.fit(data)

        _, indices = knn.kneighbors([instance])

        # Select a random neighbor
        neighbor_idx = self.rng.choice(indices[0])
        neighbor = data[neighbor_idx]

        # Interpolate between instance and neighbor
        alpha = self.rng.random()
        synthetic = instance + alpha * (neighbor - instance)

        return synthetic


class ResamplingStrategy:
    """Handles undersampling and oversampling operations."""

    def __init__(self, random_state: Optional[int] = None, k_neighbors: int = 3):
        """
        Args:
            random_state: Random seed for reproducibility
            k_neighbors: Number of neighbors for synthetic sample generation
        """
        self.random_state = random_state
        self.generator = SyntheticSampleGenerator(
            k=k_neighbors, random_state=random_state
        )
        self.rng = np.random.RandomState(random_state)

    def undersample(
        self, X: ArrayLike, n_samples: int, weights: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Undersample class using weighted sampling.

        Args:
            X: Feature matrix
            n_samples: Number of samples to select
            weights: Sample weights (higher weight = more likely to select)

        Returns:
            Tuple of (undersampled features, selected indices)
        """
        if n_samples >= len(X):
            return X.copy(), np.arange(len(X))

        # Normalize weights
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

        # Sample without replacement
        indices = self.rng.choice(len(X), size=n_samples, replace=False, p=weights)

        return X[indices], indices

    def oversample(self, X: ArrayLike, n_samples: int, weights: ArrayLike) -> ArrayLike:
        """Oversample class using synthetic sample generation.

        Args:
            X: Feature matrix
            n_samples: Target number of samples
            weights: Sample weights for selecting seed instances

        Returns:
            Oversampled feature matrix
        """
        if n_samples <= len(X):
            return X.copy()

        current_samples = list(X)
        remaining = n_samples - len(current_samples)

        # Normalize weights
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()

        # Generate synthetic samples
        for _ in range(remaining):
            # Select seed instance using weights
            seed_idx = self.rng.choice(len(X), p=weights)
            seed = X[seed_idx]

            # Generate synthetic sample
            synthetic = self.generator.generate(seed, X)
            current_samples.append(synthetic)

        return np.array(current_samples)


# ============================================================================
# MAIN SAMPLER CLASS - MULTICLASS VERSION
# ============================================================================


class ComplexityGuidedSampler:
    """
    Hybrid resampling method guided by instance complexity - Multiclass Version.

    This sampler balances imbalanced datasets by:
    1. Calculating complexity metrics for each instance
    2. Weighting instances using a Gaussian distribution
    3. For each class:
       - If class size > target: undersample (removing based on complexity weights)
       - If class size < target: oversample (generating synthetic samples)

    Supports any number of classes (binary and multiclass).

    Example:
        >>> from sklearn.datasets import make_classification
        >>> X, y = make_classification(
        ...     n_samples=1000, n_classes=3, n_clusters_per_class=1,
        ...     weights=[0.7, 0.2, 0.1], random_state=42
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
        self.target_samples_per_class: Optional[int] = None
        self.classes_: Optional[ArrayLike] = None
        self.class_counts_: Optional[Dict] = None

        # Initialize components
        self._complexity_calculator: Optional[ComplexityCalculator] = None
        self._resampling_strategy: Optional[ResamplingStrategy] = None
        self._measures_calculator: Optional[ClassificationMeasures] = None

        # Load data if provided
        if data is not None:
            if target_col is None:
                target_col = data.columns[-1]
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
        """Load and prepare data from DataFrame."""
        df = data.copy()

        # Extract target column
        if target_col not in df.columns:
            raise ValueError(
                f"target_col '{target_col}' not found in DataFrame columns"
            )

        self.y_train = df[target_col].to_numpy()
        self.X_train = df.drop(columns=[target_col]).to_numpy()

        # Initialize measures calculator if pyhard is available
        if PYHARD_AVAILABLE:
            # Rename columns to integers for pyhard compatibility
            orig_cols = list(df.columns)
            new_cols = list(range(len(orig_cols)))
            mapping = dict(zip(orig_cols, new_cols))
            df_renamed = df.rename(columns=mapping)

            if isinstance(target_col, int):
                target_col_renamed = target_col
            else:
                target_col_renamed = mapping[target_col]

            try:
                self._measures_calculator = ClassificationMeasures(
                    df_renamed, target_col_renamed
                )
            except Exception as e:
                warnings.warn(
                    f"Could not initialize pyhard ClassificationMeasures: {e}"
                )
                self._measures_calculator = None
        else:
            self._measures_calculator = None

        self._validate_data()

    def _validate_data(self) -> None:
        """Validate loaded data."""
        if len(self.X_train) < 10:
            raise ValueError("Dataset too small (minimum 10 samples required)")

        unique_classes = np.unique(self.y_train)
        if len(unique_classes) < 2:
            raise ValueError("Dataset must have at least 2 classes")

    def _calculate_complexities(self) -> ArrayLike:
        """Calculate instance complexities using the configured method.

        Returns:
            Array of complexity values
        """
        # Use custom hardness function if provided and pyhard is available
        if self.config.hardness_function and self._measures_calculator is not None:
            try:
                result = getattr(
                    self._measures_calculator, self.config.hardness_function
                )()
                return result
            except AttributeError:
                warnings.warn(
                    f"Hardness function '{self.config.hardness_function}' not found. "
                    f"Falling back to {self.config.complex_type}."
                )
            except Exception as e:
                warnings.warn(
                    f"Error using hardness function: {e}. "
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
        class_counts = list(self.class_counts_.values())

        if self.config.sampling_strategy == "median":
            return int(np.median(class_counts))
        elif self.config.sampling_strategy == "max":
            return int(np.max(class_counts))
        elif self.config.sampling_strategy == "min":
            return int(np.min(class_counts))
        else:  # 'auto' or 'mean'
            return int(np.mean(class_counts))

    def _analyze_classes(self) -> None:
        """Analyze class distribution."""
        self.classes_, counts = np.unique(self.y_train, return_counts=True)
        self.class_counts_ = dict(zip(self.classes_, counts))

    def fit(
        self, X: Optional[ArrayLike] = None, y: Optional[ArrayLike] = None
    ) -> "ComplexityGuidedSampler":
        """
        Fit the sampler to the data.

        Calculates complexities and analyzes class distribution.

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

        # Analyze classes
        self._analyze_classes()

        # Calculate target sample size
        self.target_samples_per_class = self._calculate_target_samples()

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

        # Ensure weights are positive
        weights = np.maximum(weights, 1e-10)

        # Process each class
        X_resampled_list = []
        y_resampled_list = []

        for cls in self.classes_:
            class_mask = self.y_train == cls
            X_class = self.X_train[class_mask]
            weights_class = weights[class_mask]
            current_count = len(X_class)
            target_count = self.target_samples_per_class

            if current_count > target_count:
                # Undersample this class
                X_class_resampled, _ = self._resampling_strategy.undersample(
                    X=X_class,
                    n_samples=target_count,
                    weights=weights_class,
                )
            elif current_count < target_count:
                # Oversample this class
                X_class_resampled = self._resampling_strategy.oversample(
                    X=X_class,
                    n_samples=target_count,
                    weights=weights_class,
                )
            else:
                # Keep as is
                X_class_resampled = X_class.copy()

            X_resampled_list.append(X_class_resampled)
            y_resampled_list.append(np.full(len(X_class_resampled), cls))

        # Combine all classes
        X_resampled = np.vstack(X_resampled_list)
        y_resampled = np.concatenate(y_resampled_list)

        # Shuffle the combined data
        shuffle_idx = np.random.RandomState(self.config.random_state).permutation(
            len(X_resampled)
        )
        X_resampled = X_resampled[shuffle_idx]
        y_resampled = y_resampled[shuffle_idx]

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

    def get_class_info(self) -> Dict:
        """Get information about class distribution.

        Returns:
            Dictionary with class distribution info
        """
        if self.class_counts_ is None:
            raise ValueError("Sampler must be fitted first")

        return {
            "classes": self.classes_.tolist(),
            "original_counts": self.class_counts_,
            "target_samples_per_class": self.target_samples_per_class,
            "n_classes": len(self.classes_),
        }


# ============================================================================
# LEGACY COMPATIBILITY
# ============================================================================


class Sampler(ComplexityGuidedSampler):
    """Legacy class name for backward compatibility."""

    def __init__(self, *args, **kwargs):
        warnings.warn(
            "Sampler is deprecated. Use ComplexityGuidedSampler instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_complexities(
    X: ArrayLike,
    y: ArrayLike,
    complex_type: Optional[ComplexityType] = None,
    base_classifier: ClassifierMixin = DecisionTreeClassifier,
) -> ArrayLike:
    """Calculate instance complexities."""
    if complex_type is None:
        complex_type = "overlap"

    calculator = ComplexityFactory.create(complex_type)
    return calculator.calculate(X, y)
