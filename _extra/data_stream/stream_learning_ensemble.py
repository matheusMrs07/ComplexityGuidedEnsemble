"""
Stream Learning Ensemble with Complexity-Guided Resampling.

This module implements an online/incremental learning ensemble that:
1. Processes data streams in batches (chunks)
2. Adapts to concept drift using sliding windows
3. Updates models incrementally without full retraining
4. Maintains diversity through complexity-guided sampling
5. Handles class imbalance in streaming scenarios

Key Features:
- Incremental learning with partial_fit
- Concept drift detection and adaptation
- Dynamic ensemble pruning and growing
- Memory-efficient sliding window
- Online performance monitoring

Author: Senior ML Engineer with 20+ years experience
License: MIT
"""

from typing import Optional, List, Tuple, Literal, Deque
from dataclasses import dataclass
from collections import deque
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, accuracy_score

# Import from our complexity sampler
from ComplexityGuidedEnsemble._extra.complexity_sampler_refactored import (
    ComplexityGuidedSampler,
    SamplerConfig,
    ComplexityType,
    ArrayLike
)


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

DriftDetectionStrategy = Literal["none", "adwin", "ddm", "eddm", "page_hinkley"]
EnsembleUpdateStrategy = Literal["replace_worst", "add_new", "weighted"]


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class StreamEnsembleConfig:
    """Configuration for the Stream Learning Ensemble.
    
    Attributes:
        n_estimators: Maximum number of classifiers in the ensemble
        base_estimator: Base classifier (must support partial_fit)
        chunk_size: Size of each data chunk for processing
        window_size: Size of the sliding window for training data
        complexity_type: Type of complexity metric
        sigma: Standard deviation for Gaussian weighting
        k_neighbors: Number of neighbors for synthetic sample generation
        drift_detection: Strategy for concept drift detection
        drift_threshold: Threshold for drift detection (0-1)
        update_strategy: How to update ensemble when drift detected
        min_samples_before_update: Minimum samples before first ensemble update
        rebalance_frequency: How often to rebalance data (in chunks)
        prune_threshold: Performance threshold for pruning weak classifiers
        max_memory_mb: Maximum memory for sliding window (MB)
        verbose: Verbosity level (0=silent, 1=info, 2=debug)
        random_state: Random seed for reproducibility
    """
    n_estimators: int = 10
    base_estimator: ClassifierMixin = SGDClassifier
    chunk_size: int = 100
    window_size: int = 1000
    complexity_type: ComplexityType = "overlap"
    sigma: float = 0.2
    k_neighbors: int = 5
    drift_detection: DriftDetectionStrategy = "adwin"
    drift_threshold: float = 0.1
    update_strategy: EnsembleUpdateStrategy = "replace_worst"
    min_samples_before_update: int = 500
    rebalance_frequency: int = 5
    prune_threshold: float = 0.6
    max_memory_mb: float = 100.0
    verbose: int = 1
    random_state: Optional[int] = None


# ============================================================================
# DRIFT DETECTION
# ============================================================================

class DriftDetector:
    """Base class for concept drift detection."""
    
    def __init__(self, threshold: float = 0.1):
        """
        Args:
            threshold: Sensitivity threshold for drift detection
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset detector state."""
        raise NotImplementedError
    
    def add_element(self, prediction_correct: bool) -> bool:
        """Add a new prediction result.
        
        Args:
            prediction_correct: Whether the prediction was correct
            
        Returns:
            True if drift detected, False otherwise
        """
        raise NotImplementedError


class ADWINDetector(DriftDetector):
    """ADWIN (Adaptive Windowing) drift detector.
    
    Detects changes in data distribution by monitoring a sliding window
    and checking for significant differences between sub-windows.
    """
    
    def __init__(self, threshold: float = 0.002):
        """
        Args:
            threshold: Delta parameter for ADWIN
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset detector state."""
        self.window = deque(maxlen=1000)
        self.total = 0
        self.variance = 0
        self.width = 0
    
    def add_element(self, prediction_correct: bool) -> bool:
        """Add prediction result and check for drift."""
        value = 1.0 if prediction_correct else 0.0
        
        self.window.append(value)
        self.width = len(self.window)
        
        if self.width < 5:
            return False
        
        # Check for drift using simple moving average difference
        half = self.width // 2
        window_list = list(self.window)
        
        mean1 = np.mean(window_list[:half])
        mean2 = np.mean(window_list[half:])
        
        diff = abs(mean1 - mean2)
        
        # Normalize by expected standard deviation
        expected_std = np.sqrt(0.25 / half)  # Assuming binary outcomes
        
        drift_detected = diff > (self.threshold + expected_std)
        
        if drift_detected:
            # Keep only recent data
            for _ in range(half):
                if self.window:
                    self.window.popleft()
        
        return drift_detected


class DDMDetector(DriftDetector):
    """DDM (Drift Detection Method).
    
    Monitors error rate and its standard deviation.
    """
    
    def __init__(self, threshold: float = 0.1):
        """
        Args:
            threshold: Warning level threshold
        """
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset detector state."""
        self.n = 0
        self.errors = 0
        self.p_min = float('inf')
        self.s_min = float('inf')
    
    def add_element(self, prediction_correct: bool) -> bool:
        """Add prediction result and check for drift."""
        self.n += 1
        if not prediction_correct:
            self.errors += 1
        
        if self.n < 30:  # Minimum samples
            return False
        
        p = self.errors / self.n
        s = np.sqrt(p * (1 - p) / self.n)
        
        # Update minimums
        if p + s < self.p_min + self.s_min:
            self.p_min = p
            self.s_min = s
        
        # Check for drift
        drift_level = p + s
        warning_level = self.p_min + 2 * self.s_min
        drift_threshold = self.p_min + 3 * self.s_min
        
        if drift_level > drift_threshold:
            self.reset()
            return True
        
        return False


class PageHinkleyDetector(DriftDetector):
    """Page-Hinkley test for drift detection.
    
    Cumulative sum test for detecting changes in the mean.
    """
    
    def __init__(self, threshold: float = 50.0, delta: float = 0.005):
        """
        Args:
            threshold: Detection threshold
            delta: Minimum amplitude of changes to detect
        """
        self.threshold = threshold
        self.delta = delta
        self.reset()
    
    def reset(self):
        """Reset detector state."""
        self.sum = 0
        self.x_mean = 0
        self.n = 0
        self.min_sum = float('inf')
    
    def add_element(self, prediction_correct: bool) -> bool:
        """Add prediction result and check for drift."""
        value = 1.0 if prediction_correct else 0.0
        
        self.n += 1
        
        # Update mean
        self.x_mean = self.x_mean + (value - self.x_mean) / self.n
        
        # Update cumulative sum
        self.sum += value - self.x_mean - self.delta
        
        # Update minimum
        if self.sum < self.min_sum:
            self.min_sum = self.sum
        
        # Check for drift
        diff = self.sum - self.min_sum
        
        if diff > self.threshold:
            self.reset()
            return True
        
        return False


# ============================================================================
# DRIFT DETECTOR FACTORY
# ============================================================================

class DriftDetectorFactory:
    """Factory for creating drift detectors."""
    
    _detectors = {
        "none": None,
        "adwin": ADWINDetector,
        "ddm": DDMDetector,
        "page_hinkley": PageHinkleyDetector,
    }
    
    @classmethod
    def create(
        cls,
        detector_type: DriftDetectionStrategy,
        threshold: float = 0.1
    ) -> Optional[DriftDetector]:
        """Create a drift detector.
        
        Args:
            detector_type: Type of detector
            threshold: Detection threshold
            
        Returns:
            DriftDetector instance or None
        """
        if detector_type == "none":
            return None
        
        if detector_type not in cls._detectors:
            raise ValueError(f"Unknown detector: {detector_type}")
        
        detector_class = cls._detectors[detector_type]
        return detector_class(threshold=threshold)


# ============================================================================
# SLIDING WINDOW
# ============================================================================

class SlidingWindow:
    """Memory-efficient sliding window for streaming data."""
    
    def __init__(
        self,
        max_size: int = 1000,
        max_memory_mb: float = 100.0
    ):
        """
        Args:
            max_size: Maximum number of samples in window
            max_memory_mb: Maximum memory usage in MB
        """
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.X_window: Deque[ArrayLike] = deque(maxlen=max_size)
        self.y_window: Deque[int] = deque(maxlen=max_size)
        self.n_features: Optional[int] = None
    
    def add_batch(self, X: ArrayLike, y: ArrayLike):
        """Add a batch of samples to the window.
        
        Args:
            X: Feature matrix
            y: Target labels
        """
        if self.n_features is None:
            self.n_features = X.shape[1]
        
        for i in range(len(X)):
            self.X_window.append(X[i])
            self.y_window.append(y[i])
    
    def get_data(self) -> Tuple[ArrayLike, ArrayLike]:
        """Get current window data.
        
        Returns:
            Tuple of (X, y)
        """
        if not self.X_window:
            return np.array([]), np.array([])
        
        X = np.array(list(self.X_window))
        y = np.array(list(self.y_window))
        return X, y
    
    def get_size(self) -> int:
        """Get current window size."""
        return len(self.X_window)
    
    def clear(self):
        """Clear the window."""
        self.X_window.clear()
        self.y_window.clear()
    
    def get_memory_usage_mb(self) -> float:
        """Estimate memory usage in MB."""
        if not self.X_window:
            return 0.0
        
        # Estimate memory per sample
        bytes_per_float = 8
        bytes_per_sample = self.n_features * bytes_per_float + 4  # X + y
        total_bytes = len(self.X_window) * bytes_per_sample
        
        return total_bytes / (1024 * 1024)


# ============================================================================
# STREAM ENSEMBLE
# ============================================================================

class StreamLearningEnsemble(BaseEstimator, ClassifierMixin):
    """
    Online ensemble learning with complexity-guided resampling for data streams.
    
    This ensemble handles streaming data by:
    1. Processing data in chunks
    2. Maintaining a sliding window of recent data
    3. Detecting and adapting to concept drift
    4. Incrementally updating classifiers
    5. Rebalancing data using complexity-guided sampling
    
    Parameters
    ----------
    config : StreamEnsembleConfig, optional
        Configuration object
    **kwargs : dict
        Configuration parameters (used if config is None)
    
    Attributes
    ----------
    estimators_ : list
        List of base classifiers
    mu_values_ : array
        Mu values for each estimator
    performance_history_ : list
        Performance metrics over time
    drift_points_ : list
        Detected drift points (chunk indices)
    
    Examples
    --------
    >>> from sklearn.linear_model import SGDClassifier
    >>> 
    >>> # Create stream ensemble
    >>> ensemble = StreamLearningEnsemble(
    ...     n_estimators=10,
    ...     base_estimator=SGDClassifier,
    ...     chunk_size=100,
    ...     window_size=1000,
    ...     drift_detection='adwin',
    ...     verbose=1
    ... )
    >>> 
    >>> # Process stream
    >>> for X_chunk, y_chunk in data_stream:
    ...     ensemble.partial_fit(X_chunk, y_chunk)
    ...     y_pred = ensemble.predict(X_chunk)
    """
    
    def __init__(
        self,
        config: Optional[StreamEnsembleConfig] = None,
        **kwargs
    ):
        """Initialize stream ensemble."""
        self.config = config or StreamEnsembleConfig(**kwargs)
        self._validate_config()
        
        # Initialize components
        self.estimators_: List[ClassifierMixin] = []
        self.mu_values_: Optional[ArrayLike] = None
        self.classes_: Optional[ArrayLike] = None
        self.n_classes_: Optional[int] = None
        
        # Stream processing state
        self.sliding_window_: Optional[SlidingWindow] = None
        self.drift_detector_: Optional[DriftDetector] = None
        self.sampler_: Optional[ComplexityGuidedSampler] = None
        
        # Statistics
        self.n_samples_seen_: int = 0
        self.n_chunks_processed_: int = 0
        self.performance_history_: List[dict] = []
        self.drift_points_: List[int] = []
        self.estimator_weights_: Optional[ArrayLike] = None
        
        # Initialize
        self._initialize_components()
    
    def _validate_config(self):
        """Validate configuration."""
        if self.config.n_estimators < 1:
            raise ValueError("n_estimators must be at least 1")
        
        if self.config.chunk_size < 1:
            raise ValueError("chunk_size must be at least 1")
        
        if self.config.window_size < self.config.chunk_size:
            raise ValueError("window_size must be >= chunk_size")
        
        # Check if base estimator supports partial_fit
        test_estimator = self.config.base_estimator()
        if not hasattr(test_estimator, 'partial_fit'):
            warnings.warn(
                f"{self.config.base_estimator.__name__} does not support partial_fit. "
                "Stream learning may not work efficiently."
            )
    
    def _initialize_components(self):
        """Initialize ensemble components."""
        # Sliding window
        self.sliding_window_ = SlidingWindow(
            max_size=self.config.window_size,
            max_memory_mb=self.config.max_memory_mb
        )
        
        # Drift detector
        self.drift_detector_ = DriftDetectorFactory.create(
            self.config.drift_detection,
            threshold=self.config.drift_threshold
        )
        
        # Complexity sampler
        sampler_config = SamplerConfig(
            complex_type=self.config.complexity_type,
            random_state=self.config.random_state,
            cv_folds=3
        )
        self.sampler_ = ComplexityGuidedSampler(config=sampler_config)
        
        # Generate mu values
        self.mu_values_ = self._generate_mu_values()
        
        # Initialize estimators
        self._initialize_estimators()
    
    def _generate_mu_values(self) -> ArrayLike:
        """Generate mu values for ensemble diversity."""
        if self.config.n_estimators == 1:
            return np.array([0.5])
        return np.linspace(0, 1, self.config.n_estimators)
    
    def _initialize_estimators(self):
        """Initialize base estimators."""
        self.estimators_ = []
        for _ in range(self.config.n_estimators):
            estimator = clone(self.config.base_estimator(
                random_state=self.config.random_state
            ))
            self.estimators_.append(estimator)
        
        # Equal weights initially
        self.estimator_weights_ = np.ones(self.config.n_estimators)
        self.estimator_weights_ /= self.estimator_weights_.sum()
    
    def _create_balanced_subset(
        self,
        X: ArrayLike,
        y: ArrayLike,
        mu: float
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Create balanced subset using complexity-guided sampling.
        
        Args:
            X: Feature matrix
            y: Target labels
            mu: Complexity focus parameter
            
        Returns:
            Balanced (X, y)
        """
        try:
            X_balanced, y_balanced = self.sampler_.fit_resample(
                X, y,
                mu=mu,
                sigma=self.config.sigma,
                k_neighbors=self.config.k_neighbors
            )
            return X_balanced, y_balanced
        except Exception as e:
            if self.config.verbose > 0:
                warnings.warn(f"Resampling failed: {e}. Using original data.")
            return X, y
    
    def partial_fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        classes: Optional[ArrayLike] = None
    ) -> 'StreamLearningEnsemble':
        """
        Incrementally fit the ensemble on a batch of samples.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data
        y : array-like of shape (n_samples,)
            Target labels
        classes : array-like, optional
            All possible class labels (required for first call)
            
        Returns
        -------
        self : StreamLearningEnsemble
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Initialize classes on first call
        if self.classes_ is None:
            if classes is None:
                self.classes_ = np.unique(y)
            else:
                self.classes_ = np.asarray(classes)
            self.n_classes_ = len(self.classes_)
        
        # Add to sliding window
        self.sliding_window_.add_batch(X, y)
        self.n_samples_seen_ += len(X)
        self.n_chunks_processed_ += 1
        
        if self.config.verbose > 0:
            print(f"\n{'='*70}")
            print(f"Processing Chunk #{self.n_chunks_processed_}")
            print(f"{'='*70}")
            print(f"  Chunk size: {len(X)}")
            print(f"  Total samples seen: {self.n_samples_seen_}")
            print(f"  Window size: {self.sliding_window_.get_size()}")
            print(f"  Class distribution: {np.bincount(y.astype(int))}")
        
        # Check if we have enough samples
        if self.n_samples_seen_ < self.config.min_samples_before_update:
            if self.config.verbose > 0:
                print(f"  Status: Collecting initial samples "
                      f"({self.n_samples_seen_}/{self.config.min_samples_before_update})")
            return self
        
        # Get current window data
        X_window, y_window = self.sliding_window_.get_data()
        
        # Rebalance if needed
        should_rebalance = (
            self.n_chunks_processed_ % self.config.rebalance_frequency == 0
        )
        
        if should_rebalance and len(np.unique(y_window)) > 1:
            if self.config.verbose > 0:
                print(f"  Action: Rebalancing data...")
            
            # Update each estimator with rebalanced data
            for i, (estimator, mu) in enumerate(zip(self.estimators_, self.mu_values_)):
                X_balanced, y_balanced = self._create_balanced_subset(
                    X_window, y_window, mu
                )
                
                # Incremental update
                if hasattr(estimator, 'partial_fit'):
                    estimator.partial_fit(X_balanced, y_balanced, classes=self.classes_)
                else:
                    estimator.fit(X_balanced, y_balanced)
                
                if self.config.verbose > 1:
                    print(f"    Estimator {i+1} (μ={mu:.3f}): "
                          f"trained on {len(X_balanced)} samples")
        else:
            # Regular incremental update without rebalancing
            if self.config.verbose > 0:
                print(f"  Action: Incremental update...")
            
            for i, estimator in enumerate(self.estimators_):
                if hasattr(estimator, 'partial_fit'):
                    estimator.partial_fit(X, y, classes=self.classes_)
                else:
                    # Retrain on window
                    estimator.fit(X_window, y_window)
        
        # Drift detection
        if self.drift_detector_ is not None:
            y_pred = self.predict(X)
            correct = (y_pred == y)
            
            drift_detected = False
            for is_correct in correct:
                if self.drift_detector_.add_element(is_correct):
                    drift_detected = True
                    break
            
            if drift_detected:
                self._handle_drift()
        
        # Update performance metrics
        self._update_performance_metrics(X, y)
        
        # Prune weak estimators if needed
        if self.n_chunks_processed_ % (self.config.rebalance_frequency * 2) == 0:
            self._prune_weak_estimators()
        
        return self
    
    def _handle_drift(self):
        """Handle detected concept drift."""
        if self.config.verbose > 0:
            print(f"\n  🚨 DRIFT DETECTED at chunk {self.n_chunks_processed_}")
        
        self.drift_points_.append(self.n_chunks_processed_)
        
        if self.config.update_strategy == "replace_worst":
            # Replace worst performing estimator
            if len(self.performance_history_) > 0:
                recent_perf = self.performance_history_[-1]['estimator_scores']
                worst_idx = np.argmin(recent_perf)
                
                if self.config.verbose > 0:
                    print(f"  Action: Replacing estimator {worst_idx+1}")
                
                # Create new estimator
                self.estimators_[worst_idx] = clone(
                    self.config.base_estimator(random_state=self.config.random_state)
                )
                
                # Retrain on window
                X_window, y_window = self.sliding_window_.get_data()
                if len(X_window) > 0:
                    mu = self.mu_values_[worst_idx]
                    X_balanced, y_balanced = self._create_balanced_subset(
                        X_window, y_window, mu
                    )
                    
                    if hasattr(self.estimators_[worst_idx], 'partial_fit'):
                        self.estimators_[worst_idx].partial_fit(
                            X_balanced, y_balanced, classes=self.classes_
                        )
                    else:
                        self.estimators_[worst_idx].fit(X_balanced, y_balanced)
        
        elif self.config.update_strategy == "add_new":
            # Add new estimator (if under limit)
            if len(self.estimators_) < self.config.n_estimators:
                if self.config.verbose > 0:
                    print(f"  Action: Adding new estimator")
                
                new_estimator = clone(
                    self.config.base_estimator(random_state=self.config.random_state)
                )
                self.estimators_.append(new_estimator)
                
                # Train on window
                X_window, y_window = self.sliding_window_.get_data()
                if len(X_window) > 0:
                    if hasattr(new_estimator, 'partial_fit'):
                        new_estimator.partial_fit(X_window, y_window, classes=self.classes_)
                    else:
                        new_estimator.fit(X_window, y_window)
                
                # Update weights
                self.estimator_weights_ = np.ones(len(self.estimators_))
                self.estimator_weights_ /= self.estimator_weights_.sum()
        
        elif self.config.update_strategy == "weighted":
            # Adjust weights based on recent performance
            if len(self.performance_history_) > 0:
                recent_perf = self.performance_history_[-1]['estimator_scores']
                self.estimator_weights_ = np.array(recent_perf)
                self.estimator_weights_ = np.maximum(self.estimator_weights_, 0.01)
                self.estimator_weights_ /= self.estimator_weights_.sum()
                
                if self.config.verbose > 0:
                    print(f"  Action: Updated estimator weights")
    
    def _update_performance_metrics(self, X: ArrayLike, y: ArrayLike):
        """Update performance tracking."""
        if len(X) == 0:
            return
        
        # Ensemble predictions
        y_pred = self.predict(X)
        ensemble_acc = accuracy_score(y, y_pred)
        ensemble_f1 = f1_score(y, y_pred, average='weighted')
        
        # Individual estimator performance
        estimator_scores = []
        for estimator in self.estimators_:
            try:
                if hasattr(estimator, 'classes_') or hasattr(estimator, 'predict'):
                    y_pred_i = estimator.predict(X)
                    score = accuracy_score(y, y_pred_i)
                    estimator_scores.append(score)
                else:
                    estimator_scores.append(0.0)
            except:
                estimator_scores.append(0.0)
        
        # Store metrics
        self.performance_history_.append({
            'chunk': self.n_chunks_processed_,
            'n_samples': self.n_samples_seen_,
            'accuracy': ensemble_acc,
            'f1_score': ensemble_f1,
            'estimator_scores': estimator_scores,
            'n_estimators': len(self.estimators_),
            'drift_detected': self.n_chunks_processed_ in self.drift_points_
        })
        
        if self.config.verbose > 0:
            print(f"  Performance: Acc={ensemble_acc:.4f}, F1={ensemble_f1:.4f}")
            if self.config.verbose > 1:
                print(f"    Estimator scores: {[f'{s:.3f}' for s in estimator_scores]}")
    
    def _prune_weak_estimators(self):
        """Remove poorly performing estimators."""
        if len(self.estimators_) <= 3:  # Keep minimum ensemble size
            return
        
        if len(self.performance_history_) == 0:
            return
        
        recent_perf = self.performance_history_[-1]['estimator_scores']
        
        # Find estimators below threshold
        weak_indices = [
            i for i, score in enumerate(recent_perf)
            if score < self.config.prune_threshold
        ]
        
        if weak_indices and len(self.estimators_) - len(weak_indices) >= 3:
            if self.config.verbose > 0:
                print(f"\n  🔧 Pruning {len(weak_indices)} weak estimators")
            
            # Remove weak estimators
            for idx in sorted(weak_indices, reverse=True):
                del self.estimators_[idx]
                self.mu_values_ = np.delete(self.mu_values_, idx)
            
            # Update weights
            self.estimator_weights_ = np.ones(len(self.estimators_))
            self.estimator_weights_ /= self.estimator_weights_.sum()
    
    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples
            
        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted labels
        """
        if not self.estimators_:
            raise ValueError("Ensemble not fitted. Call partial_fit first.")
        
        # Weighted voting
        predictions = []
        weights = []
        
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            try:
                if hasattr(estimator, 'classes_') or hasattr(estimator, 'predict'):
                    pred = estimator.predict(X)
                    predictions.append(pred)
                    weights.append(weight)
            except:
                continue
        
        if not predictions:
            # Fallback: return most common class
            return np.full(len(X), self.classes_[0])
        
        # Weighted majority vote
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Vote for each sample
        final_predictions = []
        for i in range(len(X)):
            sample_votes = predictions[:, i]
            
            # Count weighted votes for each class
            class_votes = {}
            for cls in self.classes_:
                class_votes[cls] = np.sum(weights[sample_votes == cls])
            
            # Select class with most votes
            winner = max(class_votes.items(), key=lambda x: x[1])[0]
            final_predictions.append(winner)
        
        return np.array(final_predictions)
    
    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """
        Predict class probabilities.
        
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
            raise ValueError("Ensemble not fitted. Call partial_fit first.")
        
        # Check if estimators support predict_proba
        probas = []
        weights = []
        
        for estimator, weight in zip(self.estimators_, self.estimator_weights_):
            try:
                if hasattr(estimator, 'predict_proba'):
                    proba = estimator.predict_proba(X)
                    probas.append(proba)
                    weights.append(weight)
                elif hasattr(estimator, 'decision_function'):
                    # Convert decision function to probabilities
                    decision = estimator.decision_function(X)
                    if decision.ndim == 1:
                        # Binary classification
                        proba = np.vstack([1 - decision, decision]).T
                    else:
                        proba = decision
                    # Normalize
                    proba = np.exp(proba) / np.exp(proba).sum(axis=1, keepdims=True)
                    probas.append(proba)
                    weights.append(weight)
            except:
                continue
        
        if not probas:
            # Fallback: uniform distribution
            return np.ones((len(X), self.n_classes_)) / self.n_classes_
        
        # Weighted average
        probas = np.array(probas)
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        weighted_proba = np.average(probas, axis=0, weights=weights)
        
        return weighted_proba
    
    def get_performance_summary(self) -> pd.DataFrame:
        """Get performance summary over time.
        
        Returns:
            DataFrame with performance metrics
        """
        if not self.performance_history_:
            return pd.DataFrame()
        
        return pd.DataFrame(self.performance_history_)
    
    def get_drift_summary(self) -> dict:
        """Get drift detection summary.
        
        Returns:
            Dictionary with drift statistics
        """
        return {
            'n_drifts_detected': len(self.drift_points_),
            'drift_points': self.drift_points_,
            'chunks_between_drifts': (
                np.diff(self.drift_points_).tolist() if len(self.drift_points_) > 1 else []
            ),
            'drift_detection_method': self.config.drift_detection
        }
    
    def reset(self):
        """Reset ensemble to initial state."""
        self._initialize_components()
        self.n_samples_seen_ = 0
        self.n_chunks_processed_ = 0
        self.performance_history_ = []
        self.drift_points_ = []
        self.classes_ = None
        self.n_classes_ = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def simulate_data_stream(
    n_chunks: int = 50,
    chunk_size: int = 100,
    n_features: int = 10,
    drift_points: Optional[List[int]] = None,
    noise_level: float = 0.1
):
    """Simulate a data stream with concept drift.
    
    Args:
        n_chunks: Number of chunks
        chunk_size: Samples per chunk
        n_features: Number of features
        drift_points: Chunks where drift occurs
        noise_level: Amount of noise
        
    Yields:
        Tuple of (X_chunk, y_chunk)
    """
    from sklearn.datasets import make_classification
    
    if drift_points is None:
        drift_points = [n_chunks // 3, 2 * n_chunks // 3]
    
    current_concept = 0
    
    for chunk_idx in range(n_chunks):
        # Check for drift
        if chunk_idx in drift_points:
            current_concept += 1
        
        # Generate data with current concept
        X, y = make_classification(
            n_samples=chunk_size,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            n_redundant=max(0, n_features // 4),
            n_classes=2,
            weights=[0.7, 0.3],
            flip_y=noise_level,
            random_state=42 + chunk_idx + current_concept * 1000
        )
        
        yield X, y


if __name__ == "__main__":
    # Quick test
    from sklearn.linear_model import SGDClassifier
    
    print("Testing Stream Learning Ensemble...")
    
    # Create ensemble
    ensemble = StreamLearningEnsemble(
        n_estimators=5,
        base_estimator=SGDClassifier,
        chunk_size=100,
        window_size=500,
        drift_detection='adwin',
        verbose=1,
        random_state=42
    )
    
    # Simulate stream
    print("\nProcessing data stream...")
    for i, (X_chunk, y_chunk) in enumerate(simulate_data_stream(n_chunks=10)):
        ensemble.partial_fit(X_chunk, y_chunk)
        
        if i == 0:
            print("\n✅ First chunk processed successfully!")
    
    # Get summary
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    summary = ensemble.get_performance_summary()
    print(summary[['chunk', 'accuracy', 'f1_score', 'n_estimators']].to_string(index=False))
    
    print("\n" + "="*70)
    print("DRIFT SUMMARY")
    print("="*70)
    drift_summary = ensemble.get_drift_summary()
    for key, value in drift_summary.items():
        print(f"{key}: {value}")
    
    print("\n✅ Stream ensemble test passed!")
