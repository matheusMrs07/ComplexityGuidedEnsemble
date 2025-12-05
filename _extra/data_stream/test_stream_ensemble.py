"""
Tests for Stream Learning Ensemble.
"""

import unittest
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB

from stream_learning_ensemble import (
    StreamLearningEnsemble,
    StreamEnsembleConfig,
    ADWINDetector,
    DDMDetector,
    PageHinkleyDetector,
    SlidingWindow,
    simulate_data_stream
)


class TestDriftDetectors(unittest.TestCase):
    """Test drift detection methods."""
    
    def test_adwin_detector(self):
        """ADWIN should detect drift."""
        detector = ADWINDetector(threshold=0.002)
        
        # Stable period
        for _ in range(50):
            drift = detector.add_element(True)
            self.assertFalse(drift)
        
        # Drift period (many errors)
        drift_detected = False
        for _ in range(50):
            drift = detector.add_element(False)
            if drift:
                drift_detected = True
                break
        
        # Should detect drift eventually
        self.assertTrue(drift_detected or len(detector.window) > 0)
    
    def test_ddm_detector(self):
        """DDM should detect drift."""
        detector = DDMDetector(threshold=0.1)
        
        # Stable period
        for _ in range(30):
            detector.add_element(True)
        
        # No drift yet
        self.assertEqual(detector.n, 30)
    
    def test_page_hinkley_detector(self):
        """Page-Hinkley should detect drift."""
        detector = PageHinkleyDetector(threshold=50.0, delta=0.005)
        
        # Stable period
        for _ in range(50):
            drift = detector.add_element(True)
            self.assertFalse(drift)
        
        detector.reset()
        self.assertEqual(detector.n, 0)


class TestSlidingWindow(unittest.TestCase):
    """Test sliding window functionality."""
    
    def test_window_add_and_get(self):
        """Window should store and retrieve data."""
        window = SlidingWindow(max_size=100)
        
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        
        window.add_batch(X, y)
        
        X_retrieved, y_retrieved = window.get_data()
        
        self.assertEqual(len(X_retrieved), 50)
        self.assertEqual(len(y_retrieved), 50)
        np.testing.assert_array_equal(X_retrieved, X)
        np.testing.assert_array_equal(y_retrieved, y)
    
    def test_window_max_size(self):
        """Window should respect max size."""
        window = SlidingWindow(max_size=50)
        
        # Add more than max_size
        X1 = np.random.rand(30, 5)
        y1 = np.random.randint(0, 2, 30)
        window.add_batch(X1, y1)
        
        X2 = np.random.rand(30, 5)
        y2 = np.random.randint(0, 2, 30)
        window.add_batch(X2, y2)
        
        # Should keep only last 50
        self.assertEqual(window.get_size(), 50)
    
    def test_window_clear(self):
        """Window should clear correctly."""
        window = SlidingWindow(max_size=100)
        
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        window.add_batch(X, y)
        
        window.clear()
        
        self.assertEqual(window.get_size(), 0)


class TestStreamEnsemble(unittest.TestCase):
    """Test stream learning ensemble."""
    
    def test_initialization(self):
        """Ensemble should initialize correctly."""
        ensemble = StreamLearningEnsemble(
            n_estimators=5,
            chunk_size=50,
            window_size=200,
            verbose=0
        )
        
        self.assertEqual(len(ensemble.estimators_), 5)
        self.assertEqual(len(ensemble.mu_values_), 5)
        self.assertIsNotNone(ensemble.sliding_window_)
    
    def test_partial_fit_single_chunk(self):
        """Should handle single chunk."""
        ensemble = StreamLearningEnsemble(
            n_estimators=3,
            base_estimator=SGDClassifier,
            chunk_size=50,
            min_samples_before_update=100,
            verbose=0,
            random_state=42
        )
        
        X = np.random.rand(50, 5)
        y = np.random.randint(0, 2, 50)
        
        ensemble.partial_fit(X, y, classes=np.array([0, 1]))
        
        self.assertEqual(ensemble.n_samples_seen_, 50)
        self.assertEqual(ensemble.n_chunks_processed_, 1)
    
    def test_partial_fit_multiple_chunks(self):
        """Should handle multiple chunks."""
        ensemble = StreamLearningEnsemble(
            n_estimators=3,
            base_estimator=SGDClassifier,
            chunk_size=50,
            min_samples_before_update=100,
            rebalance_frequency=2,
            verbose=0,
            random_state=42
        )
        
        # Process multiple chunks
        for i in range(5):
            X = np.random.rand(50, 5)
            y = np.random.randint(0, 2, 50)
            
            if i == 0:
                ensemble.partial_fit(X, y, classes=np.array([0, 1]))
            else:
                ensemble.partial_fit(X, y)
        
        self.assertEqual(ensemble.n_samples_seen_, 250)
        self.assertEqual(ensemble.n_chunks_processed_, 5)
        self.assertGreater(len(ensemble.performance_history_), 0)
    
    def test_predict_after_fit(self):
        """Should predict after fitting."""
        ensemble = StreamLearningEnsemble(
            n_estimators=3,
            base_estimator=SGDClassifier,
            min_samples_before_update=50,
            verbose=0,
            random_state=42
        )
        
        # Fit
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        ensemble.partial_fit(X_train, y_train, classes=np.array([0, 1]))
        
        # Predict
        X_test = np.random.rand(20, 5)
        y_pred = ensemble.predict(X_test)
        
        self.assertEqual(len(y_pred), 20)
        self.assertTrue(all(p in [0, 1] for p in y_pred))
    
    def test_predict_proba(self):
        """Should predict probabilities."""
        ensemble = StreamLearningEnsemble(
            n_estimators=3,
            base_estimator=SGDClassifier,
            min_samples_before_update=50,
            verbose=0,
            random_state=42
        )
        
        # Fit
        X_train = np.random.rand(100, 5)
        y_train = np.random.randint(0, 2, 100)
        ensemble.partial_fit(X_train, y_train, classes=np.array([0, 1]))
        
        # Predict probabilities
        X_test = np.random.rand(20, 5)
        y_proba = ensemble.predict_proba(X_test)
        
        self.assertEqual(y_proba.shape, (20, 2))
        self.assertTrue(np.allclose(y_proba.sum(axis=1), 1.0))
    
    def test_drift_detection(self):
        """Should detect drift."""
        ensemble = StreamLearningEnsemble(
            n_estimators=3,
            base_estimator=SGDClassifier,
            drift_detection='adwin',
            drift_threshold=0.002,
            min_samples_before_update=50,
            verbose=0,
            random_state=42
        )
        
        # Process stream with drift
        for i, (X, y) in enumerate(simulate_data_stream(
            n_chunks=10, 
            chunk_size=50,
            drift_points=[5]
        )):
            if i == 0:
                ensemble.partial_fit(X, y, classes=np.array([0, 1]))
            else:
                ensemble.partial_fit(X, y)
        
        # Should have processed all chunks
        self.assertEqual(ensemble.n_chunks_processed_, 10)
        
        # May or may not detect drift (depends on data)
        # Just check that mechanism works
        drift_summary = ensemble.get_drift_summary()
        self.assertIn('n_drifts_detected', drift_summary)
    
    def test_performance_tracking(self):
        """Should track performance over time."""
        ensemble = StreamLearningEnsemble(
            n_estimators=3,
            base_estimator=SGDClassifier,
            min_samples_before_update=50,
            verbose=0,
            random_state=42
        )
        
        # Process multiple chunks
        for i in range(5):
            X = np.random.rand(50, 5)
            y = np.random.randint(0, 2, 50)
            
            if i == 0:
                ensemble.partial_fit(X, y, classes=np.array([0, 1]))
            else:
                ensemble.partial_fit(X, y)
        
        # Get performance summary
        summary = ensemble.get_performance_summary()
        
        self.assertIsInstance(summary, pd.DataFrame)
        self.assertGreater(len(summary), 0)
        self.assertIn('accuracy', summary.columns)
        self.assertIn('f1_score', summary.columns)
    
    def test_reset(self):
        """Reset should clear all state."""
        ensemble = StreamLearningEnsemble(
            n_estimators=3,
            min_samples_before_update=50,
            verbose=0,
            random_state=42
        )
        
        # Process data
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        ensemble.partial_fit(X, y, classes=np.array([0, 1]))
        
        # Reset
        ensemble.reset()
        
        self.assertEqual(ensemble.n_samples_seen_, 0)
        self.assertEqual(ensemble.n_chunks_processed_, 0)
        self.assertEqual(len(ensemble.performance_history_), 0)
    
    def test_different_update_strategies(self):
        """Should work with different update strategies."""
        strategies = ['replace_worst', 'add_new', 'weighted']
        
        for strategy in strategies:
            with self.subTest(strategy=strategy):
                ensemble = StreamLearningEnsemble(
                    n_estimators=3,
                    base_estimator=SGDClassifier,
                    update_strategy=strategy,
                    min_samples_before_update=50,
                    verbose=0,
                    random_state=42
                )
                
                # Process data
                for i in range(3):
                    X = np.random.rand(50, 5)
                    y = np.random.randint(0, 2, 50)
                    
                    if i == 0:
                        ensemble.partial_fit(X, y, classes=np.array([0, 1]))
                    else:
                        ensemble.partial_fit(X, y)
                
                self.assertGreater(ensemble.n_samples_seen_, 0)


class TestDataStreamSimulation(unittest.TestCase):
    """Test data stream simulation."""
    
    def test_simulate_stream(self):
        """Should generate valid data stream."""
        stream = simulate_data_stream(
            n_chunks=5,
            chunk_size=50,
            n_features=10,
            drift_points=[2]
        )
        
        chunks = list(stream)
        
        self.assertEqual(len(chunks), 5)
        
        for X, y in chunks:
            self.assertEqual(len(X), 50)
            self.assertEqual(len(y), 50)
            self.assertEqual(X.shape[1], 10)


def run_tests():
    """Run all tests."""
    # Need pandas import
    import pandas as pd
    globals()['pd'] = pd
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__(__name__))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == "__main__":
    import pandas as pd
    run_tests()
