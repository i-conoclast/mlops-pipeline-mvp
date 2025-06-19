"""Tests for SimpleClassifier."""
import tempfile
import pytest
import numpy as np
from src.models.simple_classifier import SimpleClassifier


def test_simple_classifier_initialization():
    """Test SimpleClassifier initialization."""
    classifier = SimpleClassifier()
    assert classifier.model is not None
    assert classifier.is_trained is False


def test_generate_sample_data():
    """Test sample data generation."""
    classifier = SimpleClassifier()
    X, y = classifier.generate_sample_data(n_samples=100, n_features=10)
    
    assert X.shape == (100, 10)
    assert y.shape == (100,)
    assert len(np.unique(y)) == 2


def test_train_and_predict():
    """Test model training and prediction."""
    classifier = SimpleClassifier()
    X, y = classifier.generate_sample_data(n_samples=100)
    
    # Test training
    classifier.train(X, y)
    assert classifier.is_trained is True
    
    # Test prediction
    predictions = classifier.predict(X)
    assert len(predictions) == len(y)
    assert all(pred in [0, 1] for pred in predictions)


def test_predict_before_training():
    """Test that prediction fails before training."""
    classifier = SimpleClassifier()
    X, _ = classifier.generate_sample_data(n_samples=10)
    
    with pytest.raises(ValueError, match="Model must be trained"):
        classifier.predict(X)


def test_evaluate():
    """Test model evaluation."""
    classifier = SimpleClassifier()
    X, y = classifier.generate_sample_data(n_samples=100)
    
    classifier.train(X, y)
    results = classifier.evaluate(X, y)
    
    assert "accuracy" in results
    assert "classification_report" in results
    assert 0 <= results["accuracy"] <= 1


def test_save_and_load_model():
    """Test model saving and loading."""
    classifier = SimpleClassifier()
    X, y = classifier.generate_sample_data(n_samples=100)
    
    # Train model
    classifier.train(X, y)
    original_predictions = classifier.predict(X)
    
    # Save model
    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp:
        classifier.save_model(tmp.name)
        
        # Load model in new instance
        new_classifier = SimpleClassifier()
        new_classifier.load_model(tmp.name)
        
        # Test loaded model
        loaded_predictions = new_classifier.predict(X)
        np.testing.assert_array_equal(original_predictions, loaded_predictions)


def test_save_before_training():
    """Test that saving fails before training."""
    classifier = SimpleClassifier()
    
    with tempfile.NamedTemporaryFile(suffix='.joblib') as tmp:
        with pytest.raises(ValueError, match="Model must be trained"):
            classifier.save_model(tmp.name)