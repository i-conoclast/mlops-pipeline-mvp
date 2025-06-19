"""
Data processing utilities for ML pipeline.
"""
import numpy as np
from sklearn.datasets import make_classification
from typing import Tuple, Dict, Any


class DataProcessor:
    def __init__(self):
        pass
    
    def generate_classification_data(
        self, 
        n_samples: int = 1000, 
        n_features: int = 20,
        n_informative: int = None,
        n_redundant: int = None,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic classification data."""
        
        if n_informative is None:
            n_informative = min(15, n_features)
        if n_redundant is None:
            n_redundant = max(0, n_features - n_informative)
        
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_informative,
            n_redundant=n_redundant,
            n_clusters_per_class=1,
            random_state=random_state
        )
        
        return X, y
    
    def validate_data(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Validate data and return information."""
        
        # Basic validation
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        if X.shape[0] == 0:
            raise ValueError("Dataset cannot be empty")
        
        # Calculate statistics
        data_info = {
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y)),
            'class_distribution': {int(cls): int(count) for cls, count in zip(*np.unique(y, return_counts=True))},
            'feature_means': X.mean(axis=0).tolist(),
            'feature_stds': X.std(axis=0).tolist(),
            'missing_values': int(np.isnan(X).sum()),
        }
        
        # Data quality checks
        if data_info['missing_values'] > 0:
            print(f"⚠️  Warning: {data_info['missing_values']} missing values found")
        
        if data_info['n_samples'] < 100:
            print(f"⚠️  Warning: Small dataset ({data_info['n_samples']} samples)")
        
        # Check class balance
        class_counts = list(data_info['class_distribution'].values())
        if max(class_counts) / min(class_counts) > 3:
            print("⚠️  Warning: Imbalanced classes detected")
        
        print(f"✅ Data validation completed:")
        print(f"   - Samples: {data_info['n_samples']}")
        print(f"   - Features: {data_info['n_features']}")
        print(f"   - Classes: {data_info['n_classes']}")
        print(f"   - Class distribution: {data_info['class_distribution']}")
        
        return data_info
    
    def preprocess_features(self, X: np.ndarray) -> np.ndarray:
        """Basic feature preprocessing."""
        
        # Handle missing values (replace with mean)
        if np.isnan(X).any():
            X = np.where(np.isnan(X), np.nanmean(X, axis=0), X)
        
        # Basic normalization (z-score)
        X_normalized = (X - X.mean(axis=0)) / X.std(axis=0)
        
        return X_normalized


if __name__ == "__main__":
    # Demo usage
    processor = DataProcessor()
    
    # Generate data
    X, y = processor.generate_classification_data(n_samples=500, n_features=10)
    
    # Validate data
    data_info = processor.validate_data(X, y)
    
    # Preprocess features
    X_processed = processor.preprocess_features(X)
    
    print(f"\n📊 Original data shape: {X.shape}")
    print(f"📊 Processed data shape: {X_processed.shape}")
    print(f"📊 Data info: {data_info}")