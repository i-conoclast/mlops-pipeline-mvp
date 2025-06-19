"""
MLflow experiment tracking for model training.
"""
import os
import tempfile
from typing import Dict, Any

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split

from src.models.simple_classifier import SimpleClassifier


class MLflowTrainer:
    def __init__(self, experiment_name: str = "simple_classifier"):
        self.experiment_name = experiment_name
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
        # Set tracking URI (use local file store for PoC)
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Create or get experiment
        try:
            experiment_id = mlflow.create_experiment(self.experiment_name)
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(experiment_id=experiment_id)
    
    def train_and_log(
        self, 
        n_estimators: int = 100,
        n_samples: int = 1000,
        n_features: int = 20,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> str:
        """Train model and log to MLflow."""
        
        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("n_samples", n_samples)
            mlflow.log_param("n_features", n_features)
            mlflow.log_param("test_size", test_size)
            mlflow.log_param("random_state", random_state)
            
            # Create and train model
            classifier = SimpleClassifier(
                n_estimators=n_estimators, 
                random_state=random_state
            )
            
            # Generate data
            X, y = classifier.generate_sample_data(
                n_samples=n_samples, 
                n_features=n_features
            )
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Train model
            classifier.train(X_train, y_train)
            
            # Evaluate model
            train_results = classifier.evaluate(X_train, y_train)
            test_results = classifier.evaluate(X_test, y_test)
            
            # Log metrics
            mlflow.log_metric("train_accuracy", train_results["accuracy"])
            mlflow.log_metric("test_accuracy", test_results["accuracy"])
            mlflow.log_metric("train_f1", train_results["classification_report"]["macro avg"]["f1-score"])
            mlflow.log_metric("test_f1", test_results["classification_report"]["macro avg"]["f1-score"])
            
            # Log model
            mlflow.sklearn.log_model(
                classifier.model,
                "model",
                registered_model_name="SimpleClassifier"
            )
            
            # Log additional artifacts
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(f"Training completed successfully\n")
                f.write(f"Train accuracy: {train_results['accuracy']:.4f}\n")
                f.write(f"Test accuracy: {test_results['accuracy']:.4f}\n")
                f.write(f"Model type: {type(classifier.model).__name__}\n")
                temp_path = f.name
            
            mlflow.log_artifact(temp_path, "reports")
            os.unlink(temp_path)
            
            print(f"✅ Experiment logged with run_id: {run.info.run_id}")
            print(f"📊 Test accuracy: {test_results['accuracy']:.4f}")
            
            return run.info.run_id
    
    def get_best_model(self) -> Dict[str, Any]:
        """Get the best model from the experiment."""
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["metrics.test_accuracy DESC"],
            max_results=1
        )
        
        if runs.empty:
            raise ValueError("No runs found in experiment")
        
        best_run = runs.iloc[0]
        return {
            "run_id": best_run["run_id"],
            "test_accuracy": best_run["metrics.test_accuracy"],
            "model_uri": f"runs:/{best_run['run_id']}/model"
        }


if __name__ == "__main__":
    # Demo usage
    trainer = MLflowTrainer()
    
    # Run multiple experiments with different parameters
    for n_estimators in [50, 100, 200]:
        print(f"\n🚀 Training with n_estimators={n_estimators}")
        trainer.train_and_log(n_estimators=n_estimators)
    
    # Get best model
    best_model_info = trainer.get_best_model()
    print(f"\n🏆 Best model: {best_model_info}")
    print(f"📁 MLflow UI: mlflow ui --backend-store-uri file:./mlruns")