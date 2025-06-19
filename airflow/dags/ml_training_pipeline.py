"""
Airflow DAG for ML training pipeline.
"""
from datetime import datetime, timedelta
import sys
import os

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

# Add src to Python path
sys.path.append('/opt/airflow/src')

from src.experiments.mlflow_trainer import MLflowTrainer
from src.data.data_processor import DataProcessor


# Default arguments
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 6, 19),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Create DAG
dag = DAG(
    'ml_training_pipeline',
    default_args=default_args,
    description='ML model training pipeline with MLflow tracking',
    schedule_interval=timedelta(days=1),
    catchup=False,
    tags=['ml', 'training', 'mlflow'],
)


def validate_environment(**context):
    """Validate that required environment is available."""
    print("🔍 Validating environment...")
    
    # Check if MLflow is available
    try:
        import mlflow
        print(f"✅ MLflow version: {mlflow.__version__}")
    except ImportError:
        raise Exception("❌ MLflow not available")
    
    # Check if scikit-learn is available
    try:
        import sklearn
        print(f"✅ Scikit-learn version: {sklearn.__version__}")
    except ImportError:
        raise Exception("❌ Scikit-learn not available")
    
    print("✅ Environment validation completed")


def preprocess_data(**context):
    """Data preprocessing step."""
    print("📊 Starting data preprocessing...")
    
    processor = DataProcessor()
    
    # Generate sample data
    X, y = processor.generate_classification_data(
        n_samples=2000,
        n_features=25
    )
    
    # Validate data
    data_info = processor.validate_data(X, y)
    
    # Store preprocessing results in XCom
    context['task_instance'].xcom_push(
        key='data_info',
        value=data_info
    )
    
    print(f"✅ Data preprocessing completed: {data_info}")
    return data_info


def train_model(**context):
    """Model training step with MLflow tracking."""
    print("🚀 Starting model training...")
    
    # Get data info from previous task
    data_info = context['task_instance'].xcom_pull(
        task_ids='preprocess_data',
        key='data_info'
    )
    
    trainer = MLflowTrainer(experiment_name="airflow_training")
    
    # Train model with parameters from data preprocessing
    run_id = trainer.train_and_log(
        n_estimators=100,
        n_samples=data_info['n_samples'],
        n_features=data_info['n_features'],
        test_size=0.2,
        random_state=42
    )
    
    # Store run_id in XCom
    context['task_instance'].xcom_push(
        key='mlflow_run_id',
        value=run_id
    )
    
    print(f"✅ Model training completed. Run ID: {run_id}")
    return run_id


def validate_model(**context):
    """Model validation step."""
    print("🔍 Starting model validation...")
    
    # Get run_id from previous task
    run_id = context['task_instance'].xcom_pull(
        task_ids='train_model',
        key='mlflow_run_id'
    )
    
    trainer = MLflowTrainer(experiment_name="airflow_training")
    
    # Get model info
    best_model_info = trainer.get_best_model()
    
    # Validation logic (simple threshold check)
    min_accuracy = 0.8
    if best_model_info['test_accuracy'] >= min_accuracy:
        print(f"✅ Model validation passed: {best_model_info['test_accuracy']:.4f} >= {min_accuracy}")
        
        # Store validation results
        context['task_instance'].xcom_push(
            key='validation_passed',
            value=True
        )
        context['task_instance'].xcom_push(
            key='model_info',
            value=best_model_info
        )
        
        return best_model_info
    else:
        raise Exception(f"❌ Model validation failed: {best_model_info['test_accuracy']:.4f} < {min_accuracy}")


def register_model(**context):
    """Model registration step."""
    print("📝 Starting model registration...")
    
    # Get validation results
    validation_passed = context['task_instance'].xcom_pull(
        task_ids='validate_model',
        key='validation_passed'
    )
    
    model_info = context['task_instance'].xcom_pull(
        task_ids='validate_model',
        key='model_info'
    )
    
    if validation_passed:
        print(f"✅ Model registered successfully:")
        print(f"   - Run ID: {model_info['run_id']}")
        print(f"   - Test Accuracy: {model_info['test_accuracy']:.4f}")
        print(f"   - Model URI: {model_info['model_uri']}")
        
        # In a real scenario, you would promote the model to Production stage
        # mlflow.tracking.MlflowClient().transition_model_version_stage(...)
        
        return model_info
    else:
        raise Exception("❌ Cannot register model: validation failed")


# Define tasks
validate_env_task = PythonOperator(
    task_id='validate_environment',
    python_callable=validate_environment,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_model',
    python_callable=validate_model,
    dag=dag,
)

register_task = PythonOperator(
    task_id='register_model',
    python_callable=register_model,
    dag=dag,
)

# Check MLflow UI accessibility
check_mlflow_task = BashOperator(
    task_id='check_mlflow_ui',
    bash_command='echo "🌐 MLflow UI should be available at: http://localhost:5000"',
    dag=dag,
)

# Define task dependencies
validate_env_task >> preprocess_task >> train_task >> validate_task >> register_task >> check_mlflow_task