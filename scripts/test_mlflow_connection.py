"""
Test-Connection Script for Petrodora-AI: MLflow/DagsHub Verification
Purpose: Confirms that the local training environment can log metrics and parameters to DagsHub.
Author: Antigravity Assistant
"""

import os
from dotenv import load_dotenv
import mlflow
import dagshub

# Configuração de carregamento das variáveis ambientais
load_dotenv()

def initialize_dagshub() -> None:
    """
    Initializes DagsHub integration and points MLflow to the remote tracking server.
    Ensures parameters set in .env are respected.
    """
    repo_owner: str = "RichardMan13"
    repo_name: str = "Petrodora-AI"
    
    # Init dagshub: This handles setting the MLFLOW_TRACKING_URI automatically
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    
    # Verify the tracking URI matches what we expect from .env
    tracking_uri: str = mlflow.get_tracking_uri()
    print(f"MLflow Tracking URI: {tracking_uri}")

def run_test_experiment() -> None:
    """
    Starts a test run to ensure metrics are visible on the DagsHub experiments page.
    """
    print("\n[INFO] Starting test run on DagsHub...")
    
    with mlflow.start_run(run_name="v0.1-DagsHub-ConnCheck"):
        # Log basic metadata
        mlflow.log_param("test_id", "001")
        mlflow.log_param("model_base", "phi-3-mini-4k")
        
        # Log a dummy metric (1.0 = Success)
        mlflow.log_metric("connection_status", 1.0)
        
        print("[SUCCESS] Metadata and metrics logged successfully!")
        print("[ACTION] Check your DagsHub repository under 'Experiments' tab.")

if __name__ == "__main__":
    try:
        initialize_dagshub()
        run_test_experiment()
    except Exception as e:
        print(f"[ERROR] Failed to connect to DagsHub: {str(e)}")
        print("[HINT] Ensure you ran 'dagshub login' and your .env is configured correctly.")
