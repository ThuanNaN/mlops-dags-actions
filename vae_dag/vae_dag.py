from datetime import datetime, timedelta
import yaml
import os
import torch
from pathlib import Path
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from loader import ColorizationDataset
from model import VAE_CNN_Improved
from train import trainer, seed_everything

def load_config():
    with open("./config/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    return config

config = load_config()
seed_everything(config["seed"])

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

def create_temp_dir():
    cuda_available = True if torch.cuda.is_available() else False
    print(f"Using GPU: {cuda_available}")
    path_save = Path("./vae_tmp_dir") / "v0.1"
    path_save.mkdir(parents=True, exist_ok=True)
    if not path_save.exists():
        raise FileNotFoundError(f"Failed to create directory: {path_save}")


def fetch_data():
    config = load_config()
    db_params = {
        "host": config["db_host"],
        "database": config["database"],   
        "user": config["user"],      
        "password": config["password"],  
        "port": config["port"]    
    }
    dataset_name = config["dataset_name"]
    dataset = ColorizationDataset(dataset_name, db_params)
    return dataset

def create_model():
    config = load_config()
    model = VAE_CNN_Improved(latent_dim=config["latent_dim"])
    return model

def train_model():
    config = load_config()
    dataset = fetch_data()
    train_size = int(len(dataset) * config["train_ratio"])
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)
    loss_history, latent_history = trainer(train_loader, val_loader, config)
    print("Training completed!")
    return loss_history, latent_history


with DAG(
    'vae_cnn_training_pipeline',
    default_args=default_args,
    description='A DAG to train and validate VAE_CNN model',
    schedule_interval=timedelta(days=1),
    ) as dag:

    create_temp_dir_task = PythonOperator(
        task_id='create_temp_dir',
        python_callable=create_temp_dir,
        dag=dag,
    )

    train_model_task = PythonOperator(
        task_id='train_model',
        python_callable=train_model,
        dag=dag,
    )


# Define task dependencies
create_temp_dir_task >> train_model_task