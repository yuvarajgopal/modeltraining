from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from main_pipeline import my_pipeline
import os

# Auth + workspace
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

print("SUB:", os.getenv("AZURE_SUBSCRIPTION_ID"))
print("RG:", os.getenv("AZURE_RESOURCE_GROUP"))
print("WS:", os.getenv("AZURE_WORKSPACE_NAME"))


credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# (Optional) register your environment
custom_env = Environment(
    name="custom-pipeline-env",
    description="My conda environment",
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
)

ml_client.environments.create_or_update(custom_env)

# Submit pipeline
pipeline_job = my_pipeline()  # this should not rely on ml_client inside
ml_client.jobs.create_or_update(pipeline_job)

