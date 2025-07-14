import os
from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from main_pipeline import my_pipeline

# Load workspace configuration from env vars
subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

print("SUB:", subscription_id)
print("RG :", resource_group)
print("WS :", workspace_name)

# Authenticate to Azure ML
credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# Define and register environment (once per name-version)
custom_env = Environment(
    name="custom-pipeline-env",
    description="My pipeline environment",
    conda_file="environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
)

ml_client.environments.create_or_update(custom_env)

# Submit the pipeline job
pipeline_job = my_pipeline()
submitted_job = ml_client.jobs.create_or_update(pipeline_job)

# Print portal link
print(f"✅ Submitted pipeline: {submitted_job.name}")
print(f"🔗 View in portal: https://ml.azure.com/experiments/{submitted_job.experiment_name}/runs/{submitted_job.name}?wsid=/subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace_name}")

