from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import PipelineJob
import os

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

# Authenticate
credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# Load pipeline job from YAML file (adjust path if needed)
pipeline_job = ml_client.jobs.create_or_update(
    PipelineJob.load("pipeline.yml")
)

print(f"Pipeline submitted. Job ID: {pipeline_job.name}")
print(f"View in Azure ML studio: https://ml.azure.com/experiments/{pipeline_job.experiment_name}/runs/{pipeline_job.name}?wsid=/subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace_name}")
