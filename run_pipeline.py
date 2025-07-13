from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import os

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# Submit pipeline job directly by YAML path
pipeline_job = ml_client.jobs.create_or_update(
    path="pipeline.yml"  # adjust if your pipeline YAML is named differently or in a folder
)

print(f"Pipeline submitted. Job ID: {pipeline_job.name}")
print(f"View in Azure ML studio: https://ml.azure.com/experiments/{pipeline_job.experiment_name}/runs/{pipeline_job.name}?wsid=/subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace_name}")
