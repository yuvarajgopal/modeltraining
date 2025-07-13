from azure.ai.ml import MLClient
from azure.ai.ml import load_job
from azure.identity import DefaultAzureCredential
import os

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# Load pipeline job from YAML file
pipeline_job = load_job(path="pipeline.yml")  # Adjust path if needed

# Submit the job to Azure ML workspace
returned_job = ml_client.jobs.create_or_update(pipeline_job)

print(f"Pipeline submitted. Job ID: {returned_job.name}")
print(f"View in Azure ML studio: https://ml.azure.com/experiments/{returned_job.experiment_name}/runs/{returned_job.name}?wsid=/subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace_name}")
