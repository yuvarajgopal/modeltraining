from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient
from main_pipeline import my_pipeline
import os

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

credential = DefaultAzureCredential()
ml_client = MLClient(
    credential=credential,
    subscription_id=subscription_id,
    resource_group=resource_group,
    workspace_name=workspace_name
)

# Build and submit pipeline
pipeline_job = my_pipeline()
submitted_job = ml_client.jobs.create_or_update(pipeline_job)

print(f"✅ Pipeline submitted: {submitted_job.name}")
print(f"🔗 View in portal: https://ml.azure.com/experiments/{submitted_job.experiment_name}/runs/{submitted_job.name}?wsid=/subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace_name}")

