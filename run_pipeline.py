from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from main_pipeline import my_pipeline  # import the pipeline function
import os

subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
resource_group = os.getenv("AZURE_RESOURCE_GROUP")
workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

credential = DefaultAzureCredential()
ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

# Create pipeline job from function
pipeline = my_pipeline()

# Submit the pipeline job
submitted_job = ml_client.jobs.create_or_update(pipeline)

print(f"Pipeline submitted. Job ID: {submitted_job.name}")
print(f"View run: https://ml.azure.com/experiments/{submitted_job.experiment_name}/runs/{submitted_job.name}?wsid=/subscriptions/{subscription_id}/resourcegroups/{resource_group}/workspaces/{workspace_name}")
