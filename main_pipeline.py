from azure.ai.ml import dsl, command, Input, Output
from azure.ai.ml.entities import Environment
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import os


custom_env = Environment(
    name="custom-pipeline-env",
    description="Environment from environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file="environment.yml",
)


# Define your compute name
compute_name = "cpu-cluster"  # <<< IMPORTANT: CHANGE THIS TO YOUR AZURE ML COMPUTE CLUSTER NAME

# --- Authenticate and get MLClient ---
# This part is typically run outside the pipeline definition itself
# but is needed to submit the pipeline.
try:
    ml_client = MLClient(
        DefaultAzureCredential(), subscription_id, resource_group, workspace_name
    )
except Exception as e:
    print(f"Could not connect to Azure ML Workspace: {e}")
    print("Please ensure your Azure CLI is logged in: az login")
    print("Or set environment variables for service principal authentication.")
    ml_client = None # Set to None if connection fails

# --- Environment Definition ---
# It's good practice to define the path to your environment.yml
# Assuming 'environment.yml' is in the same directory as this pipeline script
# Or, if it's in the 'src' directory: os.path.join("./src", "environment.yml")
env_conda_file_path = "environment.yml" 
# Ensure environment.yml exists or update the path

# Define and register the custom environment (recommended for reusability)
# You can register it once and then refer to it by name@version in your pipeline
custom_env_name = "custom-pipeline-env"
try:
    # Try to get the environment if it's already registered
    # If not, create and register it.
    registered_env = ml_client.environments.get(custom_env_name, version="latest")
    print(f"Using existing environment: {registered_env.name} (version: {registered_env.version})")
except Exception:
    print(f"Environment '{custom_env_name}' not found or error. Creating and registering...")
    custom_env = Environment(
        name=custom_env_name,
        description="Custom environment for pipeline steps",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04", # A common base image
        conda_file=env_conda_file_path,
        # Setting auto_increment_version to True will create a new version on each registration if content changes
        # For production, you might want to manage versions more explicitly.
        auto_increment_version=True 
    )
    # Register the environment
    registered_env = ml_client.environments.create_or_update(custom_env)
    print(f"Registered new environment: {registered_env.name} (version: {registered_env.version})")

# Reference the registered environment in the pipeline
# We'll use f-string for a dynamic reference to the registered version
environment_ref = f"azureml:{registered_env.name}:{registered_env.version}"

# --- Pipeline Definition ---
@dsl.pipeline(
    compute=compute_name,  # Set default compute for all steps in the pipeline
    description="End-to-end ML Pipeline: Preprocess → Train → Register",
    default_environment=environment_ref # Set default environment for all steps
)
def my_pipeline():
    # Define outputs for each step that needs to pass data to the next
    # These outputs are represented as paths within the Azure ML datastore
    # You can specify a named output for clarity or let Azure ML generate one.

    # 1. Preprocessing Job
    preprocess_job = command(
        name="preprocess",
        display_name="Preprocess Data",
        command="python preprocess.py --output_data ${{outputs.preprocessed_data}}", # Pass output path
        code="./src", # Assumes preprocess.py is in the src directory
        outputs={
            "preprocessed_data": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount")
        }
        # compute=compute_name, # Inherits from pipeline default, or uncomment to override
        # environment=environment_ref, # Inherits from pipeline default, or uncomment to override
    )()

    # 2. Training Job
    train_job = command(
        name="train",
        display_name="Train Model",
        command="python train.py --input_data ${{inputs.training_data}} --output_model ${{outputs.trained_model}}", # Pass input/output paths
        code="./src", # Assumes train.py is in the src directory
        inputs={
            "training_data": Input(type=AssetTypes.URI_FOLDER) # Reference the output from preprocess_job
        },
        outputs={
            "trained_model": Output(type=AssetTypes.URI_FOLDER, mode="rw_mount") # Output for the trained model artifact
        }
        # compute=compute_name, # Inherits from pipeline default
        # environment=environment_ref, # Inherits from pipeline default
    )()

    # Set dependency: train_job needs preprocess_job to complete
    train_job.inputs.training_data = preprocess_job.outputs.preprocessed_data

    # 3. Registration Job
    register_job = command(
        name="register",
        display_name="Register Model",
        # Pass model path and desired model name/description to register.py
        command="python register.py --model_path ${{inputs.model_to_register}} --model_name my-ml-model --model_description 'Model from pipeline'",
        code="./src", # Assumes register.py is in the src directory
        inputs={
            "model_to_register": Input(type=AssetTypes.URI_FOLDER) # Reference the output from train_job
        }
        # compute=compute_name, # Inherits from pipeline default
        # environment=environment_ref, # Inherits from pipeline default
    )()

    # Set dependency: register_job needs train_job to complete
    register_job.inputs.model_to_register = train_job.outputs.trained_model

    # The pipeline returns outputs if you want to capture them at the pipeline level
    # For example, returning the path to the registered model or a model ID
    return {
        "final_trained_model_path": train_job.outputs.trained_model
        # You could also add an output from register_job if it returns the registered model ID
    }

# --- Instantiate and Submit the Pipeline ---
if ml_client:
    pipeline_job = my_pipeline()

    print(f"Submitting pipeline '{pipeline_job.name}' to Azure ML...")
    # You can configure pipeline settings before submission
    # pipeline_job.display_name = "My First MLOps Pipeline"
    # pipeline_job.experiment_name = "model-lifecycle-experiment"

    # Submit the pipeline job
    returned_job = ml_client.jobs.create_or_update(
        pipeline_job, experiment_name="my-model-pipeline-experiment"
    )

    print(f"Pipeline submitted. Check its status at: {returned_job.studio_url}")
else:
    print("MLClient not initialized. Cannot submit pipeline.")
