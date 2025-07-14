from azure.ai.ml import dsl, command
from azure.ai.ml.entities import Environment
from azure.ai.ml.constants import AssetTypes

# Define your compute name
compute_name = "cpu-cluster"  # ← change if different

# Shared environment
custom_env = Environment(
    name="custom-pipeline-env",
    description="Environment from environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file="environment.yml",
)

@dsl.pipeline(
    compute=compute_name,
    description="Preprocess → Train → Register"
)
def my_pipeline():
    preprocess_job = command(
        name="preprocess",
        display_name="Preprocess Step",
        command="python preprocess.py",
        code="./src",
        environment=custom_env,
        compute=compute_name,
    )()

    train_job = command(
        name="train",
        display_name="Train Step",
        command="python train.py",
        code="./src",
        environment=custom_env,
        compute=compute_name,
    )()

    register_job = command(
        name="register",
        display_name="Register Step",
        command="python register.py",
        code="./src",
        environment=custom_env,
        compute=compute_name,
    )()

    # Set dependencies
    train_job.set_dependencies([preprocess_job])
    register_job.set_dependencies([train_job])

