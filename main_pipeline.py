from azure.ai.ml import dsl, command
from azure.ai.ml.entities import Environment
from azure.ai.ml.constants import AssetTypes

# Define your compute
compute_name = "cpu-cluster"  # Replace with your actual compute

# Shared environment
env = Environment(
    name="custom-pipeline-env",
    description="Custom environment from environment.yml",
    image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
    conda_file="environment.yml"
)

# Define component functions
def get_preprocess_step():
    return command(
        name="preprocess",
        display_name="Preprocess Step",
        description="Preprocessing script",
        code="./src",
        command="python preprocess.py",
        environment=env,
        compute=compute_name,
    )

def get_train_step():
    return command(
        name="train",
        display_name="Training Step",
        description="Train the model",
        code="./src",
        command="python train.py",
        environment=env,
        compute=compute_name,
    )

def get_register_step():
    return command(
        name="register",
        display_name="Register Step",
        description="Register the model",
        code="./src",
        command="python register.py",
        environment=env,
        compute=compute_name,
    )

# Define pipeline using @dsl.pipeline
@dsl.pipeline(
    compute=compute_name,
    description="Pipeline with preprocess, train, register"
)
def my_pipeline():
    # Create jobs by calling components
    preprocess_job = get_preprocess_step()  # ✅ job
    train_job = get_train_step()            # ✅ job
    register_job = get_register_step()      # ✅ job

    # Set dependencies
    train_job.set_dependencies([preprocess_job])
    register_job.set_dependencies([train_job])

    # Return the final job if needed
    return {
        "preprocess": preprocess_job.outputs,
        "train": train_job.outputs,
        "register": register_job.outputs
    }

