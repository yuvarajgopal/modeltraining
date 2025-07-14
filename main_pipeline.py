from azure.ai.ml import dsl, command
from azure.ai.ml.entities import Environment
from azure.ai.ml.constants import AssetTypes

# Define your compute name
compute_name = "cpu-cluster"  # Change if needed

# Define command components inline
def get_preprocess_step():
    return command(
        name="preprocess_step",
        display_name="Preprocess Data",
        description="Preprocess raw input data",
        command="python preprocess.py",
        environment=Environment(
            conda_file="environment.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            name="custom-pipeline-env"
        ),
        code="./src",
        compute=compute_name
    )

def get_train_step():
    return command(
        name="train_step",
        display_name="Train Model",
        description="Train ML model",
        command="python train.py",
        environment=Environment(
            conda_file="environment.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            name="custom-pipeline-env"
        ),
        code="./src",
        compute=compute_name
    )

def get_register_step():
    return command(
        name="register_step",
        display_name="Register Model",
        description="Register trained model to workspace",
        command="python register.py",
        environment=Environment(
            conda_file="environment.yml",
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            name="custom-pipeline-env"
        ),
        code="./src",
        compute=compute_name
    )
@dsl.pipeline(
    compute=compute_name,
    description="Full ML pipeline: preprocess → train → register"
)
def my_pipeline():
    preprocess = get_preprocess_step()()   # Call component to create job
    train = get_train_step()()             # Call component to create job
    train.set_dependencies([preprocess])

    register = get_register_step()()       # Call component to create job
    register.set_dependencies([train])

