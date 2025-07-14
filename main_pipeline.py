from azure.ai.ml import dsl, command
from azure.ai.ml.constants import AssetTypes

compute_name = "cpu-cluster"

@dsl.pipeline(
    compute=compute_name,
    description="Pipeline with preprocess → train → register"
)
def my_pipeline():
    preprocess = command(
        name="preprocess",
        command="python preprocess.py",
        code="./src",
        compute=compute_name,
        environment="custom-pipeline-env",  # Use registered name only
    )()

    train = command(
        name="train",
        command="python train.py",
        code="./src",
        compute=compute_name,
        environment="custom-pipeline-env",
    )()

    register = command(
        name="register",
        command="python register.py",
        code="./src",
        compute=compute_name,
        environment="custom-pipeline-env",
    )()

    train.set_dependencies([preprocess])
    register.set_dependencies([train])

