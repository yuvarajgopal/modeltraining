from azure.ai.ml import dsl, command

compute_name = "cpu-cluster"  # ✅ Change this if you use a different compute target

@dsl.pipeline(
    compute=compute_name,
    description="My ML pipeline: preprocess → train → register"
)
def my_pipeline():
    preprocess = command(
        name="preprocess",
        command="python preprocess.py",
        code="./src",
        compute=compute_name,
        environment="custom-pipeline-env"  # Must match registered env name
    )()

    train = command(
        name="train",
        command="python train.py",
        code="./src",
        compute=compute_name,
        environment="custom-pipeline-env"
    )()

    register = command(
        name="register",
        command="python register.py",
        code="./src",
        compute=compute_name,
        environment="custom-pipeline-env"
    )()

    # Set step dependencies
    train.set_dependencies([preprocess])
    register.set_dependencies([train])

