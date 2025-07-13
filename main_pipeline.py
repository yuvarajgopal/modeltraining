from azureml.core import Workspace, Experiment, Environment, ScriptRunConfig
from azureml.core.compute import ComputeTarget
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep

# Load workspace
ws = Workspace.from_config()

# Compute
compute = ComputeTarget(workspace=ws, name='cpu-cluster')

# Environment
env = Environment.from_conda_specification(name='ml-pipeline-env', file_path='environment.yml')

# Pipeline data between steps
preprocessed_data = PipelineData('preprocessed_data', datastore=ws.get_default_datastore())
trained_model = PipelineData('trained_model', datastore=ws.get_default_datastore())

# Step 1 - Preprocess
step1 = PythonScriptStep(
    name="Preprocess Data",
    script_name="preprocess.py",
    source_directory="src",
    outputs=[preprocessed_data],
    compute_target=compute,
    environment=env,
    allow_reuse=True
)

# Step 2 - Train
step2 = PythonScriptStep(
    name="Train Model",
    script_name="train.py",
    source_directory="src",
    inputs=[preprocessed_data],
    outputs=[trained_model],
    compute_target=compute,
    environment=env,
    allow_reuse=True
)

# Step 3 - Register Model
step3 = PythonScriptStep(
    name="Register Model",
    script_name="register.py",
    source_directory="src",
    inputs=[trained_model],
    compute_target=compute,
    environment=env,
    allow_reuse=True
)

# Create pipeline
pipeline = Pipeline(workspace=ws, steps=[step1, step2, step3])
experiment = Experiment(workspace=ws, name='iris-pipeline')
run = experiment.submit(pipeline)
run.wait_for_completion(show_output=True)
