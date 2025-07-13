from azureml.core import Workspace, Environment, Experiment
from azureml.core.compute import ComputeTarget
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.steps import PythonScriptStep

# Load Azure ML workspace
ws = Workspace.from_config()

# Compute target (must exist)
compute = ComputeTarget(workspace=ws, name='cpu-cluster')

# Environment
env = Environment.from_conda_specification(name='ml-pipeline-env', file_path='environment.yml')

# Output from step 1 → input to step 2
preprocess_output = PipelineData('train_data', datastore=ws.get_default_datastore())

# Step 1 - Preprocessing
step1 = PythonScriptStep(
    name="preprocess-data",
    script_name="preprocess.py",
    source_directory="src",
    outputs=[preprocess_output],
    compute_target=compute,
    environment=env,
    allow_reuse=True
)

# Step 2 - Training
step2 = PythonScriptStep(
    name="train-model",
    script_name="train.py",
    source_directory="src",
    inputs=[preprocess_output],
    arguments=['train.csv'],
    compute_target=compute,
    environment=env,
    allow_reuse=True
)

pipeline = Pipeline(workspace=ws, steps=[step1, step2])
experiment = Experiment(ws, 'iris-ml-pipeline')
run = experiment.submit(pipeline)
run.wait_for_completion(show_output=True)
