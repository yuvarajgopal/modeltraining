from azureml.core import Run

run = Run.get_context()
model_path = 'outputs/model.pkl'
model = run.register_model(model_name='iris-model', model_path=model_path)
print(f"Model registered: {model.name} v{model.version}")
