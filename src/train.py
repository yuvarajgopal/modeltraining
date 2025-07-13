import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
from azureml.core import Run

run = Run.get_context()
dataset_path = 'train.csv'  # pulled from previous step output

df = pd.read_csv(dataset_path)
X = df.drop('species', axis=1)
y = df['species']

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

os.makedirs('outputs', exist_ok=True)
joblib.dump(model, 'outputs/model.pkl')

run.log("train_accuracy", model.score(X, y))
