import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from azureml.core import Run
import os

run = Run.get_context()
df = pd.read_csv('train.csv')  # input file from previous step

X = df.drop('species', axis=1)
y = df['species']

model = RandomForestClassifier()
model.fit(X, y)

run.log('train_accuracy', model.score(X, y))
os.makedirs('outputs', exist_ok=True)
joblib.dump(model, 'outputs/model.pkl')
