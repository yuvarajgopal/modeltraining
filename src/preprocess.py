import pandas as pd
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
train, test = train_test_split(df, test_size=0.2, random_state=42)

os.makedirs('outputs', exist_ok=True)
train.to_csv('outputs/train.csv', index=False)
test.to_csv('outputs/test.csv', index=False)
