from models.KNN_pandas import KNN
import pandas as pd

df = pd.read_csv("data/glass.data", header=None)
df.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Class']
df = df.sample(frac=1)

query_point = pd.read_csv("data/glass_query_point.data", header=None)
query_point.columns = ['Id', 'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Class']

my_knn = KNN(num_neighbors=3, data=df, attribute_names=list(df.columns))
print(my_knn.classify(query_point))
