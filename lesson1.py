import pandas as pd 

file = 'data.csv'
data = pd.read_csv(file)

#data.describe()
data.columns # для JupyterLab (увидите имена всех колонн)

data = data.dropna(axis=0)

features = ['Rooms', 'Bathroom', 'Landsize']

X = data[features]
y = data.Price

X.head() # для JupyterLab (увидите несколько первых элементов)

from sklearn.tree import DecisionTreeRegressor as DTR

model = DTR(random_state=1)
model.fit(X, y)

print(X.head())

print(model.predict(X.head()))
