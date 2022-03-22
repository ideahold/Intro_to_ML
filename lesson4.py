import pandas as pd

file = 'data.csv'
data = pd.read_csv(file)

data = data.dropna(axis=0)

features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude']

X = data[features]
y = data.Price

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.metrics import mean_absolute_error as mae

model = RFR(random_state=1)

model.fit(train_X, train_y)

pred = model.predict(val_X)

print(mae(val_y, pred))
