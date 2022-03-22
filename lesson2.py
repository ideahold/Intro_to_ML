import pandas as pd

file = 'data.csv'
data = pd.read_csv(file)

data = data.dropna(axis=0)

features = ['Rooms', 'Bathroom', 'Landsize']

X = data[features]
y = data.Price

from sklearn.tree import DecisionTreeRegressor as DTR

model = DTR(random_state=1)

model.fit(X, y)


from sklearn.metrics import mean_absolute_error

pred_home_prices = model.predict(X)
mean_absolute_error(y, pred_home_prices)


from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

model = DTR()
model.fit(train_X, train_y)

val_pred = model.predict(val_X)
print(mean_absolute_error(val_y, val_pred))
