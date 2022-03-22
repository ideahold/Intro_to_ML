from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor as DTR

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DTR(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    
    pred_val = model.predict(val_X)
    
    mae = mean_absolute_error(val_y, pred_val)
    
    return mae
    
import pandas as pd

file = 'data.csv'
data = pd.read_csv(file)

data = data.dropna(axis=0)

features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude']

X = data[features]
y = data.Price

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    
    print(f'max_leaf_nodes: {max_leaf_nodes}, MAE: {my_mae}')

    
