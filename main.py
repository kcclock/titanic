import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

train_set_path = "./DataSources/train.csv"
train_data = pd.read_csv(train_set_path)

#print(train_data.describe())
#print(train_data.head)
#print(train_data.columns)



# Prediction-Features definieren
features_1 = ['SibSp']

X = train_data[features_1]
#pd.DataFrame(X).fillna(1)
#print(X.head)

#Zielvariable y definieren
y = train_data['Survived']


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 6)


model = DecisionTreeClassifier (random_state=8)
model.fit(train_X,train_y)


predictions = model.predict(val_X)
print(mean_absolute_error(val_y, predictions))

#------------------- Das Gleiche mit einem RandomForrest-Classifier

model = RandomForestClassifier (random_state=8)
model.fit(train_X,train_y)


predictions = model.predict(val_X)
print(mean_absolute_error(val_y, predictions))

#--------------------------------



"""
test_set_path = "./DataSources/test.csv"
test_data = pd.read_csv(test_set_path)

#print(train_data.describe())
#print(train_data.head)
print(test_data.columns)

test_X = test_data['SibSp']
test_y = test_data['Survived']

predictions = model.predict(test_X)

print(mean_absolute_error(test_y, predictions))
"""