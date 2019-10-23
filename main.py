import pandas as pd
from sklearn.tree import DecisionTreeClassifier

train_set_path = "./DataSources/train.csv"
train_data = pd.read_csv(train_set_path)

#print(train_data.describe())
#print(train_data.head)
print(train_data.columns)



# Prediction-Features definieren
features_1 = ['SibSp']

X = train_data[features_1]
#pd.DataFrame(X).fillna(1)
#print(X.head)

#Zielvariable y definieren
y = train_data['Survived']


model = DecisionTreeClassifier (random_state=1)
model.fit(X,y)

print(model.predict(X))
