import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.impute import SimpleImputer

train_set_path = "./DataSources/train.csv"
train_data = pd.read_csv(train_set_path)

#print(train_data.describe())
#print(train_data.head)
#print(train_data.columns)


# Prediction-Features definieren
features_1 = ['SibSp']
features_2 = ['Age']
features_3 = ['SibSp','Age']
features_4 = ['SibSp','Pclass','PassengerId', 'Age', 'Parch']
features_5 = ['Pclass', 'SibSp']

X = train_data[features_5]

#pd.DataFrame(X).fillna(1)
#print(X.head)

#Zielvariable y definieren
y = train_data['Survived']


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 2)

#Extend and Impute missing Values

# Make copy to avoid changing original data (when imputing)
train_X_plus = train_X.copy()
val_X_plus = val_X.copy()

# Make new columns indicating what will be imputed

# Get names of columns with missing values
cols_with_missing = [col for col in train_X.columns if train_X[col].isnull().any()]
#print(str(cols_with_missing) + "BLA")

for col in cols_with_missing:
    train_X_plus[col + '_was_missing'] = train_X_plus[col].isnull()
    val_X_plus[col + '_was_missing'] = val_X_plus[col].isnull()


my_imputer = SimpleImputer()

imputed_train_X = pd.DataFrame(my_imputer.fit_transform(train_X_plus))
imputed_val_X = pd.DataFrame(my_imputer.transform(val_X_plus))

# Imputation removed column names; put them back
imputed_train_X.columns = train_X_plus.columns
imputed_val_X.columns = val_X_plus.columns


#------ DecisionTree-Model
"""
model = DecisionTreeClassifier (random_state=8)
model.fit(train_X,train_y)

predictions = model.predict(val_X)
print(mean_absolute_error(val_y, predictions))
"""
#------------------- Das Gleiche mit einem RandomForrest-Classifier

model = RandomForestClassifier (random_state=2)
model.fit(imputed_train_X,train_y)

#print(imputed_train_X.head)

predictions = model.predict(imputed_val_X)
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