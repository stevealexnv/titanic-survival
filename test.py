# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from keras.models import Sequential
from keras.layers import Dense
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Importing the dataset
df_train = pd.read_csv('dataset/train.csv')
df_test = pd.read_csv('dataset/test.csv')

df_data = [df_train, df_test]

PID = df_test['PassengerId']

# Data Cleaning

# Creating features Family Size, IsAlone, HasCabin and Title
for dataset in df_data:
    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1
    
    dataset['Title'] = dataset['Name'].apply(lambda x: x.split(',')[1])
    dataset['Title'] = dataset['Title'].apply(lambda x: x.split('.')[0])
     
# Filling missing data
for dataset in df_data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
    age_avg = dataset['Age'].mean()
    age_std = dataset['Age'].std()
    age_null_count = dataset['Age'].isnull().sum()
    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)
    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list
    dataset['Age'] = dataset['Age'].astype(int)

# Categorise Age
for dataset in df_data:    
    dataset.loc[dataset['Age'] <= 16, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
    dataset.loc[dataset['Age'] > 64, 'Age'] = 4

# Categorise Fare
for dataset in df_data:    
    dataset.loc[dataset['Fare'] == 0, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 0) & (dataset['Fare'] <= 50), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 50) & (dataset['Fare'] <= 100), 'Fare'] = 2
    dataset.loc[(dataset['Fare'] > 100) & (dataset['Fare'] <= 200), 'Fare'] = 3
    dataset.loc[(dataset['Fare'] > 200) & (dataset['Fare'] <= 500), 'Fare'] = 4
    dataset.loc[dataset['Fare'] > 500, 'Fare'] = 5
    
# Encoding categorical data    
for dataset in df_data:
    dataset['Sex'] = dataset['Sex'].map({'male': 0,
                                       'female': 1}).astype(int)
    dataset['Embarked'] = dataset['Embarked'].map({'S': 0,
                                                   'C': 1,
                                                   'Q': 2}).astype(int)

# Feature Selection
drop_elements = ['PassengerId',
                 'Name',
                 'Ticket',
                 'Cabin',
                 'SibSp',
                 'Parch']
df_train = df_train.drop(drop_elements, axis = 1)
df_test = df_test.drop(drop_elements, axis = 1)

# Creating Training and Test set arrays
X_train = df_train.iloc[:, 1:].values
X_test = df_test.values
y_train = df_train.iloc[:, 0].values

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting Random Forest Classifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
mean = accuracies.mean()
std = accuracies.std()

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Generate Submission File
submission = pd.DataFrame({'PassengerId': PID, 'Survived': y_pred})
submission.to_csv("test.csv", index=False)