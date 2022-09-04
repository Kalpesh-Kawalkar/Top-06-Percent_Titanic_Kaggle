# Importing Python Libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the dataset
training_data = pd.read_csv("../input/titanic/train.csv")
testing_data = pd.read_csv("../input/titanic/test.csv")
test_Ids = testing_data['PassengerId']

training_data.head()
training_data.shape
testing_data.head()
testing_data.shape
training_data.describe()
testing_data.describe()
training_data.isnull().sum()

training_data = training_data.fillna(training_data.mean())
training_data['Embarked'] = training_data['Embarked'].fillna('S')

training_data.dtypes

# Converting the int data types into object data type for further operations
training_data['Survived'] = training_data['Survived'].astype(str)
training_data['Pclass'] = training_data['Pclass'].astype(str)
training_data['SibSp'] = training_data['SibSp'].astype(str)
training_data['Parch'] = training_data['Parch'].astype(str)

training_data.dtypes
testing_data.isnull().sum()

training_data['Sex'].value_counts().plot(kind='bar', color='red')
training_data['Pclass'].value_counts().plot(kind='pie')

training_data['Fare'].max()
training_data['Fare'].min()
training_data['Age'].max()

plt.scatter(training_data['Fare'], training_data['Age'])
plt.xlabel('Fare')
plt.ylabel('Age')
plt.show()
sns.scatterplot(training_data['Fare'], training_data['Age'], hue=training_data['Sex'])
plt.show()

