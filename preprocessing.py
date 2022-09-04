from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
training_data['Sex'] = lb.fit_transform(training_data['Sex'])
training_data['Embarked'] = lb.fit_transform(training_data['Embarked'])
training_data['Sex'].value_counts()
training_data['Embarked'].value_counts()

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
testing_data['Sex'] = lb.fit_transform(testing_data['Sex'])
testing_data['Embarked'] = lb.fit_transform(testing_data['Embarked'])
testing_data['Sex'].value_counts()
testing_data['Embarked'].value_counts()
testing_data.head(10)

# Removing Unnecessary Columns
def clean(training_data):
    training_data = training_data.drop(['Ticket', 'Cabin', 'Name', 'PassengerId', 'SibSp', 'Parch', 'Embarked'], axis=1)
    cols = ['Fare', 'Age']
    for col in cols:
        training_data[col].fillna(training_data[col].mean(), inplace=True)
    return training_data
training_data = clean(training_data)
testing_data = clean(testing_data)

training_data.head(10)
x = training_data.drop(columns = ['Survived'])
y = training_data['Survived']
