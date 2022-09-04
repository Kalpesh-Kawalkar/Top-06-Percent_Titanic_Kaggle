from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Logistic Regression
from sklearn.linear_model import LogisticRegression
m1 = LogisticRegression()
m1.fit(x_train, y_train)
ypred_m1 = m1.predict(x_test)
print('Testing_score', m1.score(x_test, y_test))

# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
m2 = MultinomialNB()
m2.fit(x_train, y_train)
ypred_m2 = m2.predict(x_test)
print('Testing_score', m2.score(x_test, y_test))

# KNN
from sklearn.neighbors import KNeighborsClassifier
m3 = KNeighborsClassifier()
m3.fit(x_train, y_train)
ypred_m3 = m3.predict(x_test)
print('Testing_score', m3.score(x_test, y_test))

# SVM
from sklearn.svm import SVC
m4 = SVC(C=10000)
m4.fit(x_train, y_train)
ypred_m4 = m4.predict(x_test)
print('Testing_score', m4.score(x_test, y_test))

# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
m5 = RandomForestClassifier()
m5.fit(x_train, y_train)
ypred_m5 = m5.predict(x_test)
print('Testing_score', m5.score(x_test, y_test))

# Decision Tree
from sklearn import tree
m6 = tree.DecisionTreeClassifier()
m6.fit(x_train, y_train)
ypred_m6 = m6.predict(x_test)
print('Testing_score', m6.score(x_test, y_test))

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
m7 = GradientBoostingClassifier()
m7.fit(x_train, y_train)
ypred_m7 = m7.predict(x_test)
print('Testing_score', m7.score(x_test, y_test))
