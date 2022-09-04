from sklearn.ensemble import VotingClassifier
evc = VotingClassifier(estimators=[('lr', m1), ('gnb', m2), ('knn', m3), ('svm', m4), ('rf', m5), ('dt', m6), ('gbm', m7)], voting='hard')
evc.fit(x_train, y_train)
final_pred = print('Testing score', evc.score(x_test, y_test))
