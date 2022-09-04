predictions = evc.predict(testing_data)
print(predictions)

output = pd.DataFrame({'PassengerId': test_Ids.values, 'Survived': predictions})
output.to_csv('titanic_submission.csv', index=False)
print('Your Submission was Successfull..!')