from sklearn.ensemble import RandomForestRegressor
#error metric c-stat
from sklearn.metrics import roc_auc_score
import pandas as pd

from sklearn.linear_model import LinearRegression
#helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold

import numpy as np
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

##### Data cleaning ####
titanic = pd.read_csv("data/train.csv")
#replace NaN value in Age with Age median
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
#convert Name, Sex, Cabin, Embarked, and Ticket to numeric
#Sex
titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
titanic.loc[titanic["Sex"] == "female", "Sex"] = 1
print(titanic["Sex"].unique())
#Embarked
print(titanic["Embarked"].unique())
#S is most common
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
'''
#Name
name mr ->1
mrs -> 2
...
'''
#show attributes
print(titanic.head(5))
print(titanic.describe())




#####  Prediction  ####
#Target column
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
#Init algorithm class
alg = LinearRegression()
#Generate cross validation folds for titanic dataset
#return row indices corresponding to train and test
##??????????? set random state to ensure we get the same splits every time we run this
kf = KFold(titanic.shape[0], n_folds = 3, random_state = 1)

predictions = []
for train, test in kf:
	train_predictors = (titanic[predictors].iloc[train,:])
	train_target = titanic["Survived"].iloc[train]
	alg.fit(train_predictors, train_target)
	test_predictions = alg.predict(titanic[predictors].iloc[test,:])
	predictions.append(test_predictions)
	
	
	
##### Evaluate Error ####
#predictions are in 3 separate numpy arrays, concatenate them!
#here: concatenate on axis 0 -> only 1 axis
predictions = np.concatenate(predictions, axis = 0)
#map prediction to outcome
predictions[predictions > .5] = 1
predictions[predictions <=.5] = 0
#calculate accurate to evaluate
accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)

print accuracy 	
	
	
	
##### Logistic Regression ####
# result with acc about 78,3%, use regression to imrove
# Initialize our algorithm
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())



##### Test Set ####
titanic_test = pd.read_csv("data/test.csv")
#replace NaN value in Age with Age median
titanic_test["Age"] = titanic_test["Age"].fillna(titanic_test["Age"].median())
#convert Name, Sex, Cabin, Embarked, and Ticket to numeric
#Sex
titanic_test.loc[titanic_test["Sex"] == "male", "Sex"] = 0
titanic_test.loc[titanic_test["Sex"] == "female", "Sex"] = 1
print(titanic_test["Sex"].unique())
#Embarked
print(titanic_test["Embarked"].unique())
#S is most common
titanic_test["Embarked"] = titanic_test["Embarked"].fillna("S")
titanic_test.loc[titanic_test["Embarked"] == "S", "Embarked"] = 0
titanic_test.loc[titanic_test["Embarked"] == "C", "Embarked"] = 1
titanic_test.loc[titanic_test["Embarked"] == "Q", "Embarked"] = 2
#Fare
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
'''
#Name
name mr ->1
mrs -> 2
...
'''
#show attributes
print(titanic_test.head(5))
print(titanic_test.describe())


##### Submission ####
# Initialize the algorithm class
alg = LogisticRegression(random_state=1)
# Train the algorithm using all the training data
alg.fit(titanic[predictors], titanic["Survived"])
# Make predictions using the test set.
predictions = alg.predict(titanic_test[predictors])
# Create a new dataframe with only the columns Kaggle wants from the dataset.
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    })
    
print submission
