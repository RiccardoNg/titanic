from sklearn.ensemble import RandomForestRegressor
#error metric c-stat
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
#helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold

import numpy as np
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
#regression expression for extract string Mr, Mrs, Ms
import re
import operator

from sklearn.feature_selection import SelectKBest, f_classif





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



#####  Generating New Features  ####
# Generating a familysize column
titanic["FamilySize"] = titanic["SibSp"] + titanic["Parch"]
# The .apply method generates a new series
titanic["NameLength"] = titanic["Name"].apply(lambda x: len(x))
"""
def get_title(name):
	title_search = re.search(' ([A-Za-z]+)\.', name)
    # If the title exists, extract and return it.
    if title_search:
		return title_search.group(1)
    return ""
"""
def get_title(name):
	title_search = title_search = re.search(' ([A-Za-z]+)\.', name)
	if title_search:
		return title_search
	return ""
	
# Get all the titles and print how often each one occurs.
titles = titanic["Name"].apply(get_title)
print(pd.value_counts(titles))
# Map each title to an integer.  Some titles are very rare, and are compressed into the same codes as other titles.
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
for k,v in title_mapping.items():
    titles[titles == k] = v
# Verify that we converted everything.
print(pd.value_counts(titles))
# Add in the title column.
titanic["Title"] = titles



### Family Groups
# A dictionary mapping family name to id
family_id_mapping = {}
# A function to get the id given a row
def get_family_id(row):
    # Find the last name by splitting on a comma
    last_name = row["Name"].split(",")[0]
    # Create the family id
    family_id = "{0}{1}".format(last_name, row["FamilySize"])
    # Look up the id in the mapping
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            # Get the maximum id from the mapping and add one to it if we don't have an id
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]
# Get the family ids with the apply method
family_ids = titanic.apply(get_family_id, axis=1)
# There are a lot of family ids, so we'll compress all of the families under 3 members into one code.
family_ids[titanic["FamilySize"] < 3] = -1
# Print the count of each unique id.
print(pd.value_counts(family_ids))
titanic["FamilyId"] = family_ids

#####  SHOW TIME ####
#show attributes
print(titanic.head(20))
print(titanic.describe())


'''
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
'''

#####  Prediction with Random Forest  ####
#predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

### pick new features
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "FamilyId"]

# Perform feature selection
selector = SelectKBest(f_classif, k=5)
selector.fit(titanic[predictors], titanic["Survived"])

# Get the raw p-values for each feature, and transform from p-values into scores
scores = -np.log10(selector.pvalues_)

# Plot the scores.  See how "Pclass", "Sex", "Title", and "Fare" are the best?
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()

predictors = ["Pclass", "Sex", "Fare", "Title"]

alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
kf = KFold(titanic.shape[0], n_folds = 3, random_state = 1)
scores = cross_validation.cross_val_score(alg, titanic[predictors], titanic["Survived"], cv=kf)
# Take the mean of the scores (because we have one for each fold)
print(scores.mean())


#####  Parameter Tuning ^ ####
# tuning in random forest to avoid overfit, by increasing min_samples_split and min_samples_leaf





##################################################################3
	
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
    
#print submission
