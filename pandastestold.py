import pandas as pd
df = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
df.shape
test.shape
print df
print test
print(test["Embarked"].unique())
print test.Embarked.value_counts()


'''
df.Survived.value_counts()


#print df
#print df.shape
print df.Survived.value_counts()
print df.Sex.value_counts()


print df.Embarked.value_counts()
print df.Pclass.value_counts()
print df.SibSp.value_counts()
print df.Parch.value_counts()
print df.Fare.value_counts()
print df.Cabin.value_counts()

print df.Sex.value_counts().plot(kind='bar')
print df.Fare.hist(bins=5)
#print df(df.Fare.isnull())

print df[df.Age < 15].Survived.value_counts().plot(kind='barh')
'''
