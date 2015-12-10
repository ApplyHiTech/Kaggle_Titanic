import pandas as pd
import matplotlib as mpl
import sklearn 
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression


#Convert embarked
def convert_embarked(val):
    if val == "S":
        return 0
    elif val == "C":
        return 1
    elif val == "Q":
        return 2
    else:
        return 0
    
#convert sex, male = 0
def convert_sex(val):
    if val == 'male':
        return 0
    elif val == 'female':
        return 1
    else:
        print "Error in Sex"
        return 0
        
            
#Read training data 
df = pd.read_csv("train.csv")
#Edit Age
df["Age"]=df["Age"].fillna(df["Age"].median())

#edit sex
df["Sex"] = df["Sex"].apply(convert_sex)

#edit embarked
df["Embarked"] = df["Embarked"].apply(convert_embarked)



# Initialize our algorithm
alg = LogisticRegression(random_state=1)
# Compute the accuracy score for all the cross validation folds.  (much simpler than what we did before!)
scores = cross_validation.cross_val_score(alg, df[predictors], df["Survived"], cv=3)
# Take the mean of the scores (because we have one for each fold)


#Now, i need to clean up the titanic_test file
titanic_test = pd.read_csv("test.csv")

#Edit age
titanic_test["Age"] = titanic_test["Age"].fillna(df["Age"].median())
titanic_test["Fare"] = titanic_test["Fare"].fillna(titanic_test["Fare"].median())
titanic_test["Embarked"] = titanic_test["Embarked"].apply(convert_embarked)
titanic_test["Sex"] = titanic_test["Sex"].apply(convert_sex)

alg = LogisticRegression(random_state=1)

# Train the algorithm using the training data
alg.fit(df[predictors], df["Survived"])

# Make predictions using the test set.
predictions = alg.predict(titanic_test[predictors])

# Output for Kaggle
submission = pd.DataFrame({
        "PassengerId": titanic_test["PassengerId"],
        "Survived": predictions
    }).to_csv("ApplyHiTech_Titanic_Submission.csv",index=False)
