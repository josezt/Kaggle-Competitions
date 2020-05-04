
import pandas as pd
import numpy
from sklearn.ensemble import AdaBoostClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn import preprocessing



def clean_data(df):
    df = df.drop(['Name','Ticket','Cabin'],axis=1)
    df = pd.get_dummies(df, columns = ['Sex', 'Embarked'])
    x = df.values #returns a numpy array
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    df = pd.DataFrame(x_scaled)
    imp = SimpleImputer(strategy="most_frequent")
    labels = df.columns
    df = pd.DataFrame(imp.fit_transform(df))
    df.columns = labels

    return df

DIR = "/Users/JoseZarco/Documents/00_Machine_Learning/01_Kaggle/Competitions/titanic/"
data = pd.read_csv(DIR + 'train.csv')
y = data['Survived']
X = data.drop(['Survived'], axis=1)
for column in X.columns:
    nans = len(X) - X[column].count()
    print(f'In column {column} there are {nans} NaN values')



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=43)
X_train = clean_data(X_train)
X_test = clean_data(X_test)
X_train.shape
y_train.shape
#plt.figure(figsize=(16, 3))
plt.scatter(X_train['SibSp'], y_train)
plt.show()

clf = AdaBoostClassifier(n_estimators=30, random_state=0)
clf.fit(X_train, y_train);
preds_y_test = clf.predict(X_test)
preds_y_train = clf.predict(X_train)
print(accuracy_score(y_test, preds_y_test))
print(accuracy_score(y_train, preds_y_train))

clfNB = GaussianNB()
clfNB.fit(X_train, y_train);
preds_y_test = clfNB.predict(X_test)
preds_y_train = clfNB.predict(X_train)
print(accuracy_score(y_test, preds_y_test))
print(accuracy_score(y_train, preds_y_train))


param = {'max_depth':range(2,10)}
clfRF = RandomForestClassifier()
clfGrid = GridSearchCV(clfRF, param)
clfGrid.fit(X_train, y_train);
best_clf = clfGrid.best_estimator_
#clfRF.fit(X_train, y_train);
preds_y_test = best_clf.predict(X_test)
print(accuracy_score(y_test, preds_y_test))


parameters = {'kernel':('linear', 'rbf', 'poly'),
              'C':[1, 3, 4, 6, 8, 10]}
clfSVC = SVC()
clfGrid = GridSearchCV(clfSVC, parameters)
clfGrid.fit(X_train, y_train);
best_clf = clfGrid.best_estimator_
preds_y_test = best_clf.predict(X_test)
print(accuracy_score(y_test, preds_y_test))

submission_data = pd.read_csv(DIR + 'test.csv')
submission_data = clean_data(submission_data)
submission = submission_data['PassengerId'].astype('Int32')
predicts = clfRF.predict(submission_data)
submission = submission.to_frame()
submission['Survived'] = predicts
submission.to_csv(DIR + "submission.csv", index = False)
