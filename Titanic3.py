import os
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

caminho = os.path.dirname(os.path.abspath(__file__))

train_df = pd.DataFrame(pd.read_csv(os.path.join(caminho, 'train.csv'), sep=','))

test_df = pd.DataFrame(pd.read_csv(os.path.join(caminho, 'test.csv'), sep=','))



#Variáveis

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

knn = KNeighborsClassifier(n_neighbors=5)

gb = GaussianNB()

lr = LogisticRegression()

tree = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=79)

random = RandomForestClassifier(random_state=88)

svm = SVC()

scaler = MinMaxScaler()

# Drops

train_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)

test_df.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)


# NaN fills

train_df['Age'] = train_df['Age'].fillna(train_df['Age'].mean())
train_df['Embarked'] = train_df['Embarked'].fillna(train_df['Embarked'].mode()[0])

test_df['Age'] = test_df['Age'].fillna(test_df['Age'].mean())
test_df['Embarked'] = test_df['Embarked'].fillna(test_df['Embarked'].mode()[0])

# Encoding

enco_sex = pd.DataFrame(ohe.fit_transform(train_df[['Sex']]), columns=ohe.get_feature_names_out(['Sex']))
train_df = pd.concat([train_df.drop('Sex', axis=1).reset_index(drop=True), enco_sex],axis=1)
train_df['Embarked'] = train_df['Embarked'].astype('category').cat.codes



test_enco_sex = pd.DataFrame(ohe.fit_transform(test_df[['Sex']]), columns=ohe.get_feature_names_out(['Sex']))
test_df = pd.concat([test_df.drop('Sex', axis=1).reset_index(drop=True), test_enco_sex],axis=1)
test_df['Embarked'] = test_df['Embarked'].astype('category').cat.codes

# Normalizando

train_df['Fare'] = np.log1p(train_df['Fare'])

test_df['Fare'] = np.log1p(test_df['Fare'])

# Classificadores

X = train_df.drop(['Survived'], axis=1)

y = train_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=180)


#Parametros 

tree_param = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [3, 4, 5, 6, 8, 10, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10]}

knn_param = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights': ['uniform', 'distance'],
    'p': [1, 2]}

lr_param = {
    'C': [0.01, 0.1, 1, 10, 100],
    'solver': ['liblinear', 'lbfgs'],
    'penalty': ['l2'],
    'max_iter': [500,]
}

svm_param = {
    'kernel': ['linear', 'rbf', 'poly'],
    'C': [0.1, 1, 10],
    'gamma': ['scale', 0.1, 0.01]
}

random_param = {
    'n_estimators': [100, 200, 300],
    'max_depth': [4, 6, 8, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy', 'log_loss']
}


tree_grid = GridSearchCV(estimator=tree, param_grid=tree_param, cv=5, scoring='accuracy')

knn_grid = GridSearchCV(estimator=knn, param_grid=knn_param, cv= 5, scoring='accuracy')

lr_grid = GridSearchCV(estimator=lr, param_grid=lr_param, cv=5, scoring='accuracy')

svm_grid = GridSearchCV(estimator=svm, param_grid=svm_param, cv=5, scoring='accuracy')

random_grid = RandomizedSearchCV(estimator=random, param_distributions=random_param, n_iter=50, cv=5, scoring='accuracy', random_state=71, verbose=2)

# Normalizando os datasets

norm_X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

norm_X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

#Treinos com os grids 



tree_grid.fit(X_train, y_train)

best_tree = tree_grid.best_estimator_   

tree_pred = pd.DataFrame(best_tree.predict(X_test))

tree_acc = accuracy_score(y_test, tree_pred)

print(f"A precisão tree foi de: {tree_acc: .2f}")


knn_grid.fit(X_train, y_train)  

best_knn = knn_grid.best_estimator_

knn_prev = pd.DataFrame(best_knn.predict(X_test))

knn_acc = accuracy_score(y_test, knn_prev)

print(f"A precisão KNN foi de: {knn_acc: .2f}")


lr_grid.fit(norm_X_train, y_train)

best_lr = lr_grid.best_estimator_

lr_prev = pd.DataFrame(best_lr.predict(norm_X_test))

lr_acc = accuracy_score(y_test, lr_prev)

print(f"A precisão LR foi de: {lr_acc: .2f}")



svm_grid.fit(norm_X_train, y_train)

best_svm = svm_grid.best_estimator_

svm_prev = pd.DataFrame(best_svm.predict(norm_X_test))

svm_acc = accuracy_score(y_test, svm_prev)

print(f"A precisão do SVM foi de: {svm_acc: .2f}")



random_grid.fit(X_train, y_train)

best_random = random_grid.best_estimator_

random_prev = pd.DataFrame(best_random.predict(X_test))

random_acc = accuracy_score(y_test, random_prev)

print(f"A precisão da Random Forest foi: {random_acc: .2f}")











