import pandas as pd
import numpy as np
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

caminho_base = os.path.dirname(os.path.abspath(__file__))

train_df = pd.DataFrame(pd.read_csv(os.path.join(caminho_base, 'train.csv'), sep=','))
test_df = pd.DataFrame(pd.read_csv(os.path.join(caminho_base, 'test.csv'), sep=','))

scaler = MinMaxScaler()

ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

ode = OrdinalEncoder()

exp_df = train_df.copy() 

ost_df = test_df.copy()

exp_df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True) # A coluna das cabines tinha mais de 680 valores nulos.

ost_df.drop(['Cabin', 'Name', 'Ticket'], axis=1, inplace=True)

exp_df['Sex'] = ode.fit_transform(exp_df[['Sex']])

ost_df['Sex'] = ode.fit_transform(ost_df[['Sex']])

exp_df['Embarked'] = exp_df['Embarked'].astype('category').cat.codes # Transforma os locais de embarque em números.

ost_df['Embarked'] = ost_df['Embarked'].astype('category').cat.codes

exp_df['Age'] = exp_df['Age'].fillna(exp_df['Age'].mean()) # Cobrir as idades faltando. 

ost_df['Age'] = ost_df['Age'].fillna(ost_df['Age'].mean())

ost_df['Fare'] = ost_df['Fare'].fillna(ost_df['Fare'].mean())

exp_df['Age_group'] = pd.qcut(exp_df['Age'], q=5, labels= ['0', '1', '2', '3', '4'])

exp_df['Age_group'] = exp_df['Age_group'].cat.codes

exp_df['Fare_cut'] = pd.DataFrame(scaler.fit_transform(exp_df[['Fare']]))

ost_df['Age_group'] = pd.qcut(ost_df['Age'], q=5, labels=['0', '1', '2', '3', '4'])

ost_df['Age_group'] = ost_df['Age_group'].cat.codes

ost_df['Fare_cut'] = pd.DataFrame(scaler.transform(ost_df[['Fare']]))

X = exp_df.drop(['Survived', 'PassengerId', 'Age', 'Fare'], axis=1)

y = exp_df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=91)

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

prev = pd.DataFrame(knn.predict(X_test))

acc = accuracy_score(y_test, prev)

print(f"KNN Classifier accuracy: {acc:.5f}")

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X,y, test_size=0.2, random_state=68)

lr = LogisticRegression(max_iter=1000)

lr.fit(X_train_lr, y_train_lr)

lr_pred = lr.predict(X_test_lr)

lr_acc = accuracy_score(y_test_lr, lr_pred)

print(f"Logistic Regression accuracy: {lr_acc:.5f}")


X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(X,y, test_size=0.2, random_state=73)

nb = GaussianNB()

nb.fit(X_train_nb, y_train_nb)

nb_pred = nb.predict(X_test_nb)

nb_acc = accuracy_score(y_test_nb, nb_pred)

print(f"Gaussian accuracy: {nb_acc:.5f}")







