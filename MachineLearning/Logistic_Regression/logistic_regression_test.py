import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt

from logistic_regression import LogisticRegression

df = datasets.load_breast_cancer()
X , y = df.data, df.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=1234)

def accuracy(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred)/ len(y_true)
    return accuracy

regressor = LogisticRegression(learning_rate=0.0001, num_iters=1000)
regressor.fit(X_train, y_train)
preditions = regressor.predict(X_test)

print("LR classification accuracy: ", accuracy(y_test, preditions))

