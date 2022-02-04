# Mengimport library 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


# Membuat Sample X an labels Y
# n_sample yang berarti jumlah sample 
# n_features adalah julmah fitur untuk setiap sample
X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)

# membagi sample jadi hanya 80% sample yang digunakan
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Membuat visualisasi
fig = plt.figure(figsize=(8,6))
plt.scatter(X[:,0], y , color="b", marker="o", s=30)
# plt.show()

# menghitung jumlah sample
# print(X_train.shape)
# print(y_train.shape)


from linear_regression import LinearRegression

# mendefinisikan r2score
def r2_score(y_true, y_pred):
    corr_matrix = np.corrcoef(y_true, y_pred)
    corr = corr_matrix[0, 1]
    return corr ** 2

regressor = LinearRegression(lr = 0.01, n_iters=1000)
regressor.fit(X_train, y_train)
predicted = regressor.predict(X_test)

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

mse_value = mse(y_test, predicted)
print(mse_value)

acc = r2_score(y_test, predicted)
print("Accuracy:", acc)


y_pred_line = regressor.predict(X)
cmap = plt.get_cmap("viridis")
fig = plt.figure(figsize=(8,6))
m1 = plt.scatter(X_train, y_train, color=cmap(0.9), s=10)
m2 = plt.scatter(X_test, y_test, color=cmap(0.5), s=10)
plt.plot(X,y_pred_line,color='black', linewidth=2, label="Prediction")
plt.show()