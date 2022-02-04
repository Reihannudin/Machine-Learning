
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

# Menggunakan dataset iris dari library sklearn
iris = datasets.load_iris()

# Mendefinisikan variable X dengan iris dataset
# dan y dengan iris target y 
X,y = iris.data , iris.target

# Membuat beberapa sample Training dan Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=1234)

# Print shape dari X__train
print(X_train.shape)
# (120, 4) 120 = banyaknya sample, 4 = banyaknya jumlah fitur
print(X_train[0])

# print shape y_train 
print(y_train.shape)
# (120, ) artinya kita memiliki sample sebanyak 120
print(y_train)

# membuat Plot
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, edgecolor='k', s=20)
plt.show()

# Membaut KNN
listA = [1,1,1,1,2,2,2,3,3,4,5,6,7]
from collections import Counter
most_common = Counter(listA).most_common(1)

print(most_common)
# [(1, 4)] angka 1 memiliki 4 sample

print(most_common[0][0])
# (1) 


# ==========================================

# Menimport class KNN dari KNN.py y
from knn import knn

# membuat variable clf dengan Class KNN
clf = knn(k=3)
# menyesuailam X_train dan y_train
clf.fit(X_train, y_train)
# membuat prdiksi dari X_test
predictions = clf.predict(X_test)

# memeriksa keakuratan prediction
acc = np.sum(predictions == y_test) / len(y_test)
print("Accuracy",  acc)