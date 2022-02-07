from sklearn.datasets import make_blobs
from kmean import KMeans
import numpy as np

X, y = make_blobs(
    centers=4, n_samples=500, n_features=2, shuffle=True, random_state=42
)
print(X.shape)

clusters = len(np.unique(y))
print(clusters)

k = KMeans(K=clusters, max_iters=150, plot_steps=True)
y_pred = k.predict(X)

k.plot()
