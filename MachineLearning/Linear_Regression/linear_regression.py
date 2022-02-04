import numpy as np

class LinearRegression:
    
    # mendefinisikan r2score
    def r2_score(y_true, y_pred):
        corr_matrix = np.corrcoef(y_true, y_pred)
        corr = corr_matrix[0, 1]
        return corr ** 2
    
# Membuat fungsi init, lr adalah jumlah default dari lr,
# n_iters adalah jumlah banyaknya iterasi yang digunakan
    def __init__(self,lr = 0.001, n_iters = 1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        
# Mendefinisikan fungsi fit yang dimana akan menjadi model 
# X menjadi sample
# dan y menjadi label
    def fit(self, X, y):
        n_samples , n_features = X.shape
        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            dw = (1/n_samples) + np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) + np.sum(y_predicted - y)
            
            self.weights -= self.lr * dw 
            self.bias -= self.lr * db
            
# Mendefinisikan fungsi predict yang dimana akan menghasilkan 
# sample baru yang akan memperkirakan nilai dan mengembalikan nilai
    def predict(self, X):
        y_predicted = np.dot(X, self.weights) + self.bias
        return y_predicted
        

    
