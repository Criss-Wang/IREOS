import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from scipy.optimize import minimize


class KernelLogisticRegression:
    """
    Kernel logistic regression to compute p(x_j, gamma) based on current solution.
    The solution is passed in as a combination of (X,y,w).

    The method is modified from the code from https://github.com/homarques/ireos-extension.

    Parameters
    =============
    :X:
    :y: set of labels for input data points
    :w: Set of normalized weights
    :gamma: scale value for the kernel
    :mCl: Size of clump. Default = 12
    :C: Regularization constant to be adjusted by the normalized weight for each point.
        The individual constant takes a value of C * (1/mCl) ^ w_i.
        Default = 100.
    """
    def __init__(self, X, y, w, gamma=None, mCl=12, C=100):
        self.mCl = mCl
        self.X = X
        self.y = y
        self.w = w

        self.data_dim = X.shape[0] # size of data points

        if not gamma:
            # initialize gamma to be 1 / number of points
            self.gamma = 1 / self.data_dim
        else:
            self.gamma = gamma

        # precompute the entire kernel matrix
        # we don't worry about the prediction at new points. So current matrix would suffice
        # Try not to apply it on data points >= 100k (may overflow)
        self.kernel = rbf_kernel(X, gamma=gamma)

        # Initialize model parameters
        self.alphas = [1] * (self.data_dim + 1)

        # Compute regularization constants for each data point
        self.c = np.array([C * (1 / mCl) ** w[i] for i in range(len(w))])

    def logSig(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, i):
        # Main step in KLR model with the precomputed kernel. We only extract relevant rows here
        total = self.kernel[i] @ self.alphas[:-1]

        # Offset value
        total += self.alphas[-1]
        return self.logSig(total)

    def loss(self, alphas):
        # Define the loss function to be Cross Entropy Loss regularized by c_i * xi_i
        # and xi_i is the regularization value
        def predict_with_current_alphas(i):
            total = self.kernel[i] @ alphas[:-1]
            total += alphas[-1]
            return self.logSig(total)

        y_inverse = abs(1 - self.y)
        pred = np.array([predict_with_current_alphas(i) for i in range(self.data_dim)])
        err = self.y[pred > 0] @ np.log(pred[pred > 0]) + y_inverse[pred < 1] @ np.log(1 - pred[pred < 1])
        err = err * -1
        reg = (self.c * pred) @ pred
        return err + reg

    def train(self):
        # Using scipy's minimize package with the L-BFGS-B solver
        res = minimize(self.loss, self.alphas, method='L-BFGS-B', options={'gtol': 1e-5, 'disp': True})

        # save the optimized alphas
        self.alphas = res['x']
        print(res)

    def predict_all(self):
        return np.array([self.predict(i) for i in range(self.data_dim)])
