import numpy as np
from scipy.stats import norm
from .KernelLogisticRegression import KernelLogisticRegression


class IEROS():
    """
    Parameters
    ====================
    :X: Input training data points of size num_of_inputs
    :solutions_y: The set of outlier scores for each solution. Shape in num_of_solutions x num_of_inputs
    :solutions_label: The set of labels for input data points in each solution.
                      0 for outliers and 1 for inliners.
                      Shape in num_of_solutions x num_of_inputs

    -----------------------
    Warning: Ensure that output scores y_i in y should indicate the likelihood of X_i being an anomaly
    """
    def __init__(self, X, solutions_y, solutions_label):
        self.X = X
        self.y = solutions_y
        self.data_dim = self.X.shape[0]
        self.solution_dim = self.y.shape[0]
        self.label = solutions_label
        self.ws = self.compute_w(self.y)

    def compute_w(self, ys):
        # Compute the set of normalized weights for each solution based on its scores

        ws = []
        for y in ys:
            ws.append([max(0, 2 * norm.cdf(y_j, loc=y.mean(), scale=y.std()) - 1) for y_j in y])
        return np.array(ws)

    def findMaxGamma(self, max_iter=1e7, gamma=None, gammaIncreaseRate=1.1):
        """
        Finding maximum gamma to generate the set of discrete gamma values for final AUC computation.
        The formula for finding gamma_max is as follows:
        >> gamma_max = gamma | KLR(X, x_j, w, m_cl, gamma) > 0.5, for all x_j | w_j > 0.5 for all omega in Omega
        """
        iter_total = 0

        # set initial gamma to 1 / num_of_input if not specified
        if not gamma:
            gamma = 1 / self.data_dim

        # iterate each
        for i in range(self.solution_dim):
            outlier_index = []
            w = self.ws[i]
            for j in range(self.data_dim):
                if w[j] > 0.5:
                    outlier_index.append(j)

            while True:
                iter_total += 1

                # Fail to converge, the solution must be bad
                if iter_total > max_iter:
                    raise Exception("Max iter runs out, failing to get gamma")
                is_max = True

                # find KLR(X, x_j, w, m_cl, gamma) > 0.5
                klr = KernelLogisticRegression(X=self.X, y=self.label[i], w=w, gamma=gamma)
                klr.train()
                for idx in outlier_index:
                    if klr.predict(idx) <= 0.5:
                        is_max = False
                        break
                if is_max:
                    break
                gamma *= gammaIncreaseRate

        return gamma

    def weightedAverage(prob, w):
        # produce weighted average for a single gamma
        return prob @ w

    def AUC(self, setOfGamma, label, w):
        # Compute AUC of current solution from the set of gamma generated and the normalized weights
        total = 0
        for gamma in setOfGamma:
            klr = KernelLogisticRegression(X=self.X, y=label, w=w, gamma=gamma)
            klr.train()
            prob = klr.predict_all()
            total += self.weightedAverage(prob, w)
        total /= len(setOfGamma)

        return total

    def get_IEROS(self):
        # Compute IEROS values for all solutions
        scores = []
        maxGamma = self.findMaxGamma()
        setOfGamma = np.linspace(1e-5, maxGamma, num=100)
        for i in range(self.solution_dim):
            scores.append(self.AUC(setOfGamma, self.label[i], self.ws[i]))
        return np.array(scores)
