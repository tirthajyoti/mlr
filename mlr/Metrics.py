import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

class Metrics:
    """
    Methods for computing useful regression metrics
    
    sse: Sum of squared errors
    sst: Total sum of squared errors (actual vs avg(actual))
    r_squared: Regression coefficient (R^2)
    adj_r_squared: Adjusted R^2
    mse: Mean sum of squared errors
    AIC: Akaike information criterion
    BIC: Bayesian information criterion
    """
 
    def sse(self):
        """Returns sum of squared errors (model vs. actual)"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        squared_errors = (self.resid_) ** 2
        self.sq_error_ = np.sum(squared_errors)
        return self.sq_error_

    def sst(self):
        """Returns total sum of squared errors (actual vs avg(actual))"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        avg_y = np.mean(self.target_)
        squared_errors = (self.target_ - avg_y) ** 2
        self.sst_ = np.sum(squared_errors)
        return self.sst_

    def r_squared(self):
        """Returns calculated value of r^2"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        self.r_sq_ = 1 - self.sse() / self.sst()
        return self.r_sq_

    def adj_r_squared(self):
        """Returns calculated value of adjusted r^2"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        self.adj_r_sq_ = 1 - (self.sse() / self.dfe_) / (self.sst() / self.dft_)
        return self.adj_r_sq_

    def mse(self):
        """Returns calculated value of mse"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        self.mse_ = np.mean((self.predict(self.features_) - self.target_) ** 2)
        return self.mse_

    def aic(self):
        """
        Returns AIC (Akaike information criterion)
        """
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        return lm.aic

    def bic(self):
        """
        Returns BIC (Bayesian information criterion)
        """
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        return lm.bic

    def print_metrics(self):
        """Prints a report of the useful metrics for a given model object"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        items = (
            ("sse:", self.sse()),
            ("sst:", self.sst()),
            ("mse:", self.mse()),
            ("r^2:", self.r_squared()),
            ("adj_r^2:", self.adj_r_squared()),
            ("AIC:", self.aic()),
            ("BIC:", self.bic()),
        )
        for item in items:
            print("{0:8} {1:.4f}".format(item[0], item[1]))

    def summary_metrics(self):
        """Returns a dictionary of the useful metrics"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        metrics = {}
        items = (
            ("sse", self.sse()),
            ("sst", self.sst()),
            ("mse", self.mse()),
            ("r^2", self.r_squared()),
            ("adj_r^2:", self.adj_r_squared()),
            ("AIC:", self.aic()),
            ("BIC:", self.bic()),
        )
        for item in items:
            metrics[item[0]] = item[1]
        return metrics