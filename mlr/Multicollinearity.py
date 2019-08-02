import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

class Multicollinearity:
    """
    Methods for checking multicollinearity in the dataset features
    
    vif:Computes variance influence factors for each feature variable
    """

    def __init__():
        pass
    
    def corrcoef(self):
        """Returns the correlation coefficient matrix for the features"""
        if not self.is_ingested:
            print("No data ingested or fitted yet!")
            return None
        return np.corrcoef(self.features_.T)
    
    def covar(self):
        """Returns the covariance matrix for the features"""
        if not self.is_ingested:
            print("No data ingested or fitted yet!")
            return None
        return np.cov(self.features_.T)
    
    def vif(self):
        """Computes variance influence factors for each feature variable"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import (
            variance_inflation_factor as vif,
        )

        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        for i in range(self.features_.shape[1]):
            v = vif(np.matrix(self.features_), i)
            print("Variance inflation factor for feature {}: {}".format(i, round(v, 2)))