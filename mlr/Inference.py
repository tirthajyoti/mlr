import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

class Inference:
    """
    Inferential statistics: 
        standard error, 
        p-values
        t-test statistics
        F-statistics and p-value of F-test
        Confidence interval
    """

    def __init__():
        pass

    def std_err(self):
        """
        Returns standard error values of the features
        """
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        return lm.bse

    def pvalues(self):
        """
        Returns p-values of the features
        """
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        return lm.pvalues

    def tvalues(self):
        """
        Returns t-test values of the features
        """
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        return lm.tvalues

    def ftest(self):
        """
        Returns the F-statistic of the overall regression and corresponding p-value
        """
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        return (lm.fvalue, lm.f_pvalue)
    
    def conf_int(self,cols=None,alpha=0.05):
        """
        Computes the confidence interval for the given variable(s)
        
        Arguments:
        cols: List of the columns (features) for which confidence interval is sought
        alpha: Confidence level. Default is 0.05
        """
        
        assert alpha>0 and alpha < 1, "Confidence level either zero, negative, or greater than 1"
        
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        return (lm.conf_int(cols=cols,alpha=alpha))
        