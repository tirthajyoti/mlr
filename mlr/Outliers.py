import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

class Outliers:
    """
    Methods for plotting outliers, leverage, influence points
    
    cook_distance: Computes and plots Cook's distance
    influence_plot: Creates the influence plot
    leverage_resid_plot: Plots leverage vs normalized residuals' square
    """

    def __init__():
        pass

    def cook_distance(self):
        """Computes and plots Cook\'s distance"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        import statsmodels.api as sm
        from statsmodels.stats.outliers_influence import OLSInfluence as influence

        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        inf = influence(lm)
        (c, p) = inf.cooks_distance
        plt.figure(figsize=(8, 5))
        plt.title("Cook's distance plot for the residuals", fontsize=14)
        plt.stem(np.arange(len(c)), c, markerfmt=",", use_line_collection=True)
        plt.grid(True)
        plt.show()

    def influence_plot(self):
        """Creates the influence plot"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        import statsmodels.api as sm

        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        fig, ax = plt.subplots(figsize=(10, 8))
        fig = sm.graphics.influence_plot(lm, ax=ax, criterion="cooks")
        plt.show()

    def leverage_resid_plot(self):
        """Plots leverage vs normalized residuals' square"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        import statsmodels.api as sm

        lm = sm.OLS(self.target_, sm.add_constant(self.features_)).fit()
        fig, ax = plt.subplots(figsize=(10, 8))
        fig = sm.graphics.plot_leverage_resid2(lm, ax=ax)
        plt.show()