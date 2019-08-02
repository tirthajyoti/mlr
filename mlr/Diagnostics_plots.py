import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

class Diagnostics_plots:
    """
    Diagnostics plots and methods
    
    Arguments:
    fitted_vs_residual: Plots fitted values vs. residuals
    fitted_vs_features: Plots residuals vs all feature variables in a grid
    histogram_resid: Plots a histogram of the residuals (can be normalized)
    shapiro_test: Performs Shapiro-Wilk normality test on the residuals
    qqplot_resid: Creates a quantile-quantile plot for residuals comparing with a normal distribution    
    """

    def __init__():
        pass

    def fitted_vs_residual(self):
        """Plots fitted values vs. residuals"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        plt.title("Fitted vs. residuals plot", fontsize=14)
        plt.scatter(self.fitted_, self.resid_, edgecolor="k")
        plt.hlines(
            y=0,
            xmin=np.amin(self.fitted_),
            xmax=np.amax(self.fitted_),
            color="k",
            linestyle="dashed",
        )
        plt.xlabel("Fitted values")
        plt.ylabel("Residuals")
        plt.show()

    def fitted_vs_features(self):
        """Plots residuals vs all feature variables in a grid"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        num_plots = self.features_.shape[1]
        if num_plots % 3 == 0:
            nrows = int(num_plots / 3)
        else:
            nrows = int(num_plots / 3) + 1
        ncols = 3
        fig, ax = plt.subplots(nrows, ncols, figsize=(15, nrows * 3.5))
        axes = ax.ravel()
        for i in range(num_plots, nrows * ncols):
            axes[i].set_visible(False)
        for i in range(num_plots):
            axes[i].scatter(
                self.features_.T[i],
                self.resid_,
                color="orange",
                edgecolor="k",
                alpha=0.8,
            )
            axes[i].grid(True)
            axes[i].set_xlabel("Feature X[{}]".format(i))
            axes[i].set_ylabel("Residuals")
            axes[i].hlines(
                y=0,
                xmin=np.amin(self.features_.T[i]),
                xmax=np.amax(self.features_.T[i]),
                color="k",
                linestyle="dashed",
            )
        plt.show()

    def histogram_resid(self, normalized=True):
        """Plots a histogram of the residuals (can be normalized)"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        if normalized:
            norm_r = self.resid_ / np.linalg.norm(self.resid_)
        else:
            norm_r = self.resid_
        num_bins = min(20, int(np.sqrt(self.features_.shape[0])))
        plt.title("Histogram of the normalized residuals")
        plt.hist(norm_r, bins=num_bins, edgecolor="k")
        plt.xlabel("Normalized residuals")
        plt.ylabel("Count")
        plt.show()

    def shapiro_test(self, normalized=True):
        """Performs Shapiro-Wilk normality test on the residuals"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        from scipy.stats import shapiro

        if normalized:
            norm_r = self.resid_ / np.linalg.norm(self.resid_)
        else:
            norm_r = self.resid_
        _, p = shapiro(norm_r)
        if p > 0.01:
            print("The residuals seem to have come from a Gaussian process")
        else:
            print(
                "The residuals does not seem to have come from a Gaussian process.\nNormality assumptions of the linear regression may have been violated."
            )

    def qqplot_resid(self, normalized=True):
        """Creates a quantile-quantile plot for residuals comparing with a normal distribution"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        from scipy.stats import probplot

        if normalized:
            norm_r = self.resid_ / np.linalg.norm(self.resid_)
        else:
            norm_r = self.resid_
        plt.title("Q-Q plot of the normalized residuals")
        probplot(norm_r, dist="norm", plot=plt)
        plt.xlabel("Theoretical quantiles")
        plt.ylabel("Residual quantiles")
        plt.show()