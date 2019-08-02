import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from seaborn import heatmap
from mlr.Multicollinearity import Multicollinearity 

class Data_plots:
    """
    Methods for data related plots
    
    pairplot: Creates pairplot of all variables and the target
    plot_fitted: Plots fitted values against the true output values from the data
    """

    def __init__():
        pass
    
    def corrplot(self,cmap=None,annot=True):
        """
        Creates a heatmap of the correlation matrix
        
        Arguments:
        annot: Bool. Whether to write the correlation coefficient in each cell
               Default: True
        cmap : matplotlib colormap name or object, or list of colors, optional
        """
        heatmap(Multicollinearity.corrcoef(self),cmap=cmap,annot=annot)
        plt.show()
    
    def pairplot(self):
        """Creates pairplot of all variables and the target using the Seaborn library"""
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None

        print("This may take a little time. Have patience...")
        from seaborn import pairplot
        from pandas import DataFrame

        df = DataFrame(np.hstack((self.features_, self.target_.reshape(-1, 1))))
        pairplot(df)
        plt.show()

    def plot_fitted(self, reference_line=False):
        """
        Plots fitted values against the true output values from the data
        
        Arguments:
        reference_line: A Boolean switch to draw a 45-degree reference line on the plot
        """
        if not self.is_fitted:
            print("Model not fitted yet!")
            return None
        plt.title("True vs. fitted values", fontsize=14)
        plt.scatter(y, self.fitted_, s=100, alpha=0.75, color="red", edgecolor="k")
        if reference_line:
            plt.plot(y, y, c="k", linestyle="dotted")
        plt.xlabel("True values")
        plt.ylabel("Fitted values")
        plt.grid(True)
        plt.show()