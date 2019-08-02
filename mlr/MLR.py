from mlr.Metrics import Metrics
from mlr.Inference import Inference
from mlr.Diagnostics_plots import Diagnostics_plots
from mlr.Data_plots import Data_plots
from mlr.Outliers import Outliers
from mlr.Multicollinearity import Multicollinearity

import numpy as np
from pandas.api.types import is_numeric_dtype

class MyLinearRegression(Metrics, Inference, 
                        Diagnostics_plots, Data_plots, 
                        Outliers, Multicollinearity
                        ):
    def __init__(self, fit_intercept=True):
        self.coef_ = None
        self.intercept_ = None
        self.fit_intercept_ = fit_intercept
        self.is_fitted = False
        self.is_ingested = False
        self.features_ = None
        self.target_ = None

    def __repr__(self):
        return "I am a Linear Regression model!"

    def ingest_data(self, X, y):
        """
       Ingests the given data
        
        Arguments:
        X: 1D or 2D numpy array 
        y: 1D numpy array
        """
        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)

        # features and data
        self.features_ = X
        self.target_ = y
        self.is_ingested = True

    def fit(self, X=None, y=None, fit_intercept_=True):
        """
        Fit model coefficients.
        Arguments:
        X: 1D or 2D numpy array 
        y: 1D numpy array
        """

        if X != None:
            if len(X.shape) == 1:
                X = X.reshape(-1, 1)
            self.features_ = X
        if y != None:
            self.target_ = y

        # degrees of freedom of population dependent variable variance
        self.dft_ = self.features_.shape[0] - 1
        # degrees of freedom of population error variance
        self.dfe_ = self.features_.shape[0] - self.features_.shape[1] - 1

        # add bias if fit_intercept is True
        if self.fit_intercept_:
            X_biased = np.c_[np.ones(self.features_.shape[0]), self.features_]
        else:
            X_biased = self.features_
        # Assign target_ to a local variable y
        y = self.target_

        # closed form solution
        xTx = np.dot(X_biased.T, X_biased)
        inverse_xTx = np.linalg.inv(xTx)
        xTy = np.dot(X_biased.T, y)
        coef = np.dot(inverse_xTx, xTy)

        # set attributes
        if self.fit_intercept_:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coef

        # Predicted/fitted y
        self.fitted_ = np.dot(self.features_, self.coef_) + self.intercept_

        # Residuals
        residuals = self.target_ - self.fitted_
        self.resid_ = residuals

        # Set is_fitted to True
        self.is_fitted = True

    
    def fit_dataframe(self, X, y, dataframe, fit_intercept_= True):
        """
        Fit model coefficients from a Pandas DataFrame.
        
        Arguments:
        X: A list of columns of the dataframe acting as features. Must be only numerical.
        y: Name of the column of the dataframe acting as the target
        fit_intercept: Boolean, whether an intercept term will be included in the fit
        """
        
        # Code to check type of X and y arguments
        assert (
            type(X) == list
        ), "X must be a list of the names of the numerical feature/predictor columns"
        assert (
            type(y) == str
        ), "y must be a string - name of the column you want as target"
        
        # Code to check numeric data type
        is_numeric = True
        for feature in X:
            if not is_numeric_dtype(dataframe[feature]):
                is_numeric = False
                break
        if not is_numeric_dtype(dataframe[y]):
                is_numeric = False
        
        if not is_numeric:
            raise TypeError('Either one or more features or the target is not of numeric type')
            return None
            
        self.features_ = np.array(dataframe[X])
        self.target_ = np.array(dataframe[y])

        # degrees of freedom of population dependent variable variance
        self.dft_ = self.features_.shape[0] - 1
        # degrees of freedom of population error variance
        self.dfe_ = self.features_.shape[0] - self.features_.shape[1] - 1

        # add bias if fit_intercept is True
        if self.fit_intercept_:
            X_biased = np.c_[np.ones(self.features_.shape[0]), self.features_]
        else:
            X_biased = self.features_
        # Assign target_ to a local variable y
        y = self.target_

        # closed form solution
        xTx = np.dot(X_biased.T, X_biased)
        inverse_xTx = np.linalg.inv(xTx)
        xTy = np.dot(X_biased.T, y)
        coef = np.dot(inverse_xTx, xTy)

        # set attributes
        if self.fit_intercept_:
            self.intercept_ = coef[0]
            self.coef_ = coef[1:]
        else:
            self.intercept_ = 0
            self.coef_ = coef

        # Predicted/fitted y
        self.fitted_ = np.dot(self.features_, self.coef_) + self.intercept_

        # Residuals
        residuals = self.target_ - self.fitted_
        self.resid_ = residuals

        # Set is_fitted to True
        self.is_fitted = True

    def predict(self, X):
        """Output model prediction.
        Arguments:
        X: 1D or 2D numpy array
        """
        # check if X is 1D or 2D array
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        self.predicted_ = self.intercept_ + np.dot(X, self.coef_)
        return self.predicted_

    def run_diagnostics(self):
        """Runs diagnostics tests and plots"""
        Diagnostics_plots.fitted_vs_residual(self)
        Diagnostics_plots.histogram_resid(self)
        Diagnostics_plots.qqplot_resid(self)
        print()
        Diagnostics_plots.shapiro_test(self)

    def outlier_plots(self):
        """Creates various outlier plots"""
        Outliers.cook_distance(self)
        Outliers.influence_plot(self)
        Outliers.leverage_resid_plot(self)