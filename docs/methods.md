## Full list of methods available (Version 0.1.0)

Although the following list of methods is given per module basis, that is just to reflect the separate focus areas. Keeping with the goal of simple API, all of these modules are internally inherited in the main class. 

So, an user can just declare `model = mlr()`, fit the model with some data, `model.fit(X,y)` and then call any of the following methods on the same object `model`. You do not need to call separate modules, at this point.

### `Data_plots` module

* `corrplot()`: Creates a heatmap of the correlation matrix

* `pairplot()`: Creates pairplot of all variables and the target using the `Seaborn` library

* `plot_fitted()`: Plots fitted values against the true output values from the data

### `Diagnostics_plots` module

* `fitted_vs_residual()`: lots fitted values vs. residuals

* `fitted_vs_features()`: Plots residuals vs all feature variables in a grid

* `histogram_resid()`: Plots a histogram of the residuals (by default, normalized)

* `qqplot_resid()`: Creates a quantile-quantile (Q-Q) plot for (standardized) residuals comparing with a normal distribution

* `shapiro_test()`: Performs Shapiro-Wilk normality test on the residuals (by default, normalized)

### `Inference` module

* `std_err()`: Returns standard error values of the features after fitting regression model

* `pvalues()`: Returns p-values of the features after fitting regression model

* `tvalues()`: Returns t-test statistics of the features after fitting regression model

* `ftest()`: Returns the F-statistic of the overall regression and corresponding p-value (as a tuple)

* `conf_int()`: Computes the confidence interval for the given variable(s), passed on as `cols`. Default confidence level is set at 0.05, which can be changed by the user using `alpha` argument.

### `Metrics` module

* `sse()`: Returns sum of squared errors (model vs. actual)

* `sst()`: Returns total sum of squared errors (actual vs avg(actual))

* `r_squared()`: Returns calculated value of r^2 (coefficient of regression)

* `adj_r_squared()`: Returns calculated value of adjusted r^2

* `mse()`: Returns calculated value of mean-square error (MSE)

* `aic()`: Returns AIC (Akaike information criterion)

* `bic()`: Returns BIC (Bayesian information criterion)

* `print_metrics()`: Prints a report of the useful metrics for a given model object

* `summary_metrics()`: Returns a dictionary of the useful metrics

### `Multicollinearity` module

* `corrcoef()`: Returns the correlation coefficient matrix for the features

* `covar()`: Returns the covariance matrix for the features

* `vif()`: Computes variance influence factors for each feature variable

### `Outliers` module

* `cook_distance()`: Computes and plots Cook's distance

* `influence_plot()`: Creates the influence plot

* `leverage_resid_plot()`: Plots leverage vs normalized residuals' square

**More features will be added in the future releases!**