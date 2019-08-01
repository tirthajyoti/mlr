# mlr (`pip install mlr`)

A lightweight, easy-to-use Python package that combines the `scikit-learn`-like simple API with the power of **statistical inference tests**, **visual residual analysis**, **outlier visualization**, **multicollinearity test**, found in packages like `statsmodels` and R language.

Authored and maintained by **Dr. Tirthajyoti Sarkar ([Website](https://tirthajyoti.github.io), [LinkedIn profile](https://www.linkedin.com/in/tirthajyoti-sarkar-2127aa7/))**

### Useful regression metrics,
* MSE, SSE, SST 
* R^2, Adjusted R^2

### Inferential statistics,
* Standard errors
* Confidence intervals
* p-values 
* t-test values 
* F-statistic
* AIC (Akaike Information Criterion), and BIC (Bayesian Information Criterion)

### Visual residual analysis,
* Plots of fitted vs. features, 
* Plot of fitted vs. residuals, 
* Histogram of standardized residuals
* Q-Q plot of standardized residuals

### Outlier detection
* Influence plot
* Cook's distance plot

### Multicollinearity
* Pairplot
* Variance infletion factors (VIF)
* Covariance matrix
* Correlation matrix
* Correlation matrix heatmap

## Install

```
pip install mlr
```

## Quick Start

Import the `MyLinearRegression` class,

```
from MLR import MyLinearRegression as mlr
import numpy as np
```

Generate some random data

```
num_samples=40
num_dim = 5
X = 10*np.random.random(size=(num_samples,num_dim))
coeff = np.array([2,-3.5,1.2,4.1,-2.5])
y = np.dot(coeff,X.T)+10*np.random.randn(num_samples)
```

Make a model instance,

```
model = mlr()
```

Ingest the data

```
model.ingest_data(X,y)
```

Fit,

```
model.fit()
```

## This looks like Scikit-learn's estimator. What's special?

So far, it looks similar to the linear regression estimator of Scikit-Learn, doesn't it?
<br>Here comes the difference,

### Print all kinds of regression model metrics,

```
print ("R-squared: ",model.r_squared())
print ("Adjusted R-squared: ",model.adj_r_squared())
print("MSE: ",model.mse())

>> R-squared:  0.8191595335258663
>> Adjusted R-squared:  0.7925653472796703
>> MSE:  66.47972366962215

```

### Or, print all the metrics at once!

```
model.print_metrics()

>> sse:     3264.0067
   sst:     12971.2524
   mse:     81.6002
   r^2:     0.7484
   adj_r^2: 0.7114
   AIC:     301.5883
   BIC:     311.7216
```

