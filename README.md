# Gradient Boosting on Multivariate Regression Methods 
### Rini Gupta

I will be evaluating the efficacy of gradient boosting and extreme gradient boosting in comparison to locally weighted linear regression and random forest regression on its own. To do this, I will use the Boston Housing dataset and the cars dataset to compare the performance on two distinct experiments. As the name suggests, multivariate regression includes multiple input features that are used to predict the response variable, y. I will utilize cross-validation techniques to compare the mean squared errors and mean absolute errors of locally weighted linear regression, random forest regression, boosted locally weighted regression, and extreme gradient boosting. 

In simple terms, boosting takes weak learners and makes them into strong learners (Singh 2018). The trees that are fit are on a modified version of the original data. Gradient boosting is a greedy algorithm that gradually trains many models. Friedman's extreme gradient boosting was developed in 2001 with regularization mechanisms to avoid overfitting (Maklin 2020). Like gradient boosting, extreme gradient boosting is a tree-based algorithm. One of the main strengths of extreme gradient boosting is the speed at which it runs, particularly in comparison to a deep neural network. Extreme gradient boosting is also referred to as XGBoost. XGBoost is theorized to outperform random forest regression. This theory will be examined in later sections of this paper. 

![image](https://user-images.githubusercontent.com/76021844/155905225-1096a2cb-3d04-4a06-ae16-e25534d482c9.png)


### Loading in the Data/Import Necessary Libraries 
```
import numpy as np
import pandas as pd
from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, griddata, LinearNDInterpolator, NearestNDInterpolator
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split as tts
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
import matplotlib.pyplot as plt
from matplotlib import pyplot
import lightgbm as lgb
import xgboost as xgb
```
```
cars = pd.read_csv('cars.csv')
boston = pd.read_csv('Boston Housing Prices.csv')
```

### Locally Weighted Regression and Boosted Version 
We need kernels for the locally weighted linear regression. I will be including options of tricubic, quartic, and Epanechnikov.
```
# Tricubic Kernel
def Tricubic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,70/81*(1-d**3)**3)

# Quartic Kernel
def Quartic(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,15/16*(1-d**2)**2)

# Epanechnikov Kernel
def Epanechnikov(x):
  if len(x.shape) == 1:
    x = x.reshape(-1,1)
  d = np.sqrt(np.sum(x**2,axis=1))
  return np.where(d>1,0,3/4*(1-d**2)) 
```

Here is the implementation of locally weighted linear regression I will be using. 
```
#Defining the kernel local regression model

def lw_reg(X, y, xnew, kern, tau, intercept):
    # tau is called bandwidth K((x-x[i])/(2*tau))
    n = len(X) # the number of observations
    yest = np.zeros(n)

    if len(y.shape)==1: # here we make column vectors
      y = y.reshape(-1,1)

    if len(X.shape)==1:
      X = X.reshape(-1,1)
    
    if intercept:
      X1 = np.column_stack([np.ones((len(X),1)),X])
    else:
      X1 = X

    w = np.array([kern((X - X[i])/(2*tau)) for i in range(n)]) # here we compute n vectors of weights

    #Looping through all X-points
    for i in range(n):          
        W = np.diag(w[:,i])
        b = np.transpose(X1).dot(W).dot(y)
        A = np.transpose(X1).dot(W).dot(X1)
        #A = A + 0.001*np.eye(X1.shape[1]) # if we want L2 regularization
        #theta = linalg.solve(A, b) # A*theta = b
        beta, res, rnk, s = lstsq(A, b)
        yest[i] = np.dot(X1[i],beta)
    if X.shape[1]==1:
      f = interp1d(X.flatten(),yest,fill_value='extrapolate')
    else:
      f = LinearNDInterpolator(X, yest)
    output = f(xnew) # the output may have NaN's where the data points from xnew are outside the convex hull of X
    if sum(np.isnan(output))>0:
      g = NearestNDInterpolator(X,y.ravel()) 
      # output[np.isnan(output)] = g(X[np.isnan(output)])
      output[np.isnan(output)] = g(xnew[np.isnan(output)])
    return output
```

Below, we "boost" the locally weighted linear regression model. 
```
def boosted_lwr(X, y, xnew, kern, tau, intercept):
  Fx = lw_reg(X,y,X,kern,tau,intercept) # we need this for training the Decision Tree
  new_y = y - Fx
  model = RandomForestRegressor(n_estimators=100,max_depth=2)
  model.fit(X,new_y)
  output = model.predict(xnew) + lw_reg(X,y,xnew,kern,tau,intercept)
  return output 
```

Now we will evaluate the regression methods.

## Boston Housing Dataset

First, I select the top three features from the Boston housing dataset and the response variable. 
```
X = boston[['lstat', 'rooms', 'distance']].values
y = boston['cmedv'].values
```

Next, I examine the mean squared errors and mean absolute errors of lowess, boosted lowess, random forest, and extreme gradient boosting. 
```
mse_lwr = []
mse_blwr = []
mse_rf = []
mse_xgb = []
kf = KFold(n_splits=10,shuffle=True,random_state = 310)
scale = StandardScaler()
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
  yhat_lwr = lw_reg(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
  yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
  model_rf = RandomForestRegressor(n_estimators=100,max_depth=3)
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)
  model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
  model_xgb.fit(xtrain,ytrain)
  yhat_xgb = model_xgb.predict(xtest)
  mse_lwr.append(mse(ytest,yhat_lwr))
  mse_blwr.append(mse(ytest,yhat_blwr))
  mse_rf.append(mse(ytest,yhat_rf))
  mse_xgb.append(mse(ytest,yhat_xgb))
print('The Cross-validated Mean Squared Error for LWR is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for BLWR is : '+str(np.mean(mse_blwr)))
print('The Cross-validated Mean Squared Error for RF is : '+str(np.mean(mse_rf)))
print('The Cross-validated Mean Squared Error for XGB is : '+str(np.mean(mse_xgb)))
```
The Cross-validated Mean Squared Error for LWR is : 23.03885019369726

The Cross-validated Mean Squared Error for BLWR is : 21.397098682397974

The Cross-validated Mean Squared Error for RF is : 16.376944002440666

The Cross-validated Mean Squared Error for XGB is : 15.151288412927352

![image](https://user-images.githubusercontent.com/76021844/155891345-3f31df37-0787-430a-8c99-cf19906fc0c7.png)

```
mae_blwr = []
mae_rf = []
mae_xgb = []
kf = KFold(n_splits=10,shuffle=True,random_state = 310)
scale = StandardScaler()
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
  yhat_lwr = lw_reg(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
  yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
  model_rf = RandomForestRegressor(n_estimators=100,max_depth=3)
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)
  model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
  model_xgb.fit(xtrain,ytrain)
  yhat_xgb = model_xgb.predict(xtest)
  mae_lwr.append(mae(ytest,yhat_lwr))
  mae_blwr.append(mae(ytest,yhat_blwr))
  mae_rf.append(mae(ytest,yhat_rf))
  mae_xgb.append(mae(ytest,yhat_xgb))
print('The Cross-validated Mean Absolute Error for LWR is : '+str(np.mean(mae_lwr)))
print('The Cross-validated Mean Absolute Error for BLWR is : '+str(np.mean(mae_blwr)))
print('The Cross-validated Mean Absolute Error for RF is : '+str(np.mean(mae_rf)))
print('The Cross-validated Mean Absolute Error for XGB is : '+str(np.mean(mae_xgb)))
```
The Cross-validated Mean Absolute Error for LWR is : 3.035373115096472

The Cross-validated Mean Absolute Error for BLWR is : 2.984027557777

The Cross-validated Mean Absolute Error for RF is : 2.973486754592399

The Cross-validated Mean Absolute Error for XGB is : 2.749126247980659
![image](https://user-images.githubusercontent.com/76021844/155891465-1f4ad6cc-c3dd-473f-a601-8699d116c054.png)

## Cars Dataset

Again, loading in the data. 

```
X = cars[['ENG','CYL','WGT']].values
y = cars['MPG'].values
```
Repeating the same steps above:

```
mse_lwr = []
mse_blwr = []
mse_rf = []
mse_xgb = []
kf = KFold(n_splits=10,shuffle=True,random_state = 310)
scale = StandardScaler()
  # this is the Cross-Validation Loop
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
  yhat_lwr = lw_reg(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
  yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
  model_rf = RandomForestRegressor(n_estimators=100,max_depth=3)
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)
  model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
  model_xgb.fit(xtrain,ytrain)
  yhat_xgb = model_xgb.predict(xtest)
  mse_lwr.append(mse(ytest,yhat_lwr))
  mse_blwr.append(mse(ytest,yhat_blwr))
  mse_rf.append(mse(ytest,yhat_rf))
  mse_xgb.append(mse(ytest,yhat_xgb))
print('The Cross-validated Mean Squared Error for LWR is : '+str(np.mean(mse_lwr)))
print('The Cross-validated Mean Squared Error for BLWR is : '+str(np.mean(mse_blwr)))
print('The Cross-validated Mean Squared Error for RF is : '+str(np.mean(mse_rf)))
print('The Cross-validated Mean Squared Error for XGB is : '+str(np.mean(mse_xgb)))
```
The Cross-validated Mean Squared Error for LWR is : 17.023884753688034

The Cross-validated Mean Squared Error for BLWR is : 17.146775362988894

The Cross-validated Mean Squared Error for RF is : 17.180808479783916

The Cross-validated Mean Squared Error for XGB is : 16.80923534387059
![image](https://user-images.githubusercontent.com/76021844/155891554-e7007723-ec73-455b-a9e9-b12bffa7b822.png)

```
mae_lwr = []
mae_blwr = []
mae_rf = []
mae_xgb = []
kf = KFold(n_splits=10,shuffle=True,random_state = 310)
scale = StandardScaler()
for idxtrain, idxtest in kf.split(X):
  xtrain = X[idxtrain]
  ytrain = y[idxtrain]
  ytest = y[idxtest]
  xtest = X[idxtest]
  xtrain = scale.fit_transform(xtrain)
  xtest = scale.transform(xtest)
  yhat_lwr = lw_reg(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
  yhat_blwr = boosted_lwr(xtrain,ytrain, xtest,Epanechnikov,tau=0.9,intercept=True)
  model_rf = RandomForestRegressor(n_estimators=100,max_depth=3)
  model_rf.fit(xtrain,ytrain)
  yhat_rf = model_rf.predict(xtest)
  model_xgb = xgb.XGBRegressor(objective ='reg:squarederror',n_estimators=100,reg_lambda=20,alpha=1,gamma=10,max_depth=3)
  model_xgb.fit(xtrain,ytrain)
  yhat_xgb = model_xgb.predict(xtest)
  mae_lwr.append(mae(ytest,yhat_lwr))
  mae_blwr.append(mae(ytest,yhat_blwr))
  mae_rf.append(mae(ytest,yhat_rf))
  mae_xgb.append(mae(ytest,yhat_xgb))
print('The Cross-validated Mean Absolute Error for LWR is : '+str(np.mean(mae_lwr)))
print('The Cross-validated Mean Absolute Error for BLWR is : '+str(np.mean(mae_blwr)))
print('The Cross-validated Mean Absolute Error for RF is : '+str(np.mean(mae_rf)))
print('The Cross-validated Mean Absolute Error for XGB is : '+str(np.mean(mae_xgb)))
```
The Cross-validated Mean Absolute Error for LWR is : 2.99984075010725

The Cross-validated Mean Absolute Error for BLWR is : 3.0259622587687782

The Cross-validated Mean Absolute Error for RF is : 3.017520401629253

The Cross-validated Mean Absolute Error for XGB is : 2.9823656928920412

![image](https://user-images.githubusercontent.com/76021844/155891596-4fe2db10-376b-4804-a29d-78022ce4e1b9.png)


## Conclusion
Extreme gradient boosting proved to the most effective method across both datasets and metrics. Furthermore, boosting lowess did not always improve performance. This trend might indicate that the performance of lowess may not be able to be improved. Random forest performed better on the Boston Housing dataset compared to cars. In conclusion, extreme gradient boosting produced markedly low mean-squared error values and mean-absolute error values, even compared to the random forest regressor. Extreme gradient boosting has the added benefit of runtime speed, which makes it an excellent choice to improve the performance of a regression method.  

### References

Dobilas, S. (2022, February 5). XGBoost: Extreme gradient boosting - how to improve on regular gradient boosting? Medium. Retrieved February 27, 2022, from https://towardsdatascience.com/xgboost-extreme-gradient-boosting-how-to-improve-on-regular-gradient-boosting-5c6acf66c70a 

Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine. The Annals of Statistics, 29(5), 1189–1232. http://www.jstor.org/stable/2699986

Maklin, C. (2020, May 9). XGBoost Python example. Medium. Retrieved February 27, 2022, from https://towardsdatascience.com/xgboost-python-example-42777d01001e 

Multivariate Regression. OARC Stats. (n.d.). Retrieved February 27, 2022, from https://stats.oarc.ucla.edu/stata/dae/multivariate-regression-analysis/ 

Saxena, S. (2021, March 25). Gradient boosting machine: Gradient boosting machine for data science. Analytics Vidhya. Retrieved February 27, 2022, from https://www.analyticsvidhya.com/blog/2021/03/gradient-boosting-machine-for-data-scientists/ 

Sheridan, R. (n.d.). Extreme Gradient Boosting as a Method for Quantitative Structure Activity Relationships. https://doi.org/10.1021/acs.jcim.6b00591.s011 

Singh, H. (2018, November 4). Understanding gradient boosting machines. Medium. Retrieved February 27, 2022, from https://towardsdatascience.com/understanding-gradient-boosting-machines-9be756fe76ab 

