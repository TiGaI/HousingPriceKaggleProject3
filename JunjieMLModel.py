### import necessary packages
import pandas as pd
import numpy as np
from scipy import stats


#Import Train and Test set
train_df = pd.read_csv('https://raw.githubusercontent.com/TiGaI/HousingPriceKaggleProject3/xiangwei/data/cleaned_train.csv')
# test_df = pd.read_csv('https://raw.githubusercontent.com/TiGaI/HousingPriceKaggleProject3/xiangwei/data/cleaned_test.csv')

train_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
# test_df.drop(['Unnamed: 0'], axis = 1, inplace = True)

### change the dtypes of the 'GarageQuality' and 'GarageAge'
train_df['GarageQuality'] = train_df['GarageQuality'].astype('int64')
train_df['GarageAge'] = train_df['GarageAge'].astype('int64')
# train_df['GarageQuality'] = train_df['GarageQuality'].astype('int64')
# train_df['GarageAge'] = train_df['GarageAge'].astype('int64')


numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
# Check the skew of all numerical features
skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
#normalize columns greater than 0.8 skewness
skewness = skewness[abs(skewness) > 0.80]

for feat in skewed_feats:
	train_df[feat], fitted_lambda = stats.boxcox(train_df[feat])
	test_df[feat] = stats.boxcox(test_df[feat], fitted_lambda)

#Cross-Validation function
n_folds = 5

def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)


# Here is the list of such models:

# linear_model.ElasticNetCV([l1_ratio, eps, …])	Elastic Net model with iterative fitting along a regularization path.
# linear_model.LarsCV([fit_intercept, …])	Cross-validated Least Angle Regression model.
# linear_model.LassoCV([eps, n_alphas, …])	Lasso linear model with iterative fitting along a regularization path.
# linear_model.LassoLarsCV([fit_intercept, …])	Cross-validated Lasso, using the LARS algorithm.
# linear_model.LogisticRegressionCV([Cs, …])	Logistic Regression CV (aka logit, MaxEnt) classifier.
# linear_model.MultiTaskElasticNetCV([…])	Multi-task L1/L2 ElasticNet with built-in cross-validation.
# linear_model.MultiTaskLassoCV([eps, …])	Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.
# linear_model.OrthogonalMatchingPursuitCV([…])	Cross-validated Orthogonal Matching Pursuit model (OMP).
# linear_model.RidgeCV([alphas, …])	Ridge regression with built-in cross-validation.
# linear_model.RidgeClassifierCV([alphas, …])	Ridge classifier with built-in cross-validation.

### MODEL IMPORTS
# LINEAR MODELcd
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, BayesianRidge, LassoLarsIC, KernelRidge

#Grid Search
lasso_params = {'alpha':[0.02, 0.024, 0.025, 0.026, 0.03]}
ridge_params = {'alpha':[200, 230, 250,265, 270, 275, 290, 300, 500]}
eNet_params = {"max_iter": [1, 5, 10, 100, 500, 100],
                      "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                      "l1_ratio": np.arange(0.0, 1.0, 0.1)}
bayesianRidge_params = {"n_iter": [100, 200, 300],
						"alpha_1": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
						"alpha_2": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
						"lambda_1": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
						"lambda_2": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
}

#Base Regression Model:
baseRegression = LinearRegression().fit(X, y)

#Grid Search
from sklearn.model_selection import GridSearchCV
lasso_Best = GridSearchCV(Lasso(), param_grid=lasso_params).fit(X, y)
lasso_Best.best_estimator_

ridge_Best = GridSearchCV(ridge(), param_grid=ridge_params).fit(X, y)
ridge_Best.best_estimator_

eNet_Best = GridSearchCV(ElasticNet(), parametersGrid, scoring='r2', cv=5).fit(X, y)
eNet_Best.best_estimator_

BayesianRidge_Best = GridSearchCV(BayesianRidge(), bayesianRidge_params, scoring='r2', cv=5).fit(X, y)
eNet_Best.best_estimator_

KRR = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)



from sklearn.isotonic import IsotonicRegression

ir = IsotonicRegression()
y_ir = ir.fit_transform()





# Tree Base Model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

