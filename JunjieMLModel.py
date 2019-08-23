### import necessary packages
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import uniform

#Import Train and Test set

train_df = pd.read_csv('https://raw.githubusercontent.com/TiGaI/HousingPriceKaggleProject3/xiangwei/data/train_schrank.csv')

train_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
train_df = train_df.drop(train_df[(train_df['GrLivArea']>4000)].index)
train_df['GarageQuality'] = train_df['GarageQuality'].astype('int64')
train_df['GarageAge'] = train_df['GarageAge'].astype('int64')

# test_df = pd.read_csv('https://raw.githubusercontent.com/TiGaI/HousingPriceKaggleProject3/xiangwei/data/cleaned_test.csv')

catTrain = train_df.loc[:, train_df.dtypes == 'object']
list(catTrain.columns)

train_full = pd.get_dummies(train_df, 
                          columns=list(catTrain.columns), 
                          drop_first=True)


y = train_full['SalePrice']
y_ad = train_full['SalePriceAd']
X = train_full.drop(['SalePrice', 'SalePriceAd'], 1)

# train_df.drop(['Unnamed: 0'], axis = 1, inplace = True)
# # test_df.drop(['Unnamed: 0'], axis = 1, inplace = True)

# ### change the dtypes of the 'GarageQuality' and 'GarageAge'
# train_df['GarageQuality'] = train_df['GarageQuality'].astype('int64')
# train_df['GarageAge'] = train_df['GarageAge'].astype('int64')
# # train_df['GarageQuality'] = train_df['GarageQuality'].astype('int64')
# # train_df['GarageAge'] = train_df['GarageAge'].astype('int64')


# numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
# # Check the skew of all numerical features
# skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
# skewness = pd.DataFrame({'Skew' :skewed_feats})
# #normalize columns greater than 0.8 skewness
# skewness = skewness[abs(skewness) > 0.80]

# for feat in skewed_feats:
# 	train_df[feat], fitted_lambda = stats.boxcox(train_df[feat])
# 	test_df[feat] = stats.boxcox(test_df[feat], fitted_lambda)

# #Cross-Validation function
# #You can use gridsearchCV to do this without calling this function
# n_folds = 5

# def rmsle_cv(model):
#     kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
#     rmse= np.sqrt(-cross_val_score(model, train.values, y_train, scoring="neg_mean_squared_error", cv = kf))
#     return(rmse)

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
# LINEAR MODEL
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, BayesianRidge, LassoLarsIC
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
#Grid Search
# lasso_params = {'alpha':[0.02, 0.024, 0.025, 0.026, 0.03]}
# ridge_params = {'alpha':[200, 230, 250,265, 270, 275, 290, 300, 500]}
# eNet_params = {"max_iter": [1, 5, 10, 100, 500],
#                       "alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
#                       "l1_ratio": np.arange(0.0, 1.0, 0.1)}

# lassoLars_params = {"criterion": ['bic', 'aic'],
# 					"max_iter": [1, 5, 10, 100, 200, 500]
# 				}
# KRR_params = {"alpha": [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
# 					"degree": [1, 2, 3, 5, 7, 9],
# 					"kernel": ['linear', 'polynomial'],
# 					"coef0": [1,1.5,2,2.5]
# 				}

#Base Regression Model:
# baseRegression = LinearRegression().fit(X, y)
# print(baseRegression)
#Grid Search
# from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
# lasso_Best = GridSearchCV(Lasso(), param_grid=lasso_params, scoring='r2', cv=10).fit(X, y)
# lasso_Best.best_estimator_

# ridge_Best = GridSearchCV(ridge(), param_grid=ridge_params, scoring='r2', cv=10).fit(X, y)
# ridge_Best.best_estimator_

# eNet_Best = GridSearchCV(ElasticNet(), parametersGrid, scoring='r2', cv=5).fit(X, y)
# eNet_Best.best_estimator_

# metric for evaluation
def rmse(y_true, y_pred):
    diff = y_pred - y_true
    sum_sq = sum(diff**2)    
    n = len(y_pred)   
    
    return np.sqrt(sum_sq/n)

def GridSearchModel(model, params, X, y, folds):
	# model_Best = GridSearchCV(model, param_grid=params, scoring='neg_mean_squared_error', cv=folds).fit(X, y)
	
	model_Best = RandomizedSearchCV(model, params, n_iter=500, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1).fit(X,y)

	model = model_Best.best_estimator_        
	best_idx = model_Best.best_index_

	grid_results = pd.DataFrame(model_Best.cv_results_) 
	cv_mean = abs(grid_results.loc[best_idx,'mean_test_score'])
	cv_std = grid_results.loc[best_idx,'std_test_score']

	y_pred = model.predict(X)
	cv_score = pd.Series({'mean':cv_mean,'std':cv_std})

	print(model)
	print(cv_mean)
	print(cv_std)

	# print stats on model performance         
	print('----------------------')
	print(model)
	print('----------------------')
	print('score=',model.score(X,y))
	print('rmse=',rmse(y, y_pred))
	print('cross_val: mean=',cv_mean,', std=',cv_std)

	

	return model, cv_score, grid_results

from xgboost.sklearn import XGBClassifier

opt_models = dict()
score_models = pd.DataFrame(columns=['mean','std'])
folds = 5

#
param_test1 = {
 'max_depth':range(2,10,1),
 'min_child_weight':range(1,10,1),
 'gamma': np.arange(0,0.5,0.1),
 'subsample': np.arange(0.5,1,0.1),
 'colsample_bytree': np.arange(0.5,1,0.1)
}

model_Best = RandomizedSearchCV(XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
 min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
 objective= 'binary:logistic', nthread=4, scale_pos_weight=1, seed=27), 
 param_test1, n_iter=100, scoring='neg_mean_squared_error',n_jobs=-1,verbose=1, cv=5)

model_Best.fit(X,y)

model = model_Best.best_estimator_        
best_idx = model_Best.best_index_

grid_results = pd.DataFrame(model_Best.cv_results_) 
cv_mean = abs(grid_results.loc[best_idx,'mean_test_score'])
cv_std = grid_results.loc[best_idx,'std_test_score']

y_pred = model.predict(X)
cv_score = pd.Series({'mean':cv_mean,'std':cv_std})
print(model)
print(cv_mean)
print(cv_std)

# print stats on model performance         
print('----------------------')
print(model)
print('----------------------')
print('score=',model.score(X,y))
print('rmse=',rmse(y, y_pred))
print('cross_val: mean=',cv_mean,', std=',cv_std)

#Linear Regression
#bayesianRidege
# param_grid = {"n_iter": np.arange(100,1000,10),
# 						"alpha_1": np.linspace(0.00001, 0.0001, num=10),
# 						"alpha_2": np.linspace(0.00001, 0.0001, num=10),
# 						"lambda_1": np.linspace(0.00001, 0.0001, num=10),
# 						"lambda_2": np.linspace(0.00001, 0.0001, num=10)
# }


# # Create regularization hyperparameter distribution using uniform distribution
# alpha_1 = uniform(loc=1e-06, scale=100*1e-06)
# alpha_2 = uniform(loc=1e-06, scale=100*1e-06)
# lambda_1 = uniform(loc=1e-06, scale=100*1e-06)
# lambda_2 = uniform(loc=1e-06, scale=100*1e-06)

# # Create hyperparameter options
# hyperparameters = dict(alpha_1=alpha_1,alpha_2=alpha_2,lambda_1=lambda_1, lambda_2=lambda_2)


# model = 'BayesianRidge'
# opt_models[model] = BayesianRidge()
# opt_models[model], cv_score, grid_results = GridSearchModel(opt_models[model], hyperparameters, X, y, folds)
# cv_score.name = model
# score_models = score_models.append(cv_score)

# BayesianRidge(alpha_1=8.9069716221632035e-05, alpha_2=1.5886463981587004e-05,
#               compute_score=False, copy_X=True, fit_intercept=True,
#               lambda_1=3.0623466647473011e-05, lambda_2=0.00010096401281732915,
#               n_iter=300, normalize=False, tol=0.001, verbose=False)
# ----------------------
# score= 0.919816497719
# rmse= 0.691426678543
# cross_val: mean= 0.615306961663 , std= 0.161369225449

# from scipy.stats import randint
# # #tree base model
# # param_grid = {'n_estimators': np.arange(100,500,1),
# #               'max_features': np.arange(5,100,5),
# #               'min_samples_split': np.arange(2,20,1)}
# # Create regularization hyperparameter distribution using uniform distribution
# n_estimators = randint(50, 500)
# max_features = randint(5, 100)
# min_samples_split = randint(2, 12)

# # Create hyperparameter options
# hyperparameters = dict(n_estimators=n_estimators,max_features=max_features,min_samples_split=min_samples_split)

# model = 'RandomForest'
# opt_models[model] = RandomForestRegressor()
# opt_models[model], cv_score, grid_results = GridSearchModel(opt_models[model], hyperparameters, X, y, folds)
# cv_score.name = model
# score_models = score_models.append(cv_score)
# from sklearn.isotonic import IsotonicRegression
# Isotonic = IsotonicRegression().fit(X, y)
# y_pred = Isotonic.predict(X)


# # print stats on model performance         
# print('----------------------')
# print(Isotonic)
# print('----------------------')
# print('score=',Isotonic.score(X,y))
# print('rmse=',rmse(y, y_pred))

# param_grid = {'n_estimators': np.arange(100,500,10),
#               'max_depth': np.arange(1,5,1),
#               'min_samples_split': np.arange(2,20,1)}

# model = 'GradientBoosting'
# opt_models[model] = GradientBoostingRegressor()
# opt_models[model], cv_score, grid_results = GridSearchModel(opt_models[model], param_grid, X, y, folds)
# cv_score.name = model
# score_models = score_models.append(cv_score)


# param_grid = {'n_estimators': np.arange(100,500,10),
#               'max_depth': np.arange(1,5,1)
#              }

# model = 'XGB'
# opt_models[model] = XGBRegressor()
# opt_models[model], cv_score, grid_results = GridSearchModel(opt_models[model], param_grid, X, y, folds)
# cv_score.name = model
# score_models = score_models.append(cv_score)


# lassoLarsIC_Best = GridSearchCV(LassoLarsIC(), lassoLars_params, scoring='r2', cv=5).fit(X, y)
# lassoLarsIC_Best.best_estimator_

# #Kernel ridge regression (KRR) combines ridge regression (linear least squares with l2-norm regularization) with the kernel trick. It thus learns a linear function in the space induced by the respective kernel and the data. For non-linear kernels, this corresponds to a non-linear function in the original space.
# KRR_Best = GridSearchCV(KernelRidge(), KRR_params, scoring='r2', cv=5).fit(X, y)
# KRR_Best.best_estimator_

# # The isotonic regression finds a non-decreasing approximation of a function while minimizing the mean squared error on the training data. The benefit of such a model is that it does not assume any form for the target function such as linearity.
# # https://scikit-learn.org/stable/auto_examples/plot_isotonic_regression.html

# # no parameter needed
# Isotonic = IsotonicRegression().fit(X, y)


# #SAVING model for feature use in PICKLE

# import pickle
# # Save the trained model as a pickle string. 
# baseRegression_model = pickle.dumps(baseRegression, "baseRegression.pkl") 
  
# # Load the pickled model 
# baseRegression_from_pickle = pickle.loads(baseRegression_model) 
  
# # Use the loaded pickled model to make predictions 
# baseRegression_from_pickle.predict(X_test) 

# lasso_Best_model = pickle.dumps(lasso_Best, "lasso_Best.pkl") 
  
# # Load the pickled model 
# lasso_Best_from_pickle = pickle.loads(lasso_Best_model) 
  
# # Use the loaded pickled model to make predictions 
# lasso_Best_from_pickle.predict(X_test) 


# # Tree Base Model
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor

