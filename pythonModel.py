### import necessary packages
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import uniform

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, BayesianRidge, LassoLarsIC
from sklearn.svm import LinearSVR, SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

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

# metric for evaluation
def rmse(y_true, y_pred):
    diff = y_pred - y_true
    sum_sq = sum(diff**2)    
    n = len(y_pred)   
    
    return np.sqrt(sum_sq/n)

def GridSearchModel(model, params, X, y, folds):
	# model_Best = GridSearchCV(model, param_grid=params, scoring='neg_mean_squared_error', cv=folds).fit(X, y)
	
	model_Best = RandomizedSearchCV(model, params, n_iter=100, scoring='neg_mean_squared_error',n_jobs=-1,verbose=1, cv=3).fit(X,y)

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


###################
#
# RUN MODEL
######################


# Create hyperparameter options
# Create regularization hyperparameter distribution using uniform distribution
alpha_1 = uniform(loc=1e-06, scale=100*1e-06)
alpha_2 = uniform(loc=1e-06, scale=100*1e-06)
lambda_1 = uniform(loc=1e-06, scale=100*1e-06)
lambda_2 = uniform(loc=1e-06, scale=100*1e-06)

hyperparameters = dict(alpha_1=alpha_1,alpha_2=alpha_2,lambda_1=lambda_1, lambda_2=lambda_2)


model = 'BayesianRidge'
opt_models[model] = BayesianRidge()
opt_models[model], cv_score, grid_results = GridSearchModel(opt_models[model], hyperparameters, X, y, folds)
cv_score.name = model
score_models = score_models.append(cv_score)


# Create hyperparameter options
n_estimators = randint(50, 500)
max_features = randint(5, 100)
min_samples_split = randint(2, 12)
hyperparameters = dict(n_estimators=n_estimators,max_features=max_features,min_samples_split=min_samples_split)

model = 'RandomForest'
opt_models[model] = RandomForestRegressor()
opt_models[model], cv_score, grid_results = GridSearchModel(opt_models[model], hyperparameters, X, y, folds)
cv_score.name = model
score_models = score_models.append(cv_score)