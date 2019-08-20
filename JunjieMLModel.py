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

### MODEL IMPORTS
# LINEAR MODEL
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
# 
from sklearn.isotonic import IsotonicRegression

ir = IsotonicRegression()
y_ir = ir.fit_transform()

# Tree Base Model
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

