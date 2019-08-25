# load the necessary models 
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from xgboost.sklearn import XGBRegressor

##### loading training and test dataset
train_df = pd.read_csv("./data/train_dummified.csv")
#train_df = pd.read_csv("./data/train_df.csv")

test_df = pd.read_csv("./data/test_dummified.csv")
#test_df = pd.read_csv("./data/test_df.csv")

##### manipulate the price dataset as y
y_read = pd.read_csv("./data/price.csv")

##### take log at the 
y_train = np.log(y_read["SalePrice"])
X_train = train_df
X_test = test_df


############################################################ tuning ###########################################################################
### the parameters need to be tuned
param_dist = {
    "colsample_bytree": [0.5, 0.6, 0.7, 0.8],
    "min_child_weight": [1.5, 6, 10],
    "max_depth": [3, 5, 7],
    "n_estimators": [2000],
    "subsample": [0.5, 0.7, 0.95],
}

model_select = GridSearchCV(
    estimator=xgb.XGBRegressor(
        learning_rate=0.1,
        n_estimators=33,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        nthread=4,
        scale_pos_weight=1,
        seed=27,
    ),
    param_grid=param_dist,
    scoring="neg_mean_squared_error",
    n_jobs=1,
    iid=False,
    cv=5,
)
model_select.fit(X, y)
print(model_select.best_params_)

# {'colsample_bytree': 0.6, 'max_depth': 3, 'min_child_weight': 1.5, 'n_estimators': 2000, 'subsample': 0.7}

##############################################################################################################################################


#################################################### model running ###########################################################################
#### run the model with the best parameters
regressor = XGBRegressor(base_score=0.5, booster='gbtree',
                                    colsample_bylevel=1, colsample_bynode=1,
                                    colsample_bytree=0.6, gamma=0,
                                    importance_type='gain', learning_rate=0.1,
                                    max_delta_step=0, max_depth=3,
                                    min_child_weight=1.5, missing=None,
                                    n_estimators=2000, n_jobs=1, nthread=4,
                                    objective='reg:squarederror',
                                    predictor='gpu_predictor', random_state=0,
                                    reg_alpha=0, reg_lambda=1,
                                    scale_pos_weight=1, seed=27, silent=None,
                                    subsample=0.7, tree_method='gpu_hist',
                                    verbosity=1)

# fit the model with X_train and y_train
regressor.fit(X_train, y_train)

# output the predicted results as dataframe 
results = pd.DataFrame(regressor.predict(X_test))

# exponentiate the results 
results = np.exp(results)

# save the results as csv
results.to_csv(r'./data/price_xgboost.csv', index = False)

#
