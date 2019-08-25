# load the necessary models 
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

##### loading training and test dataset
train_df = pd.read_csv("./data/train_dummified.csv")
test_df = pd.read_csv("./data/test_dummified.csv")

##### manipulate the price dataset as y
y_read = pd.read_csv("./data/price.csv")

##### take log at the 
y_train = np.log(y_read["SalePrice"])
X_train = train_df
X_test = test_df

############################################################ tuning ###########################################################################
# #### the parameters need to be tuned
# param_dist = {
#     'n_estimators': [50, 100, 200, 500],
#     'learning_rate' : [0.5,1],
#     'loss' : ['linear', 'square', 'exponential']
#  }

# #### use RandomizedSearchCV
# model_select = RandomizedSearchCV(AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth=4), n_estimators=24, learning_rate = 0.01, loss = 'linear'),
#  param_distributions = param_dist,
#  cv=5,
#  n_iter = 24,
#  n_jobs=-1)

# model_select.fit(X_train, y_train)

# print(model_select.best_params_)
############################################################ RUN #############################################################################
#### run the model with the best parameters
regressor = AdaBoostRegressor(base_estimator = DecisionTreeRegressor(max_depth=4), n_estimators=500, learning_rate = 1, loss = 'square')

# fit the model with X_train and y_train
regressor.fit(X_train, y_train)

# output the predicted results as dataframe 
results = pd.DataFrame(regressor.predict(X_test))

# exponentiate the results 
results = np.exp(results)

# save the results as csv
results.to_csv(r'./data/price_adaboost.csv', index = False)

