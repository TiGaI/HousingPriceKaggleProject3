# load the necessary models 
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.model_selection import KFold, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

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
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [3, 6, 9, 12, 15, 18]
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

model_select = RandomizedSearchCV(estimator = RandomForestRegressor(random_state = 42), param_distributions = random_grid, 
	n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)

model_select.fit(X_train, y_train)

print(model_select.best_params_)

### with the dummified results
#{'n_estimators': 2000, 'min_samples_split': 10, 'min_samples_leaf': 1, 'max_features': 'sqrt', 'max_depth': 12, 'bootstrap': False}
##############################################################################################################################################


#################################################### model running ###########################################################################
#### run the model with the best parameters
regressor = RandomForestRegressor(n_estimators = 2000, min_samples_split = 10, min_samples_leaf = 1, max_features = 'sqrt', max_depth = 12,
	bootstrap = False, random_state = 42)

# fit the model with X_train and y_train
regressor.fit(X_train, y_train)

# output the predicted results as dataframe 
results = pd.DataFrame(regressor.predict(X_test))

# exponentiate the results 
results = np.exp(results)

# save the results as csv
results.to_csv(r'./data/price_rf.csv', index = False)

#############################################################################################################################################