import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
#from scipy import stats
#from scipy.special import boxcox1p
from utils import *

if __name__ == '__main__':
	### housing price index in Ames
	schoolrank = pd.read_csv('https://raw.githubusercontent.com/TiGaI/HousingPriceKaggleProject3/xiangwei/data/SchoolRanking.csv')
	ameshpi = pd.read_csv('https://raw.githubusercontent.com/TiGaI/HousingPriceKaggleProject3/xiangwei/data/ameshpi.csv')
	### the interest rate from 2006 to 2010
	interest = pd.read_csv('https://raw.githubusercontent.com/TiGaI/HousingPriceKaggleProject3/xiangwei/data/interest.csv')
	train = pd.read_csv('https://raw.githubusercontent.com/TiGaI/HousingPriceKaggleProject3/master/data/train.csv')
	test = pd.read_csv('https://raw.githubusercontent.com/TiGaI/HousingPriceKaggleProject3/master/data/test.csv')
	saleprice = AddHPI(ameshpi, train)

	train.drop(['SalePrice', 'Qrtr'], axis = 1, inplace = True)

	dataset = pd.concat([train,test], axis = 0)

	### drop the ID column
	dataset.drop('Id', axis = 1, inplace = True)

	# impute the missing values in the model
	dataset = ImputeMissingValue(dataset)

	# complete the feature engineering
	dataset = FeatureEngineering(dataset)

	# add the interest rate
	dataset = AddInterest(interest, dataset)

	# add the school rank to the dataset
	dataset = SchoolRank(schoolrank, dataset)
	
	# drop the unnecessary columns
	dataset.drop(['Unnamed: 0'], axis = 1, inplace = True)

	# standardize the numeric variables
	dataset = Standardize(dataset)

	# normalize the numeric variables
	#dataset = Normalize(dataset, 0.15)
	
	# dummify the variables
	dataset_dummified = Dummify(dataset)

	# Check missing values
	CheckMissing(dataset)

	train_df = dataset.iloc[0:1460]

	test_df = dataset.iloc[1460:2919]


	train_dummified = dataset_dummified.iloc[0:1460]

	test_dummified = dataset_dummified.iloc[1460:2919]

	saleprice.to_csv(r'./data/price.csv', index = False)
	train_df.to_csv(r'./data/train_df.csv', index = False)
	test_df.to_csv(r'./data/test_df.csv', index = False)
	train_dummified.to_csv(r'./data/train_dummified.csv', index = False)
	test_dummified.to_csv(r'./data/test_dummified.csv', index = False)
		
