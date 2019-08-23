import pandas as pd
import numpy as np
import re
from utils import *

if __name__ == '__main__':
	### remember to change test to train if train is needed 
	name = 'test'
	filename = 'https://raw.githubusercontent.com/TiGaI/HousingPriceKaggleProject3/master/data/' + name + '.csv'
	dataset = pd.read_csv(filename)

	### drop the ID column
	dataset.drop('Id', axis = 1, inplace = True)

	### the interest rate from 2006 to 2010
	interest = pd.read_csv('https://raw.githubusercontent.com/TiGaI/HousingPriceKaggleProject3/xiangwei/data/interest.csv')
	### housing price index in Ames
	ameshpi = pd.read_csv('https://raw.githubusercontent.com/TiGaI/HousingPriceKaggleProject3/xiangwei/data/ameshpi.csv')
	schoolrank = pd.read_csv('https://raw.githubusercontent.com/TiGaI/HousingPriceKaggleProject3/xiangwei/data/SchoolRanking.csv')

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
	
	# dummify the variables
	dataset_dummified = Dummify(dataset)

	# Check missing values
	CheckMissing(dataset)

	if 'SalePrice' in list(dataset.columns):
		saleprice = AddHPI(ameshpi, dataset)
		#df.drop(['SalePrice'], axis=1, inplace = True)
		saleprice.to_csv(r'./data/price.csv', index = False)

	dataset.to_csv(r'./data/'+ name + '.csv', index = False)
	dataset_dummified.to_csv(r'./data/'+ name + '_dummified.csv', index = False)
		
