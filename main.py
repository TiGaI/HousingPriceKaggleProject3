import pandas as pd
import numpy as np
import re
from utils import *

if __name__ == '__main__':
	filename = 'https://raw.githubusercontent.com/TiGaI/HousingPriceKaggleProject3/master/data/test.csv'
	df = pd.read_csv(filename)
	df.drop('Id', axis = 1, inplace = True)
	interest_df = pd.read_csv('https://raw.githubusercontent.com/TiGaI/HousingPriceKaggleProject3/xiangwei/data/interest.csv')
	ameshpi = pd.read_csv('https://raw.githubusercontent.com/TiGaI/HousingPriceKaggleProject3/xiangwei/data/ameshpi.csv')
	sch_rank = pd.read_csv('https://raw.githubusercontent.com/TiGaI/HousingPriceKaggleProject3/xiangwei/data/SchoolRanking.csv')

	df = ImputeMissingValue(df)
	df = FeatureEngineering(df)
	df = AddInterest(interest_df, df)
	df = SchoolRank(sch_rank, df)
	df.drop(['Unnamed: 0'], axis = 1, inplace = True)

	if 'SalePrice' in list(df.columns):
		saleprice = AddHPI(ameshpi, df)
		#df.drop(['SalePrice'], axis=1, inplace = True)
		saleprice.to_csv(r'./data/price.csv', index = False)

	df.to_csv(r'./data/test.csv', index = False)
		
