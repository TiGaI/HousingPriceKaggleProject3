import pandas as pd
import numpy as np
import re

def ImputeMissingValue(df):
	## fill in the NAs of LotFrontage with 0
	df['LotFrontage'].fillna(0, inplace=True) 

	## fill in the NAs of MasVnrArea with 0
	df['MasVnrArea'].fillna(0, inplace=True)

	## fill in the NAs of MasVnrType with None
	df['MasVnrType'].fillna('None', inplace=True) 

	## fill in the NAs of PoolQC with None
	df['PoolQC'].fillna('None',inplace=True)  

	## fill in the NAs of 'Electrical' with 'SBrkr'
	df['Electrical'].fillna('SBrkr', inplace=True)

	## Fill the NAs of 'Fence' with 'None'
	df['Fence'].fillna('None', inplace=True)

	## Fill in the NAs of 'MscFeature' with 'None'
	df['MiscFeature'].fillna('None', inplace=True)

	# fill in the NAs of GarageYrBlt with YearBuilt
	df['GarageYrBlt'].fillna(df['YearBuilt'], inplace=True)

	# fill in the NAs of 'GarageType' with 'None'
	df['GarageType'].fillna("None", inplace=True)

	return df

def FeatureEngineering(df):
	## create a new variable indicating whether the house has a pool or not
	df['Pool'] = df['PoolArea'].apply(lambda x: 1 if x > 0 else 0)  

	## Hash 'PoolQC'
	PoolReplacement = {
		'PoolQC': {
			'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5
		}
	}

	# replace the PoolQC
	df.replace(PoolReplacement, inplace=True)

	## Hash Garage related 
	garabgeReplacement = {
		'GarageQual': {
			np.nan:0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5
		},
		'GarageCond': {
			np.nan:0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5
		},
		'GarageFinish': {
			np.nan:0, 'Unf': 1, 'RFn': 2, 'Fin': 3
	   }
	}

	# replace 'GarageQual', 'GarageCond', 'GarageFinish'
	df.replace(garabgeReplacement, inplace=True)

	df["GarageQuality"] = (df["GarageQual"] + df["GarageCond"])/2

	## Hash 'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure', 'BsmtCond', 'BsmtQual'
	BsmtReplacement = {
		'BsmtFinType1': {
			np.nan:0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6
		},
		'BsmtFinType2': {
			np.nan:0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6
		},
		'BsmtExposure': {
			np.nan:0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4
		},
		'BsmtCond': {
			np.nan:0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5
		},
		'BsmtQual': {
			np.nan:0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5
		},
	}

	# replace 'BsmtFinType1', 'BsmtFinType2', 'BsmtExposure', 'BsmtCond', 'BsmtQual'
	df.replace(BsmtReplacement, inplace = True)

	# convert the type of 'BsmtCond' to int64
	df["BsmtCond"] = df["BsmtCond"].astype(np.int64)

	## Hash FireplaceQu
	FirePlaceReplacement ={
		'FireplaceQu': {
			np.nan: 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5
		}
	}

	# replace 'FireplaceQu'
	df.replace(FirePlaceReplacement, inplace = True)

	### replace the conditions and qualities with numbers: 'ExterQual', 'ExterCond, 'HeatingQC', 'KitchenQual', 'CentralAir', 'LotShape'
	CondReplacement = {
		'ExterQual': {
			'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5
		},
		'ExterCond': {
			'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5
		},
		'HeatingQC': {
			'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5
		},
		'KitchenQual': {
			'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5
		},
		'CentralAir': {
			'N': 0, 'Y': 1
		},
		'LotShape': {
			'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3
		},
		'Functional': {
			'Typ': 5, 'Min1': 4, 'Min2': 4, 'Mod': 3, 'Maj1': 2, 'Maj2': 2, 'Sev': 1, 'Sal': 0
		},
		'LandSlope': {
			'Gtl': 0, 'Mod': 1, 'Sev': 2
		}
	}

	# replace those Variables
	df.replace(CondReplacement, inplace = True)

	### find the age of the house by subtracting YrSold and YearRemodAdd
	df['HouseAge'] = df['YrSold'] - df['YearRemodAdd']

	### find the age of the garage by subtracting YrSold and GarageYrBlt
	df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']   ### this has 0.6 correlation with HouseAge

	### create a binary remodel variable to denote whether it has been remodeled or not
	df['Remodel'] = (np.where(df['YearBuilt'] == df['YrSold'], 0, 1))

	### create a new variable TotolBath to sum up the num of bathrooms in the house
	df['TotalBath'] = df['FullBath'] + 0.5 * df['HalfBath'] + df['BsmtFullBath'] + 0.5 * df['BsmtHalfBath']

	df['WeightTotalBsmtSF'] = df.apply (lambda row: WeightedBasement(row), axis=1)

	### Take care of PorchSF 
	df['TotalPorchSF'] = df['OpenPorchSF'] + df['EnclosedPorch'] + df['3SsnPorch'] + df['ScreenPorch']

	## replace the MSZoning by Mean and median
	MSZoninglReplacement = {
	   'MSZoning': {
		  'C (all)': 0, 'RM': 1, 'RH': 2, 'RL': 3, 'FV': 4
	   }
	}

	# replace the MSZoning
	df.replace(MSZoninglReplacement, inplace=True)

	## Take care of Condition1 and Condition 2
	df['Condition2Renew'] = np.where(df['Condition1'] == df['Condition2'], 0, 1)

	df['PUD'] = df.apply (lambda row: PUDCol(row), axis=1)

	## collapse 'HouseStyle' into 'SFoyer', 'Floor', 'Finish', 'Slvl'
	tempstyle = df['HouseStyle'].str.split('(\d*\.\d+|\d+)([A-Za-z]+)', expand=True)
	tempstyle.rename(columns={0:'SFoyer',1:'Floor', 2:'Finish',3:'Slvl'}, inplace=True)
	tempstyle['Slvl'] = np.where(tempstyle['SFoyer']=="Slvl", 1, 0)
	tempstyle['SFoyer'] = np.where(tempstyle['SFoyer']=="SFoyer", 1, 0)
	tempstyle['Finish'] = np.where(tempstyle['Finish']=="Story", 1, tempstyle['Finish'])
	tempstyle['Finish'] = np.where(tempstyle['Finish']=="Fin", 1, tempstyle['Finish'])
	tempstyle['Finish'] = np.where(tempstyle['Finish']=="Unf", 0, tempstyle['Finish'])
	
	#tempstyle
	df_full = pd.concat([df, tempstyle], axis = 1)
	df_full['Floor'] = df_full.apply (lambda row: FloorCol(row), axis=1)
	df_full['Floor'] = df_full['Floor'].astype(np.float16)

	### fill in the blanks for Finish
	df_full['Finish'].fillna(1, inplace=True)

	### Variables to drop
	df_full.drop(['Condition2', 'Alley', 'YearBuilt', 'YearRemodAdd', 
		'Exterior2nd', 'HouseStyle', 'MSSubClass', 'YearBuilt', 
		'YearRemodAdd', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
		'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'GarageCond', 
		'GarageQual', 'GarageYrBlt'], axis=1, inplace = True)

	return df_full

def AddInterest(interest_df, df):
	
	return pd.merge(df, interest_df,  how='left', left_on=['YrSold','MoSold'], right_on = ['YrSold','MoSold'])

def AddHPI(ameshpi, df):
	ameshpi['Qrtr'] = ameshpi['MoSold'].apply(lambda i : (i+2)//3)

	df['Qrtr'] = df['MoSold'].apply(lambda i:(i+2)//3)
	del ameshpi['MoSold']

	df_full = pd.merge(df, ameshpi,  how='left', on=['YrSold','Qrtr'])

	df_full['SalePriceAd'] = (df_full['SalePrice']/(df_full['ATNHPIUS11180Q']/100))

	return df_full[['SalePrice', 'SalePriceAd']]


def SchoolRank(sch_rank, df):
	sch_rank1 = sch_rank.loc[((sch_rank['District'] == 'Ames Community School District') | (sch_rank['District']== 'Gilbert Community School District')) & (sch_rank['SchoolType'] == 'Elementary')].copy()

	sch_rank2 = sch_rank1[['District', 'School_Name', 'Rank']].copy()

	sch_rank2['Neighborhood'] = sch_rank2.apply (lambda row: schPlace(row), axis=1)

	SchReplace = {
	   'Rank': {
		  'Unable to Rate': 0, 'Acceptable': 1, 'Commendable': 2, 'High-Performing': 3
	   }
	}
	sch_rank2.replace(SchReplace, inplace=True)

	sch_rank3 = sch_rank2.groupby('Neighborhood').agg({'Rank':'mean'}).copy()

	df_full = pd.merge(df, sch_rank3, on='Neighborhood', how='left')
	df_full['Rank'].fillna(df_full['Rank'].mean(), inplace=True)

	return df_full


def WeightedBasement(x):
	'''
	this function will calculate the WeightedBasement area of the house and return such area

	'''
	if x['BsmtFinType1'] == 1:
		return x['TotalBsmtSF'] 
	else:
		if x['BsmtFinType2'] != 1:
			return (x['BsmtFinType1'] * x['BsmtFinSF1'] + x['BsmtFinType2'] * x['BsmtFinSF2'])
		else:
			return (x['BsmtFinType1'] * x['BsmtFinSF1'] + x['BsmtUnfSF'])

def PUDCol(x):
	'''
	this function will produce the indicator function to indicate whether the house is PUD or not
	'''
	if x['MSSubClass'] in [120,150,160,180]:
		return 1
	else:
		return 0


def FloorCol(df):
	'''
	this function will fill in the blanks for floor based on the value of 'MSSubClass'

	'''
	if df['HouseStyle'] == 'SFoyer':
		if df['MSSubClass'] == 85:
			return 1.5
		if df['MSSubClass'] == 90:
			return 1.5
		if df['MSSubClass'] == 180:
			return 2.5
		if df['MSSubClass'] == 120:
			return 1
	elif df['HouseStyle'] == 'SLvl':
		if df['MSSubClass'] == 80:
			return 1.5
		if df['MSSubClass'] == 180:
			return 2.5
		if df['MSSubClass'] == 190:
			return 2.5
		if df['MSSubClass'] == 20:
			return 1
		if df['MSSubClass'] == 90:
			return 2
		if df['MSSubClass'] == 60:
			return 2
	else:
		return df['Floor']


def Qrtr(x):
	'''
	this function will convert Month to Quarter
	'''
	for i in x['MoSold']:
		return (i+2)//3


def schPlace(x):
	'''
	this function will return the neighborhood name based on the school
	'''
	if x['School_Name'] == 'Edwards Elementary School':
		return 'Edwards'
	elif x['School_Name'] == 'Gilbert Elementary School':
		return 'Gilbert'
	elif x['School_Name'] == 'Sawyer Elementary School':
		return 'Sawyer'
	elif x['School_Name'] == 'Fellows Elementary School':
		return 'NAmes'
	elif x['School_Name'] == 'Meeker Elementary School':
		return 'NWAmes'
	elif x['School_Name'] == 'Gilbert Intermediate School':
		return 'Gilbert'
	elif x['School_Name'] == 'Mitchell Elementary School':
		return 'Mitchel'
	elif x['School_Name'] == 'Northwood Pre-School':
		return 'NWAmes'