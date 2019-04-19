import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.ensemble import RandomForestRegressor # Pour le Random Forest
import torch

NUMBER_SELECTED_FEATURES = 80

def write_submission(Y_submit,index_submit,file_name='submission.csv'):
	y_submit=pd.DataFrame()
	y_submit['Id']= index_submit
	y_submit['SalePrice']= Y_submit
	y_submit.to_csv(file_name,index=False)

def load_datas() :
	train = pd.read_csv('data/train.csv')
	X_submit = pd.read_csv('data/test.csv')
	X = train.drop('SalePrice',axis=1)
	y = train['SalePrice']
	return X_submit,X,y

def delete_missing_features(X,X_submit) :
	col_dropped =list()
	for col in X.columns :
		if X[col].isnull().sum() > 0.5*X.shape[0] :col_dropped.append(col)
	X=X.drop(col_dropped,axis=1)
	X_submit=X_submit.drop(col_dropped,axis=1)
	print('Columns dropped because to much nans :',col_dropped)
	return(X,X_submit)

def replace_nans(X,X_submit) :
	numeric_feats = X.dtypes[X.dtypes != "object"].index
	X[numeric_feats] = X[numeric_feats].fillna(X[numeric_feats].median())
	X_submit[numeric_feats] = X_submit[numeric_feats].fillna(X[numeric_feats].median())
	object_feats = X.dtypes[X.dtypes == "object"].index
	for col in object_feats :
		X[col]=X[col].fillna(X[col].value_counts().idxmax())
		X_submit[col]=X_submit[col].fillna(X[col].value_counts().idxmax())
	return(X,X_submit)

def replace_categories(X,X_submit,approch='dummies'):
	m_train = X.shape[0]
	m_test = X_submit.shape[0]
	data_X = X.append(X_submit,ignore_index=True)
	if approch=='dummies' :
		data_X= pd.get_dummies(data_X)
	else :
		for col in data_X.columns :
			if data_X[col].dtypes == object :data_X[col] = data_X[col].astype("category").cat.codes
	X = data_X[0:m_train]
	X_submit = data_X[m_train:]
	X.set_index('Id',inplace=True)
	X_submit.set_index('Id',inplace=True)
	return X,X_submit

def set_new_features(X,X_submit) :
	#Total surface ( might be useful)
	X['TotalSF']=X['TotalBsmtSF']+X['1stFlrSF']+X['2ndFlrSF']
	X_submit['TotalSF']=X_submit['TotalBsmtSF']+X_submit['1stFlrSF']+X_submit['2ndFlrSF']
	#Relate the Two best
	#X['Combo']=X['TotalSF']*X['OverallQual']
	#X_submit['Combo']=X_submit['TotalSF']*X_submit['OverallQual']
	return(X,X_submit)


def feature_scaling(X,X_submit) :
	scaler = MinMaxScaler().fit(X)
	X[X.columns]=scaler.transform(X[X.columns])
	X_submit[X_submit.columns]=scaler.transform(X_submit[X_submit.columns])
	return X,X_submit

def select_features(X,y,X_submit) :
	rf = RandomForestRegressor(n_estimators=100) # On utilise le n_estimator max du processeur et max_features ="auto" par defaut
	rf.fit(X,y)
	ranking = np.argsort(-rf.feature_importances_) # Ranking nous donne les variables dans l'ordre d'importance
	selected_variables = X.columns.values[ranking[0:NUMBER_SELECTED_FEATURES]]
	X=X[selected_variables]
	X_submit=X_submit[selected_variables]
	return(X,X_submit)

def reshape(X_train, X_test, y_train, y_test, X_submit) :
	return np.array(X_train.T),\
		   np.array(X_test.T),\
		   np.array(y_train.values.reshape(y_train.shape[0], 1).T),\
		   np.array(y_test.values.reshape(y_test.shape[0], 1).T),\
		   np.array(X_submit.T)

def convert_to_torch(X_train, X_test, y_train, y_test, X_submit) :
	X_train = torch.from_numpy(X_train.to_numpy()).float()
	X_test = torch.from_numpy(X_test.to_numpy()).float()
	y_train = torch.from_numpy(y_train.to_numpy()).float()
	y_test = torch.from_numpy(y_test.to_numpy()).float()
	X_submit = torch.from_numpy(X_submit.to_numpy()).float()
	return X_train, X_test, y_train, y_test,X_submit

def load_preprocess_datas() :
	X_submit, X, y = load_datas()
	index_submit = X_submit['Id']
	X, X_submit = delete_missing_features(X, X_submit)
	X, X_submit = replace_nans(X, X_submit)
	X, X_submit = replace_categories(X, X_submit, approch='dummies')  # Dummies marche grave mieu
	X, X_submit = set_new_features(X, X_submit)
	X, X_submit = feature_scaling(X, X_submit)
	X, X_submit = select_features(X, y, X_submit)
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 1)
	X_train, X_test, y_train, y_test, X_submit = convert_to_torch(X_train, X_test, y_train, y_test, X_submit)
	return X_train, X_test, y_train, y_test,X_submit,index_submit

def describe_datasets(X_train, X_test, y_train, y_test, X_submit) :
	print('Number of examples : ',X_train.shape[1])
	print('          features : ', X_train.shape[0])
	print('X_train shape : ',X_train.shape)
	print('y_train shape : ',y_train.shape)
	print('X_test shape : ',X_test.shape)
	print('y_test shape : ', y_test.shape)
	print('X_submit shape : ',X_submit.shape)



if __name__ == "__main__" :
	X_train, X_test, y_train, y_test, X_submit,_ = load_preprocess_datas()
	describe_datasets(X_train, X_test, y_train, y_test, X_submit)
