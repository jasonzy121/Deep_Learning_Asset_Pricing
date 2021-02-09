import os
import time
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D
import statsmodels.api as sm
from scipy.stats import f
from sklearn.linear_model import LinearRegression

sns.set_style("white")

def deco_print(line, end='\n'):
	print('>==================> ' + line, end=end)

"""
from src.utils import load_dataframe
df = load_dataframe('datasets/F-F_Research_Data_5_Factors_2x3.CSV', skiprows=2, nrows=659)
df_train = df.loc['196701':'198612']
df_valid = df.loc['198701':'199112']
df_test = df.loc['199201':'201612']
rf_train = df_train.loc[:,'RF'].values / 100
rf_valid = df_valid.loc[:,'RF'].values / 100
rf_test = df_test.loc[:,'RF'].values / 100
"""
def load_dataframe(path, skiprows, nrows):
	df = pd.read_csv(path, skiprows=skiprows, nrows=nrows)
	df.rename(columns={'Unnamed: 0':'month'}, inplace=True)
	df.set_index('month', inplace=True)
	return df

def sort_by_task_id(folder_list):
	folder_id_list = [(folder, int(folder.split('_')[1])) for folder in folder_list]
	folder_id_list_sorted = sorted(folder_id_list, key=lambda t:t[1])
	return [folder for folder, _ in folder_id_list_sorted]

def Markowitz(r):
	Sigma = r.T.dot(r) / r.shape[0]
	mu = np.mean(r, axis=0)
	w = np.dot(np.linalg.pinv(Sigma), mu)
	return w

def sharpe(r):
	return np.mean(r / r.std())

def load_sorted_results(path, by=['sharpe_valid', 'sharpe_test']):
	store = pd.HDFStore(path)
	df = store['summary']
	store.close()
	return [df.sort_values(by=[col], ascending=False) for col in by]