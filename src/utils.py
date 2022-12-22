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

def max_1_month_loss(sdf):
	return sdf.min()

def max_drawdown(sdf):
	max_drawdown = 0
	current_drawdown = 0
	for i in range(len(sdf)):
		r = sdf[i]
		if r > 0 and current_drawdown > 0:
			max_drawdown = max(max_drawdown, current_drawdown)
			current_drawdown = 0
		elif r < 0:
			current_drawdown += 1
	return max_drawdown

def calculateTurnover(R, w, Rf):
	# append risk free return
	w = np.concatenate([w, 1 - w.sum(axis=1, keepdims=True)], axis=1)
	R = R + Rf[:,np.newaxis]
	R = np.concatenate([R, Rf[:,np.newaxis]], axis=1)
	#
	w_plus = np.maximum(w, 0) # + portfolio weight at time t
	w_minus = - np.minimum(w, 0) # - portfolio weight at time t
	R_SDF_plus = np.sum(R*w_plus, axis=1, keepdims=True) # + portfolio excess return at time t+1
	R_SDF_minus = np.sum(R*w_minus, axis=1, keepdims=True) # - portfolio excess return at time t+1
	FT_plus = (1+R_SDF_plus)[:-1]*w_plus[1:] - ((1+R)*w_plus)[:-1]
	FT_minus = (1+R_SDF_minus)[:-1]*w_minus[1:] - ((1+R)*w_minus)[:-1]
	T_plus = np.sum(np.abs(FT_plus), axis=1) / np.sum(w_plus[:-1], axis=1)
	T_minus = np.sum(np.abs(FT_minus), axis=1) / np.sum(w_minus[:-1], axis=1)
	return T_plus, T_minus, T_plus.mean(), T_minus.mean()

def calculateTurnover_with_dl(dl, w, Rf):
	for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
		w_reshape = np.zeros_like(mask, dtype=float)
		w_reshape[mask] = w
		return calculateTurnover(R, w_reshape, Rf)

def load_sorted_results(path, by=['sharpe_valid', 'sharpe_test']):
	store = pd.HDFStore(path)
	df = store['summary']
	store.close()
	return [df.sort_values(by=[col], ascending=False) for col in by]

def create_characteristic_projected_portfolios(dl):
	for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
		R_reshape = R[mask]
		I_reshape = I[mask] - 0.5
		splits = np.sum(mask, axis=1).cumsum()[:-1]
		R_list = np.split(R_reshape, splits)
		I_list = np.split(I_reshape, splits)
	portfolio_list = []
	for R_t, I_t in zip(R_list, I_list):
		portfolio_list.append(R_t.dot(I_t))
	return np.array(portfolio_list)

def eval_RtnFcst(model, sess, dl, value=None, quantile=0.1):
	R_pred = model.getPrediction(sess, dl)
	for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
		portfolio = construct_long_short_portfolio(R_pred, R, mask, value=value, low=quantile, high=quantile)
	return portfolio, sharpe(portfolio)

def eval_GAN_long_short(model, sess, dl, initial_state=None, value=None, quantile=0.1, sort_by_value=False):
	w = model.getWeightWithData(sess, dl, initial_state=initial_state, normalized=False)
	for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
		if (value is not None) and sort_by_value:
			tmp = np.zeros_like(value)
			tmp[mask] = -w
			portfolio = construct_long_short_portfolio(value[mask], R, mask, value=tmp, low=0, high=2*quantile, normalize=False)
		else:
			portfolio = construct_long_short_portfolio(-w, R, mask, value=value, low=quantile, high=quantile)
	return portfolio, sharpe(portfolio)

def eval_SDF_model(model, sess, dl, initial_state=None):
	SDF = model.getSDF(sess, dl, initial_state)[:,0]
	portfolio = 1 - SDF
	return SDF, portfolio, sharpe(portfolio)

def OLS(X, y, alpha=False):
	ols = sm.OLS(y, np.concatenate([X,np.ones((X.shape[0],1))], axis=1))
	ols_result = ols.fit()
	if alpha:
		return (ols_result.params[-1], ols_result.tvalues[-1], ols_result.pvalues[-1])
	else:
		print(ols_result.summary())

def GRS(X, y):
	T, N = y.shape # T * N
	Q = X.shape[1]
	X_with_intercept = np.concatenate([np.ones((T,1)), X], axis=1) # T * Q+1
	B_hat = np.dot(y.T.dot(X_with_intercept), np.linalg.inv(X_with_intercept.T.dot(X_with_intercept))) # N * Q+1
	alpha_hat = B_hat[:,0]
	y_hat = X_with_intercept.dot(B_hat.T) # T * N
	Sigma_hat = (y.T.dot(y) - y_hat.T.dot(y)) / T
	k = T - np.dot(X.T.sum(axis=1), np.dot(np.linalg.inv(X.T.dot(X)), X.T.sum(axis=1)))
	Lambda = 1.0 * (T-N-Q)/T/N*k*np.dot(alpha_hat, np.dot(np.linalg.pinv(Sigma_hat), alpha_hat))
	return (Lambda, 1-f.cdf(Lambda, N, T-N-Q))

def construct_decile_portfolios(w, R, mask, value=None, decile=10):
	R = R[mask]
	N_i = np.sum(mask.astype(int), axis=1)
	N_i_cumsum = np.cumsum(N_i)
	w_split = np.split(w, N_i_cumsum)[:-1]
	R_split = np.split(R, N_i_cumsum)[:-1]

	# value weighted
	value_weighted = False
	if value is not None:
		value_weighted = True
		value = value[mask]
		value_split = np.split(value, N_i_cumsum)[:-1]

	portfolio_returns = []

	for j in range(mask.shape[0]):
		R_j = R_split[j]
		w_j = w_split[j]
		if value_weighted:
			value_j = value_split[j]
			R_w_j = [(R_j[k], w_j[k], value_j[k]) for k in range(N_i[j])]
		else:
			R_w_j = [(R_j[k], w_j[k], 1) for k in range(N_i[j])]
		R_w_j_sorted = sorted(R_w_j, key=lambda t:t[1])
		n_decile = N_i[j] // decile
		R_decile = []
		for i in range(decile):
			R_decile_i = 0.0
			value_sum_i = 0.0
			for k in range(n_decile):
				R_decile_i += R_w_j_sorted[i*n_decile+k][0] * R_w_j_sorted[i*n_decile+k][2]
				value_sum_i += R_w_j_sorted[i*n_decile+k][2]
			R_decile.append(R_decile_i / value_sum_i)
		portfolio_returns.append(R_decile)
	return np.array(portfolio_returns)

def construct_long_short_portfolio(w, R, mask, value=None, low=0.1, high=0.1, normalize=True):
	# use masked R and value
	N_i = np.sum(mask.astype(int), axis=1)
	N_i_cumsum = np.cumsum(N_i)
	w_split = np.split(w, N_i_cumsum)[:-1]
	R_split = np.split(R, N_i_cumsum)[:-1]

	# value weighted
	value_weighted = False
	if value is not None:
		value_weighted = True
		value_split = np.split(value, N_i_cumsum)[:-1]

	portfolio_returns = []

	for j in range(mask.shape[0]):
		R_j = R_split[j]
		w_j = w_split[j]
		if value_weighted:
			value_j = value_split[j]
			R_w_j = [(R_j[k], w_j[k], value_j[k]) for k in range(N_i[j])]
		else:
			R_w_j = [(R_j[k], w_j[k], 1) for k in range(N_i[j])]
		R_w_j_sorted = sorted(R_w_j, key=lambda t:t[1])
		n_low = int(low * N_i[j])
		n_high = int(high * N_i[j])

		if n_high == 0.0:
			portfolio_return_high = 0.0
		else:
			portfolio_return_high = 0.0
			value_sum_high = 0.0
			for k in range(n_high):
				portfolio_return_high += R_w_j_sorted[-k-1][0] * R_w_j_sorted[-k-1][2]
				value_sum_high += R_w_j_sorted[-k-1][2]
			if normalize:
				portfolio_return_high /= value_sum_high

		if n_low == 0:
			portfolio_return_low = 0.0
		else:
			portfolio_return_low = 0.0
			value_sum_low = 0.0
			for k in range(n_low):
				portfolio_return_low += R_w_j_sorted[k][0] * R_w_j_sorted[k][2]
				value_sum_low += R_w_j_sorted[k][2]
			if normalize:
				portfolio_return_low /= value_sum_low

		portfolio_returns.append(portfolio_return_high - portfolio_return_low)
	return np.array(portfolio_returns)

def plot_SDF(df, columns=None, n_train=240, n_valid=60, plotPath=None, dateEnd='20161201', figsize=(8,6)):
	date = pd.date_range(end=dateEnd, periods=df.shape[0], freq='MS')
	df.loc[:,'date'] = date
	df.set_index('date', inplace=True)
	fig = plt.figure(figsize=figsize)
	if columns is None:
		columns = df.columns
	for var in columns:
		s = df.loc[:,var]
		s_cumsum = pd.concat([s.iloc[:n_train].cumsum(), s.iloc[n_train:(n_train+n_valid)].cumsum(), s.iloc[(n_train+n_valid):].cumsum()])
		plt.scatter(s_cumsum.index, s_cumsum, s=10, label=var)
	plt.ylabel('Cumulative Excess Return')
	plt.legend()
	if plotPath:
		plt.savefig(os.path.join(plotPath, 'sdf.pdf'))
		plt.savefig(os.path.join(plotPath, 'sdf.png'))
	plt.show()

# def plot_SMV(SMV, n_train=240, n_valid=60, plotPath=None, dateEnd='20161201', figsize=(8,6)):
# 	date = pd.date_range(end=dateEnd, periods=SMV.shape[0], freq='MS')
# 	df = pd.DataFrame(SMV, columns=['Macro_%d' %k for k in range(SMV.shape[1])])
# 	df.loc[:,'date'] = date
# 	df.set_index('date', inplace=True)
# 	for var in df.columns:
# 		s = df.loc[:,var]
# 		fig = plt.figure(figsize=figsize)
# 		plt.scatter(s.index, s, s=10, label='Train')
# 		plt.scatter(s.index[n_train:], s[n_train:], s=10, label='Valid')
# 		plt.scatter(s.index[(n_train+n_valid):], s[(n_train+n_valid):], s=10, label='Test')
# 		plt.legend()
# 		plt.ylabel(var)
# 		if plotPath:
# 			plt.savefig(os.path.join(plotPath, '%s.pdf' %var))
# 			plt.savefig(os.path.join(plotPath, '%s.png' %var))
# 		plt.show()

def plot_SMV(SMV, plotPath=None, dateEnd='20161201', figsize=(9,12)):
	df = pd.read_csv('datasets/USRECM.csv')
	y = df.loc[:,'USRECM'].values

	date = pd.date_range(end=dateEnd, periods=SMV.shape[0], freq='MS')
	df_idx = pd.DataFrame(y[:,np.newaxis], index=date)

	df = pd.DataFrame(SMV, columns=['Macro_%d' %k for k in range(SMV.shape[1])])
	df.loc[:,'date'] = date
	df.set_index('date', inplace=True)

	fig = plt.figure(figsize=figsize)

	ax1 = plt.subplot(411)
	ax1.scatter(date, df.loc[:,'Macro_0'], s=10)
	ax1.set_ylabel('Macro_0')
	ax1.fill_between(date, -1, 1, where=y > 0.5, facecolor='gray', alpha=0.5)
	ax1.set_xticks([])

	ax2 = plt.subplot(412)
	ax2.scatter(date, df.loc[:,'Macro_1'], s=10)
	ax2.set_ylabel('Macro_1')
	ax2.fill_between(date, -1, 1, where=y > 0.5, facecolor='gray', alpha=0.5)
	ax2.set_xticks([])

	ax3 = plt.subplot(413)
	ax3.scatter(date, df.loc[:,'Macro_2'], s=10)
	ax3.set_ylabel('Macro_2')
	ax3.fill_between(date, -1, 1, where=y > 0.5, facecolor='gray', alpha=0.5)
	ax3.set_xticks([])

	ax4 = plt.subplot(414)
	ax4.scatter(date, df.loc[:,'Macro_3'], s=10)
	ax4.fill_between(date, -1, 1, where=y > 0.5, facecolor='gray', alpha=0.5)
	ax4.set_ylabel('Macro_3')

	if plotPath:
		plt.savefig(os.path.join(plotPath, 'SMV.pdf'))
		plt.savefig(os.path.join(plotPath, 'SMV.png'))
	plt.show()

def plot_variable_importance(var, imp, labelColor, plotPath=None, normalize=True, top=30, figsize=(8,6), color2category=None):
	if normalize:
		if min(imp) >= 0:
			imp = imp / sum(imp)
		else:
			deco_print('WARNING: Unable to normalize due to negative importance value! ')
	var_imp = list(zip(var, imp, labelColor))
	var_imp_sort = sorted(var_imp, key=lambda t:-t[1])
	var_top = [var_imp_sort[i][0] for i in range(top)]
	imp_top = [var_imp_sort[i][1] for i in range(top)]
	color_top = [var_imp_sort[i][2] for i in range(top)]
	y_pos = np.arange(top)
	if not color2category:
		fig, ax = plt.subplots(figsize=figsize)
		ax.barh(y_pos, imp_top, align='center', color='blue')
	else:
		fig, (ax, ax_cb) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios':[19,1]})
		ax.barh(y_pos, imp_top, align='center', color=color_top)
	ax.set_yticks(y_pos)
	ax.set_yticklabels(var_top, fontweight='bold')
	ticklabels = ax.yaxis.get_ticklabels()
	for i in range(len(ticklabels)):
		ticklabels[i].set_color(color_top[i])
	ax.invert_yaxis()
	if color2category:
		color_list = list(color2category.keys())
		nColors = len(color_list)
		ticks = [color2category[color] for color in color_list]
		cmap = colors.ListedColormap(color_list)
		cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap)
		cb.set_ticks(np.linspace(0,1,nColors+1)[:-1] + 1./nColors/2)
		cb.set_ticklabels(ticks)
		cb.set_label('Category')
	if plotPath:
		plt.savefig(os.path.join(plotPath, 'VI_top_%d.pdf' %top), bbox_inches='tight')
		plt.savefig(os.path.join(plotPath, 'VI_top_%d.png' %top), bbox_inches='tight')
	plt.show()

def plotWeight1D(x, v, xlabel, 
	ylabel=None, idx_y=None, zlabel=None, idx_z=None, quantiles=[0.1, 0.25, 0.5, 0.75, 0.9], 
	plotPath=None, idx=None, figsize=(8,6), label='weight'):
	fig = plt.figure(figsize=figsize)
	plt.tick_params(labelsize=18)
	if ylabel is None:
		plt.scatter(x, v)
	else:
		if zlabel is None:
			for i in range(len(quantiles)):
				plt.scatter(x, v[i], label='%s %d%%' %(ylabel, int(quantiles[i] * 100)))
			plt.legend()
		else:
			for i in range(len(quantiles)):
				for j in range(len(quantiles)):
					plt.scatter(x, v[i,j], label='%s %d%%,%s %d%%' %(ylabel, int(quantiles[i]*100), zlabel, int(quantiles[j]*100)))
			plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=3, frameon=True)
			# plt.legend(loc='upper center', bbox_to_anchor=(1.2, 0.8), ncol=1)
	dy = 0.1 * (v.max() - v.min())
	plt.ylim(v.min()-dy, v.max()+dy)
	# v_mid = (v.max() + v.min()) / 2
	# plt.ylim(v_mid-0.04, v_mid+0.04) # GAN w
	# plt.ylim(v_mid-0.06, v_mid+0.06) # GAN w quantile
	# plt.ylim(v_mid-0.015, v_mid+0.015) # FFN w
	# plt.ylim(v_mid-0.06, v_mid+0.06) # EN w
	# plt.ylim(v_mid-0.3, v_mid+0.3) # LS w
	plt.xlabel(xlabel, fontsize=18)
	plt.ylabel(label, fontsize=18)
	if plotPath:
		if idx_y is None:
			plt.savefig(os.path.join(plotPath, 'w_x_%d.pdf' %idx), bbox_inches='tight')
			plt.savefig(os.path.join(plotPath, 'w_x_%d.png' %idx), bbox_inches='tight')
		else:
			if idx_z is None:
				plt.savefig(os.path.join(plotPath, 'w_x_%d_y_%d_quantile.pdf' %(idx, idx_y)), bbox_inches='tight')
				plt.savefig(os.path.join(plotPath, 'w_x_%d_y_%d_quantile.png' %(idx, idx_y)), bbox_inches='tight')
			else:
				plt.savefig(os.path.join(plotPath, 'w_x_%d_y_%d_z_%d_quantile.pdf' %(idx, idx_y, idx_z)), bbox_inches='tight')
				plt.savefig(os.path.join(plotPath, 'w_x_%d_y_%d_z_%d_quantile.png' %(idx, idx_y, idx_z)), bbox_inches='tight')
	plt.show()

def plotWeight2D(x, y, v, xlabel, ylabel, plotPath=None, idx_x=None, idx_y=None, figsize=(8,6), label='weight'):
	fig = plt.figure(figsize=figsize)
	levels = np.linspace(np.min(v),np.max(v),51)
	im = plt.contourf(x, y, v, levels=levels, cmap=cm.magma_r)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	cbar = plt.colorbar(im)
	cbar.ax.set_ylabel(label)
	if plotPath:
		plt.savefig(os.path.join(plotPath, 'w_x_%d_y_%d.pdf' %(idx_x, idx_y)))
		plt.savefig(os.path.join(plotPath, 'w_x_%d_y_%d.png' %(idx_x, idx_y)))
	plt.show()

def plotWeight3D(x, y, v, zs, xlabel, ylabel, zlabel, plotPath=None, idx_x=None, idx_y=None, idx_z=None, figsize=(8,6), label='weight'):
	fig = plt.figure(figsize=figsize)
	ax = fig.gca(projection='3d')
	levels = np.linspace(np.min(v),np.max(v),51)
	for k in range(len(zs)):
		z = zs[k]
		im = ax.contourf(x, y, v[:,:,k], offset=z, levels=levels, cmap=cm.magma_r)

	ax.set_xlabel(xlabel)
	ax.set_ylabel(ylabel)
	ax.set_zlabel(zlabel)
	ax.set_zlim(zs[0], zs[-1])
	cbar = plt.colorbar(im)
	cbar.ax.set_ylabel(label)
	if plotPath:
		plt.savefig(os.path.join(plotPath, 'w_x_%d_y_%d_z_%d.pdf' %(idx_x, idx_y, idx_z)))
		plt.savefig(os.path.join(plotPath, 'w_x_%d_y_%d_z_%d.png' %(idx_x, idx_y, idx_z)))
	plt.show()

def plot_decile_portfolios(df, n_train=240, n_valid=60, dateEnd='20161201', plotPath=None, figsize=(8,6)):
	date = pd.date_range(end=dateEnd, periods=df.shape[0], freq='MS')
	df.loc[:,'date'] = date
	df.set_index('date', inplace=True)
	fig = plt.figure(figsize=figsize)
	for decile in df.columns:
		s = df.loc[:,decile]
		# s_cumsum = s.cumsum()
		# plt.scatter(s_cumsum.index, s_cumsum / s.std(), s=10, label=decile)
		s = pd.concat([s.iloc[:n_train].cumsum() / s.iloc[:n_train].std(),
				s.iloc[n_train:(n_train+n_valid)].cumsum() / s.iloc[n_train:(n_train+n_valid)].std(),
				s.iloc[(n_train+n_valid):].cumsum() / s.iloc[(n_train+n_valid):].std()])
		plt.scatter(s.index, s, s=10, label=decile)
	# plt.axvline(x=df.index[n_train], color='gray', linestyle='--')
	# plt.axvline(x=df.index[n_train+n_valid], color='gray', linestyle='--')
	plt.ylim(-75,125)
	plt.ylabel('Cumulative Excess Return')
	plt.legend()
	if plotPath:
		plt.savefig(os.path.join(plotPath, 'decile_portfolios.pdf'))
		plt.savefig(os.path.join(plotPath, 'decile_portfolios.png'))
	plt.show()

def decomposeReturn(R, beta):
	R_hat = beta.dot(R) / beta.dot(beta) * beta
	residual = R - R_hat
	return R_hat, residual

def UnexplainedVariation(R, residual):
	return np.mean(np.square(residual)) / np.mean(np.square(R))

def FamaMcBethAlpha(residual):
	return np.sqrt(np.mean(np.square(residual.mean(axis=0))))

def R2(R, residual, cross_sectional=False):
	if cross_sectional:
		return 1 - np.mean(np.square(residual.mean(axis=0))) / np.mean(np.square(R.mean(axis=0)))
	else:
		return 1 - np.mean(np.square(residual)) / np.mean(np.square(R))

def plotReturnDecile(dl, beta, charName='All', decile=10, plotPath=None, figsize=(8,6)):
	if charName != 'All':
		fig, ax = plt.subplots(figsize=figsize)
		var2color = {charName:'blue'}
		charList = [charName]
	else:
		fig, (ax, ax_cb) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={'width_ratios':[19,1]})
		var2color, color2category = dl.getIndividualFeatureColarLabelMap()
		charList = dl.getIndividualFeatureList()

		color_list = list(color2category.keys())
		nColors = len(color_list)
		ticks = [color2category[color] for color in color_list]
		cmap = colors.ListedColormap(color_list)
		cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap)
		cb.set_ticks(np.linspace(0,1,nColors+1)[:-1] + 1./nColors/2)
		cb.set_ticklabels(ticks)
		cb.set_label('Category')

	time_start = time.time()
	cnt = 0
	for var in charList:
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			char = I[mask, dl.getIdxByIndividualFeature(var)]
			R = R[mask]
		splits = np.sum(mask, axis=1).cumsum()[:-1]
		R_split = np.split(R, splits)
		beta_split = np.split(beta, splits)
		char_split = np.split(char, splits)
		R_decile = []
		R_hat_decile = []
		for R_t, beta_t, char_t in zip(R_split, beta_split, char_split):
			tmp_t = sorted(list(zip(R_t, beta_t, char_t)), key=lambda t:t[2])
			tmp_t_decile = np.split(tmp_t, np.arange(1,decile) * (len(tmp_t) // decile))
			R_t_decile = []
			beta_t_decile = []
			for i in range(decile):
				R_t_decile_list, beta_t_decile_list, _ = list(zip(*tmp_t_decile[i]))
				R_t_decile.append(np.mean(R_t_decile_list))
				beta_t_decile.append(np.mean(beta_t_decile_list))
			R_t_decile = np.array(R_t_decile)
			beta_t_decile = np.array(beta_t_decile)
			R_t_decile_hat, _ = decomposeReturn(R_t_decile, beta_t_decile)
			R_decile.append(R_t_decile)
			R_hat_decile.append(R_t_decile_hat)
		R_decile = np.array(R_decile)
		R_hat_decile = np.array(R_hat_decile)
		ax.scatter(R_decile.mean(axis=0), R_hat_decile.mean(axis=0), s=15, color=var2color[var])
		cnt += 1
		time_last = time.time() - time_start
		time_est = time_last / cnt * len(charList)
		deco_print('Plotting Variable: %s\tElapse / Estimate: %.2fs / %.2fs' %(var, time_last, time_est))
	xlim = np.linspace(*ax.get_xlim())
	ax.plot(xlim, xlim, color='black')
	ax.set_xlabel('Excess Return')
	ax.set_ylabel('Projected Excess Return')
	if plotPath:
		plt.savefig(os.path.join(plotPath, 'projected_return.pdf'), bbox_inches='tight')
		plt.savefig(os.path.join(plotPath, 'projected_return.png'), bbox_inches='tight')
	plt.show()

def calculateStatisticsDecile(dl, beta, charName, decile=10):
	for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
		char = I[mask, dl.getIdxByIndividualFeature(charName)]
		R = R[mask]

	splits = np.sum(mask, axis=1).cumsum()[:-1]
	R_split = np.split(R, splits)
	beta_split = np.split(beta, splits)
	char_split = np.split(char, splits)
	R_decile = []
	residual_decile = []
	beta_decile = []

	for R_t, beta_t, char_t in zip(R_split, beta_split, char_split):
		tmp_t = sorted(list(zip(R_t, beta_t, char_t)), key=lambda t:t[2])
		tmp_t_decile = np.split(tmp_t, np.arange(1,decile) * (len(tmp_t) // decile))
		R_t_decile = []
		beta_t_decile = []

		for i in range(decile):
			R_t_decile_list, beta_t_decile_list, _ = list(zip(*tmp_t_decile[i]))
			R_t_decile.append(np.mean(R_t_decile_list))
			beta_t_decile.append(np.mean(beta_t_decile_list))

		R_t_decile = np.array(R_t_decile)
		beta_t_decile = np.array(beta_t_decile)
		beta_decile.append(beta_t_decile)
		R_t_decile_hat, residual_t_decile = decomposeReturn(R_t_decile, beta_t_decile)
		R_decile.append(R_t_decile)
		residual_decile.append(residual_t_decile)

	beta_decile = np.array(beta_decile)
	R_decile = np.array(R_decile)
	residual_decile = np.array(residual_decile)
	UV_decile = np.array([UnexplainedVariation(R_decile[:,[i]], residual_decile[:,[i]]) for i in range(decile)])
	Alpha_decile = np.array([FamaMcBethAlpha(residual_decile[:,[i]]) for i in range(decile)])
	R2_CS_decile = np.array([R2(R_decile[:,[i]], residual_decile[:,[i]], cross_sectional=True) for i in range(decile)])
	UV = UnexplainedVariation(R_decile, residual_decile)
	Alpha = FamaMcBethAlpha(residual_decile)
	R2_CS = R2(R_decile, residual_decile, cross_sectional=True)
	return UV, Alpha, R2_CS, UV_decile, Alpha_decile, R2_CS_decile

def calculateStatisticsDoubleSorted5(dl, beta, charName1, charName2):
	for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
		char1 = I[mask, dl.getIdxByIndividualFeature(charName1)]
		char2 = I[mask, dl.getIdxByIndividualFeature(charName2)]
		R = R[mask]

	splits = np.sum(mask, axis=1).cumsum()[:-1]
	R_split = np.split(R, splits)
	beta_split = np.split(beta, splits)
	char1_split = np.split(char1, splits)
	char2_split = np.split(char2, splits)
	R_bucket = []
	residual_bucket = []
	beta_bucket = []

	for R_t, beta_t, char1_t, char2_t in zip(R_split, beta_split, char1_split, char2_split):
		R_t_bucket = np.zeros(25)
		beta_t_bucket = np.zeros(25)
		cnt_t_bucket = np.zeros(25, dtype=int)

		for R_i, beta_i, char1_i, char2_i in list(zip(R_t, beta_t, char1_t, char2_t)):
			idx_char1 = int((char1_i - 1e-8) * 5)
			idx_char2 = int((char2_i - 1e-8) * 5)
			idx = idx_char1 * 5 + idx_char2
			R_t_bucket[idx] += R_i
			beta_t_bucket[idx] += beta_i
			cnt_t_bucket[idx] += 1

		assert(cnt_t_bucket.sum() == len(R_t))
		R_t_bucket = R_t_bucket / cnt_t_bucket
		beta_t_bucket = beta_t_bucket / cnt_t_bucket
		R_t_bucket_hat, residual_t_bucket = decomposeReturn(R_t_bucket, beta_t_bucket)
		beta_bucket.append(beta_t_bucket)
		R_bucket.append(R_t_bucket)
		residual_bucket.append(residual_t_bucket)

	beta_bucket = np.array(beta_bucket)
	R_bucket = np.array(R_bucket)
	residual_bucket = np.array(residual_bucket)
	UV_bucket = np.array([UnexplainedVariation(R_bucket[:,[i]], residual_bucket[:,[i]]) for i in range(25)])
	Alpha_bucket = np.array([FamaMcBethAlpha(residual_bucket[:,[i]]) for i in range(25)])
	R2_CS_bucket = np.array([R2(R_bucket[:,[i]], residual_bucket[:,[i]], cross_sectional=True) for i in range(25)])
	UV = UnexplainedVariation(R_bucket, residual_bucket)
	Alpha = FamaMcBethAlpha(residual_bucket)
	R2_CS = R2(R_bucket, residual_bucket, cross_sectional=True)
	return UV, Alpha, R2_CS, UV_bucket, Alpha_bucket, R2_CS_bucket

def calculateR2Decile(dl, beta, mu, charName, decile=10):
	for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
		char = I[mask, dl.getIdxByIndividualFeature(charName)]
		R = R[mask]

	splits = np.sum(mask, axis=1).cumsum()[:-1]
	R_split = np.split(R, splits)
	beta_split = np.split(beta, splits)
	char_split = np.split(char, splits)
	mu_split = np.split(mu, splits)
	R_decile = []
	residual_decile = []

	for R_t, mu_t, beta_t, char_t in zip(R_split, mu_split, beta_split, char_split):
		tmp_t = sorted(list(zip(R_t, mu_t, beta_t, char_t)), key=lambda t:-t[3])
		tmp_t_decile = np.split(tmp_t, np.arange(1,decile) * (len(tmp_t) // decile))
		R_t_decile = []
		mu_t_decile = []
		beta_t_decile = []

		for i in range(decile):
			R_t_decile_list, mu_t_decile_list, beta_t_decile_list, _ = list(zip(*tmp_t_decile[i]))
			R_t_decile.append(np.mean(R_t_decile_list))
			mu_t_decile.append(np.mean(mu_t_decile_list))
			beta_t_decile.append(np.mean(beta_t_decile_list))

		R_t_decile = np.array(R_t_decile)
		mu_t_decile = np.array(mu_t_decile)
		beta_t_decile = np.array(beta_t_decile)
		R_hat_t_decile = beta_t_decile.dot(mu_t_decile) / beta_t_decile.dot(beta_t_decile) * beta_t_decile
		residual_t_decile = R_t_decile - R_hat_t_decile
		R_decile.append(R_t_decile)
		residual_decile.append(residual_t_decile)

	R_decile = np.array(R_decile)
	residual_decile = np.array(residual_decile)
	R_2 = R2(R_decile, residual_decile)
	R_2_decile = np.array([R2(R_decile[:,[i]], residual_decile[:,[i]]) for i in range(decile)])
	return R_2, R_2_decile

def calculateR2DoubleSorted5(dl, beta, mu, charName1, charName2):
	for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
		char1 = I[mask, dl.getIdxByIndividualFeature(charName1)]
		char2 = I[mask, dl.getIdxByIndividualFeature(charName2)]
		R = R[mask]

	splits = np.sum(mask, axis=1).cumsum()[:-1]
	R_split = np.split(R, splits)
	beta_split = np.split(beta, splits)
	char1_split = np.split(char1, splits)
	char2_split = np.split(char2, splits)
	mu_split = np.split(mu, splits)
	R_bucket = []
	residual_bucket = []

	for R_t, mu_t, beta_t, char1_t, char2_t in zip(R_split, mu_split, beta_split, char1_split, char2_split):
		R_t_bucket = np.zeros(25)
		mu_t_bucket = np.zeros(25)
		beta_t_bucket = np.zeros(25)
		cnt_t_bucket = np.zeros(25, dtype=int)

		for R_i, mu_i, beta_i, char1_i, char2_i in list(zip(R_t, mu_t, beta_t, char1_t, char2_t)):
			idx_char1 = int((char1_i - 1e-8) * 5)
			idx_char2 = int((char2_i - 1e-8) * 5)
			idx = idx_char1 * 5 + idx_char2
			R_t_bucket[idx] += R_i
			mu_t_bucket[idx] += mu_i
			beta_t_bucket[idx] += beta_i
			cnt_t_bucket[idx] += 1

		assert(cnt_t_bucket.sum() == len(R_t))
		R_t_bucket = R_t_bucket / cnt_t_bucket
		mu_t_bucket = mu_t_bucket / cnt_t_bucket
		beta_t_bucket = beta_t_bucket / cnt_t_bucket
		R_hat_t_bucket = beta_t_bucket.dot(mu_t_bucket) / beta_t_bucket.dot(beta_t_bucket) * beta_t_bucket
		residual_t_bucket = R_t_bucket - R_hat_t_bucket
		R_bucket.append(R_t_bucket)
		residual_bucket.append(residual_t_bucket)

	R_bucket = np.array(R_bucket)
	residual_bucket = np.array(residual_bucket)
	R_2 = R2(R_bucket, residual_bucket)
	R_2_bucket = np.array([R2(R_bucket[:,[i]], residual_bucket[:,[i]]) for i in range(25)])
	return R_2, R_2_bucket