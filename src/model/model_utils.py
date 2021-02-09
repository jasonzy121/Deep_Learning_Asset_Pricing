import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import BasicRNNCell
from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops.rnn_cell import MultiRNNCell

def create_rnn_cell(cell_type, num_units, num_layers=1, dp_input_keep_prob=1.0, dp_output_keep_prob=1.0):
	def single_cell(num_units):
		if cell_type == 'rnn':
			cell_class = BasicRNNCell
		elif cell_type == 'gru':
			cell_class = GRUCell
		elif cell_type == 'lstm':
			cell_class = LSTMCell
		else:
			raise ValueError('Cell Type Not Supported! ')

		if dp_input_keep_prob != 1.0 or dp_output_keep_prob != 1.0:
			return DropoutWrapper(cell_class(num_units=num_units),
				input_keep_prob=dp_input_keep_prob,
				output_keep_prob=dp_output_keep_prob)
		else:
			return cell_class(num_units=num_units)

	assert(len(num_units) == num_layers)
	if num_layers > 1:
		return MultiRNNCell([single_cell(num_units[i]) for i in range(num_layers)])
	else:
		return single_cell(num_units[0])

def initial_state_size(cell_type, num_units):
	state_size = sum(num_units)
	if cell_type == 'rnn' or cell_type == 'gru':
		return state_size
	elif cell_type == 'lstm':
		return state_size * 2

def getFactor(beta, dl, normalized=False, norm='l2'):
	for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
		R_reshape = R[mask]
		splits = np.sum(mask, axis=1).cumsum()[:-1]
		beta_list = np.split(beta, splits)
		R_list = np.split(R_reshape, splits)
		F_list = []
		for R_i, beta_i in zip(R_list, beta_list):
			if normalized:
				if norm == 'l1':
					F_list.append(R_i.dot(beta_i) / np.absolute(beta_i).sum())
				else:
					F_list.append(R_i.dot(beta_i) / np.sqrt(beta_i.dot(beta_i)))
			else:
				F_list.append(R_i.dot(beta_i) / beta_i.dot(beta_i))
		return np.array(F_list)

def decomposeReturn(w, dl):
	for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
		R_reshape = R[mask]
		splits = np.sum(mask, axis=1).cumsum()[:-1]
		w_list = np.split(w, splits)
		R_list = np.split(R_reshape, splits)
		R_hat_list = []
		residual_list = []
		for R_i, w_i in zip(R_list, w_list):
			R_hat_i = w_i.dot(R_i) / w_i.dot(w_i) * w_i
			residual_i = R_i - R_hat_i
			R_hat_list.append(R_hat_i)
			residual_list.append(residual_i)
		R_hat = np.zeros_like(mask, dtype=float)
		residual = np.zeros_like(mask, dtype=float)
		R_hat[mask] = np.concatenate(R_hat_list)
		residual[mask] = np.concatenate(residual_list)
	return R_hat, residual, mask, R

def calculateStatistics(w, dl):
	R_hat, residual, mask, R = decomposeReturn(w, dl)
	T = mask.shape[0]                                                                                                                                                             
	T_i = np.sum(mask, axis=0)
	N_t = np.sum(mask, axis=1)
	stat1 = 1 - np.mean(np.square(residual).sum(axis=1) / N_t) / np.mean(np.square(R * mask).sum(axis=1) / N_t) # EV
	stat2 = 1 - np.mean(np.square(residual.sum(axis=0) / T_i)) / np.mean(np.square((R * mask).sum(axis=0) / T_i)) # XS-R2
	stat3 = 1 - np.mean(np.square(residual.sum(axis=0) / T_i) * T_i) / np.mean(np.square((R * mask).sum(axis=0) / T_i) * T_i) # XS-R2 weighted
	return stat1, stat2, stat3