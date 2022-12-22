import copy
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.core.framework import summary_pb2
from tensorflow.python.layers.core import Dense
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple

from .model_base import ModelBase
from .model_utils import create_rnn_cell
from .model_utils import initial_state_size
from .model_utils import calculateStatistics
from src.utils import deco_print
from src.utils import sharpe
from src.utils import plotWeight1D
from src.utils import plotWeight2D
from src.utils import plotWeight3D
from src.utils import plot_variable_importance

class FeedForwardModelWithNA_GAN_Ensembled:
	def __init__(self, logdirs, model_params, mode, tSize, force_var_reuse=False, global_step=None):
		self._logdirs = logdirs
		self._model = FeedForwardModelWithNA_GAN(model_params, mode, tSize, force_var_reuse=force_var_reuse, global_step=global_step)

	def getZeroInitialState(self):
		return [self._model.getZeroInitialState() for _ in self._logdirs]

	def getNextInitialState(self, sess, dl, initial_state):
		INITIAL_next = []
		for logdir, INITIAL in zip(self._logdirs, initial_state):
			self._model.loadSavedModel(sess, logdir)
			INITIAL_next.append(self._model.getNextInitialState(sess, dl, INITIAL))
		return INITIAL_next

	### correct normalized weight
	def getWeightWithData(self, sess, dl, initial_state=None, normalized=False):
		if initial_state is None:
			initial_state = [None for _ in self._logdirs]
		w = []
		for logdir, INITIAL in zip(self._logdirs, initial_state):
			self._model.loadSavedModel(sess, logdir)
			w.append(self._model.getWeightWithData(sess, dl, initial_state=INITIAL, normalized=False))
		w = np.array(w).mean(axis=0)
		if normalized:
			for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
				w_list = np.split(w, np.sum(mask, axis=1).cumsum()[:-1])
				w = np.concatenate([item / np.absolute(item).sum() for item in w_list])
		return w

	def getSDF(self, sess, dl, initial_state=None):
		if initial_state is None:
			initial_state = [None for _ in self._logdirs]
		SDF = []
		for logdir, INITIAL in zip(self._logdirs, initial_state):
			self._model.loadSavedModel(sess, logdir)
			SDF.append(self._model.getSDF(sess, dl, initial_state=INITIAL))
		return np.array(SDF).mean(axis=0)

	### correct normalized SDF
	def getNormalizedSDF(self, sess, dl, initial_state=None):
		w = self.getWeightWithData(sess, dl, initial_state=initial_state, normalized=True)
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			R_weighted_list = np.split(R[mask] * w, np.sum(mask, axis=1).cumsum()[:-1])
			SDF = np.array([[item.sum()] for item in R_weighted_list]) + 1
		return SDF

	def getSDFFactor(self, sess, dl, initial_state=None):
		return 1 - self.getSDF(sess, dl, initial_state=initial_state)

	### add normalized SDF factor
	def getNormalizedSDFFactor(self, sess, dl, initial_state=None):
		return 1 - self.getNormalizedSDF(sess, dl, initial_state=initial_state)

	def calculateStatistics(self, sess, dl, initial_state=None):
		w = self.getWeightWithData(sess, dl, initial_state=initial_state)
		return calculateStatistics(w, dl)

	def plotIndividualFeatureImportance(self, sess, dl, plotPath=None, initial_state=None, top=30, figsize=(8,6)):
		if initial_state == None:
			initial_state = [None for _ in self._logdirs]
		gradients_list = []
		for logdir, INITIAL in zip(self._logdirs, initial_state):
			if not os.path.exists(os.path.join(logdir, 'ave_absolute_gradient.npy')):
				self._model.loadSavedModel(sess, logdir)
				self._model._saveIndividualFeatureImportance(sess, dl, logdir, initial_state=INITIAL)
			gradients_list.append(np.load(os.path.join(logdir, 'ave_absolute_gradient.npy')))
		gradients = np.array(gradients_list).mean(axis=0)

		gradients_sorted = sorted([(idx, gradients[idx], dl.getIndividualFeatureByIdx(idx)) for idx in range(len(gradients))], key=lambda t:-t[1])
		imp = [item for _, item, _ in gradients_sorted]
		var = [item for _, _, item in gradients_sorted]
		self._variable_rank = var
		var2color, color2category = dl.getIndividualFeatureColarLabelMap()
		labelColor = [var2color[item] for item in var]
		plot_variable_importance(var, imp, labelColor, plotPath=plotPath, top=top, figsize=figsize, color2category=color2category)

	def plotMacroFeatureImportance(self, sess, dl, plotPath=None, initial_state=None, top=30, figsize=(8,6)):
		if initial_state == None:
			initial_state = [None for _ in self._logdirs]
		gradients_list = []
		for logdir, INITIAL in zip(self._logdirs, initial_state):
			if not os.path.exists(os.path.join(logdir, 'ave_absolute_gradient_macro.npy')):
				self._model.loadSavedModel(sess, logdir)
				self._model._saveMacroFeatureImportance(sess, dl, logdir, initial_state=INITIAL)
			gradients_list.append(np.load(os.path.join(logdir, 'ave_absolute_gradient_macro.npy')))
		gradients = np.array(gradients_list).mean(axis=0)

		gradients_sorted = sorted([(idx, gradients[idx], dl.getMacroFeatureByIdx(idx)) for idx in range(len(gradients))], key=lambda t:-t[1])
		imp = [item for _, item, _ in gradients_sorted]
		var = [item for _, _, item in gradients_sorted]
		labelColor = ['red'] * self._model._macro_feature_dim
		plot_variable_importance(var, imp, labelColor, plotPath=plotPath, top=top, figsize=figsize)

	def plotWeight1DWithChar(self, sess, dl, charName='All', plotPath=None, sampleFreqPerAxis=50, figsize=(8,6), xlim=[-0.5,0.5]):
		meanMacroFeature, stdMacroFeature = dl.getMacroFeatureMeanStd()
		if charName == 'All':
			idxList = [i for i in range(self._model._individual_feature_dim)]
		else:
			idxList = [dl.getIdxByIndividualFeature(charName)]
		x = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1)
		time_start = time.time()
		cnt = 0
		for idx in idxList:
			xlabel = dl.getFeatureByIdx(idx)
			v_list = []
			for logdir in self._logdirs:
				self._model.loadSavedModel(sess, logdir)
				wFun = self._model.construct1DNonlinearFunction(sess, idx, 
					meanMacroFeature=meanMacroFeature, 
					stdMacroFeature=stdMacroFeature)
				v = np.zeros_like(x)
				for i in range(len(x)):
					v[i] = wFun(x[i])
				v_list.append(v)
			v = np.array(v_list).mean(axis=0)
			cnt += 1
			time_last = time.time() - time_start
			time_est = time_last / cnt * len(idxList)
			deco_print('Plotting Variable: %s\tElapse / Estimate: %.2fs / %.2fs' %(xlabel, time_last, time_est))
			plotWeight1D(x, -v, xlabel, plotPath=plotPath, idx=idx, figsize=figsize)

	def plotWeight(self, sess, dl, plotPath=None, sampleFreqPerAxis=50, figsize=(8,6), xlim=[-0.5,0.5]):
		meanMacroFeature, stdMacroFeature = dl.getMacroFeatureMeanStd()
		deco_print('Please enter the idx for the plot you want! ')
		idx_plot = int(input('Please select from 1.1D, 2.2D Contour 3.2D Contour Slice 4.1D Curves (2 Variables) 5.1D Curves (3 Variables): '))

		if idx_plot == 1:
			idx = int(input('Enter variable idx for x (0 - %d, %d - %d): ' \
				%(self._model._individual_feature_dim - 1, self._model._individual_feature_dim, self._model._individual_feature_dim + self._model._macro_feature_dim - 1)))
			xlabel = dl.getFeatureByIdx(idx)
			# x_left = float(input('Enter variate lower bound for %d (%s): ' %(idx, xlabel)))
			# x_right = float(input('Enter variate upper bound for %d (%s): ' %(idx, xlabel)))
			x = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1)

			v_list = []
			for logdir in self._logdirs:
				self._model.loadSavedModel(sess, logdir)
				wFun = self._model.construct1DNonlinearFunction(sess, idx, 
					meanMacroFeature=meanMacroFeature, 
					stdMacroFeature=stdMacroFeature)
				v = np.zeros_like(x)
				for i in range(len(x)):
					v[i] = wFun(x[i])
				v_list.append(v)
			v = np.array(v_list).mean(axis=0)
			plotWeight1D(x, -v, xlabel, plotPath=plotPath, idx=idx, figsize=figsize)

		if idx_plot == 4:
			quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
			idx_x = int(input('Enter variable idx for x (0 - %d, %d - %d): ' \
				%(self._model._individual_feature_dim - 1, self._model._individual_feature_dim, self._model._individual_feature_dim + self._model._macro_feature_dim - 1)))
			xlabel = dl.getFeatureByIdx(idx_x)
			idx_y = int(input('Enter variable idx for y (0 - %d, %d - %d): ' \
				%(self._model._individual_feature_dim - 1, self._model._individual_feature_dim, self._model._individual_feature_dim + self._model._macro_feature_dim - 1)))
			ylabel = dl.getFeatureByIdx(idx_y)
			x = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1)
			v = np.zeros((len(quantiles), len(x)))

			for j in range(len(quantiles)):
				v_list = []
				for logdir in self._logdirs:
					self._model.loadSavedModel(sess, logdir)
					wFun = self._model.construct2DNonlinearFunction(sess, idx_x, idx_y, 
						meanMacroFeature=meanMacroFeature,
						stdMacroFeature=stdMacroFeature)
					v_j = np.zeros_like(x)
					for i in range(len(x)):
						v_j[i] = wFun(x[i], quantiles[j]-0.5)
					v_list.append(v_j)
				v[j] = np.array(v_list).mean(axis=0)
			plotWeight1D(x, -v, xlabel, ylabel=ylabel, idx_y=idx_y, plotPath=plotPath, idx=idx_x, figsize=figsize)

		if idx_plot == 5:
			quantiles = [0.2, 0.5, 0.7]
			idx_x = int(input('Enter variable idx for x (0 - %d, %d - %d): ' \
				%(self._model._individual_feature_dim - 1, self._model._individual_feature_dim, self._model._individual_feature_dim + self._model._macro_feature_dim - 1)))
			xlabel = dl.getFeatureByIdx(idx_x)
			idx_y = int(input('Enter variable idx for y (0 - %d, %d - %d): ' \
				%(self._model._individual_feature_dim - 1, self._model._individual_feature_dim, self._model._individual_feature_dim + self._model._macro_feature_dim - 1)))
			ylabel = dl.getFeatureByIdx(idx_y)
			idx_z = int(input('Enter variable idx for z (0 - %d, %d - %d): ' \
				%(self._model._individual_feature_dim - 1, self._model._individual_feature_dim, self._model._individual_feature_dim + self._model._macro_feature_dim - 1)))
			zlabel = dl.getFeatureByIdx(idx_z)
			x = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1)
			v = np.zeros((len(quantiles), len(quantiles), len(x)))
			for jz in range(len(quantiles)):
				v_list = []
				for logdir in self._logdirs:
					self._model.loadSavedModel(sess, logdir)
					wFun = self._model.construct2DNonlinearFunction(sess, idx_x, idx_y, idx_z=idx_z, v_z=quantiles[jz]-0.5, 
						meanMacroFeature=meanMacroFeature,
						stdMacroFeature=stdMacroFeature)
					v_j = np.zeros((len(quantiles), len(x)))
					for jy in range(len(quantiles)):
						for i in range(len(x)):
							v_j[jy,i] = wFun(x[i], quantiles[jy]-0.5)
					v_list.append(v_j)
				v[:,jz,:] = np.array(v_list).mean(axis=0)
			plotWeight1D(x, -v, xlabel, 
				ylabel=ylabel, idx_y=idx_y, zlabel=zlabel, idx_z=idx_z, quantiles=quantiles,
				plotPath=plotPath, idx=idx_x, figsize=figsize)

		if idx_plot == 2:
			idx_x = int(input('Enter variable idx for x (0 - %d, %d - %d): ' \
				%(self._model._individual_feature_dim - 1, self._model._individual_feature_dim, self._model._individual_feature_dim + self._model._macro_feature_dim - 1)))
			xlabel = dl.getFeatureByIdx(idx_x)
			# x_left = float(input('Enter variate lower bound for %d (%s): ' %(idx_x, xlabel)))
			# x_right = float(input('Enter variate upper bound for %d (%s): ' %(idx_x, xlabel)))
			idx_y = int(input('Enter variable idx for y (0 - %d, %d - %d): ' \
				%(self._model._individual_feature_dim - 1, self._model._individual_feature_dim, self._model._individual_feature_dim + self._model._macro_feature_dim - 1)))
			ylabel = dl.getFeatureByIdx(idx_y)
			# y_left = float(input('Enter variate lower bound for %d (%s): ' %(idx_y, ylabel)))
			# y_right = float(input('Enter variate upper bound for %d (%s): ' %(idx_y, ylabel)))
			x = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1)
			y = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1)
			x, y = np.meshgrid(x, y)

			v_list = []
			for logdir in self._logdirs:
				self._model.loadSavedModel(sess, logdir)
				wFun = self._model.construct2DNonlinearFunction(sess, idx_x, idx_y, 
					meanMacroFeature=meanMacroFeature,
					stdMacroFeature=stdMacroFeature)
				v = np.zeros_like(x)
				for i in range(v.shape[0]):
					for j in range(v.shape[1]):
						v[i,j] = wFun(x[i,j], y[i,j])
				v_list.append(v)
			v = np.array(v_list).mean(axis=0)
			plotWeight2D(x, y, -v, xlabel, ylabel, plotPath=plotPath, idx_x=idx_x, idx_y=idx_y, figsize=figsize)

		if idx_plot == 3:
			idx_x = int(input('Enter variable idx for x (0 - %d, %d - %d): ' \
				%(self._model._individual_feature_dim - 1, self._model._individual_feature_dim, self._model._individual_feature_dim + self._model._macro_feature_dim - 1)))
			xlabel = dl.getFeatureByIdx(idx_x)
			# x_left = float(input('Enter variate lower bound for %d (%s): ' %(idx_x, xlabel)))
			# x_right = float(input('Enter variate upper bound for %d (%s): ' %(idx_x, xlabel)))
			idx_y = int(input('Enter variable idx for y (0 - %d, %d - %d): ' \
				%(self._model._individual_feature_dim - 1, self._model._individual_feature_dim, self._model._individual_feature_dim + self._model._macro_feature_dim - 1)))
			ylabel = dl.getFeatureByIdx(idx_y)
			# y_left = float(input('Enter variate lower bound for %d (%s): ' %(idx_y, ylabel)))
			# y_right = float(input('Enter variate upper bound for %d (%s): ' %(idx_y, ylabel)))
			idx_z = int(input('Enter variable idx for z (0 - %d, %d - %d): ' \
				%(self._model._individual_feature_dim - 1, self._model._individual_feature_dim, self._model._individual_feature_dim + self._model._macro_feature_dim - 1)))
			zlabel = dl.getFeatureByIdx(idx_z)
			# z_left = float(input('Enter variate lower bound for %d (%s): ' %(idx_z, zlabel)))
			# z_right = float(input('Enter variate upper bound for %d (%s): ' %(idx_z, zlabel)))

			x = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1)
			y = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1)
			zs = np.linspace(xlim[0], xlim[1], 4)
			x, y = np.meshgrid(x, y)

			v_list = []
			for logdir in self._logdirs:
				self._model.loadSavedModel(sess, logdir)
				v = np.zeros((len(y), len(x), len(zs)))
				for k in range(len(zs)):
					z = zs[k]
					wFun = self._model.construct2DNonlinearFunction(sess, idx_x, idx_y, idx_z=idx_z, v_z=z, 
						meanMacroFeature=meanMacroFeature,
						stdMacroFeature=stdMacroFeature)
					for i in range(len(y)):
						for j in range(len(x)):
							v[i,j,k] = wFun(x[i,j], y[i,j])
				v_list.append(v)
			v = np.array(v_list).mean(axis=0)
			plotWeight3D(x, y, -v, zs, xlabel, ylabel, zlabel, plotPath=plotPath, idx_x=idx_x, idx_y=idx_y, idx_z=idx_z, figsize=figsize)

class FeedForwardModelWithNA_GAN(ModelBase):
	def __init__(self, model_params, mode, tSize, force_var_reuse=False, global_step=None):
		super(FeedForwardModelWithNA_GAN, self).__init__(model_params, mode, global_step)
		self._force_var_reuse = force_var_reuse
		self._tSize = tSize
		self._macro_feature_dim = self.model_params['macro_feature_dim']
		self._individual_feature_dim = self.model_params['individual_feature_dim']

		self._I_macro_placeholder = tf.placeholder(dtype=tf.float32, shape=[self._tSize, self._macro_feature_dim], name='macroFeaturePlaceholder')
		self._I_placeholder = tf.placeholder(dtype=tf.float32, shape=[self._tSize, None, self._individual_feature_dim], name='individualFeaturePlaceholder')
		self._R_placeholder = tf.placeholder(dtype=tf.float32, shape=[self._tSize, None], name='returnPlaceholder')
		self._mask_placeholder = tf.placeholder(dtype=tf.bool, shape=[self._tSize, None], name='maskPlaceholder')
		self._dropout_placeholder = tf.placeholder_with_default(1.0, shape=[], name='Dropout')

		self._nSize = tf.shape(self._R_placeholder)[1]

		self._residual_loss_factor = self.model_params['residual_loss_factor'] if 'residual_loss_factor' in self.model_params else 0.0

		if self.model_params['weighted_loss']:
			self._loss_weight = tf.placeholder(dtype=tf.float32, shape=[None], name='weightPlaceholder')

		if self.model_params['use_rnn']:
			self._state_size = initial_state_size(self.model_params['cell_type_rnn'], self.model_params['num_units_rnn'])
			self._initial_state_placeholder = tf.placeholder(dtype=tf.float32, shape=[1, self._state_size])

			if self.model_params['cell_type_rnn'] == 'lstm':
				splits = [2*unit for unit in self.model_params['num_units_rnn']]
				self._initial_state = tuple([LSTMStateTuple(*tf.split(value=layer_state, num_or_size_splits=2, axis=1))
						for layer_state in tf.split(value=self._initial_state_placeholder, num_or_size_splits=splits, axis=1)])
			else:
				self._initial_state = tuple(tf.split(value=self._initial_state_placeholder, num_or_size_splits=self.model_params['num_units_rnn'], axis=1))

			if self.model_params['num_layers_rnn'] == 1:
				self._initial_state = self._initial_state[0]

			self._rnn_input = tf.expand_dims(self._I_macro_placeholder, axis=0)

		with tf.variable_scope(name_or_scope='Model_Layer', reuse=self._force_var_reuse):
			self._build_forward_pass_graph()
			self._loss_unc = self._add_loss(tf.ones(shape=(1, self._tSize, self._nSize))) # unconditional loss
			self._loss_residual = self._add_loss_residual()

		if self._mode == 'train':
			with tf.variable_scope(name_or_scope='Moment_Layer', reuse=self._force_var_reuse):
				self._build_forward_pass_graph_moment()
				self._loss = self._add_loss(self._h) # conditional loss
			self._train_model_op_unc = self._build_train_op(self._loss_unc + self._residual_loss_factor * self._loss_residual, scope='Model_Layer')
			self._update_moment_op = self._build_train_op(-self._loss, scope='Moment_Layer')
			self._train_model_op = self._build_train_op(self._loss + self._residual_loss_factor * self._loss_residual, scope='Model_Layer')

	def _build_forward_pass_graph_moment(self):
		if self.model_params['use_rnn']:
			with tf.variable_scope('RNN_Layer'):
				rnn_cell = create_rnn_cell(
					cell_type=self.model_params['cell_type_rnn_moment'],
					num_units=self.model_params['num_units_rnn_moment'],
					num_layers=self.model_params['num_layers_rnn_moment'],
					dp_input_keep_prob=self._dropout_placeholder,
					dp_output_keep_prob=1.0)

				rnn_outputs, rnn_state = tf.nn.dynamic_rnn(
					cell=rnn_cell,
					inputs=self._rnn_input,
					initial_state=None,
					dtype=tf.float32)
				self._macro_nn_input_moment = tf.squeeze(rnn_outputs, axis=0)
		else:
			self._macro_nn_input_moment = self._I_macro_placeholder

		with tf.variable_scope('NN_Layer'):
			I_macro_tile = tf.tile(tf.expand_dims(self._macro_nn_input_moment, axis=1), [1,self._nSize,1]) # T * N * macro_feature_dim
			I_concat = tf.concat([I_macro_tile, self._I_placeholder], axis=2) # T * N * (macro_feature_dim + individual_feature_dim)

			h_l = I_concat
			for l in range(self.model_params['num_layers_moment']):
				with tf.variable_scope('dense_layer_%d' %l):
					layer_l = Dense(units=self.model_params['hidden_dim_moment'][l], activation=tf.nn.relu)
					h_l = layer_l(h_l)
					h_l = tf.nn.dropout(h_l, self._dropout_placeholder)

			with tf.variable_scope('last_dense_layer'):
				layer = Dense(units=self.model_params['num_condition_moment'], activation=tf.nn.tanh)
				self._h = tf.transpose(layer(h_l), perm=[2,0,1]) # num_condition_moment * T * N

	def _build_forward_pass_graph(self):
		if self.model_params['use_rnn']:
			with tf.variable_scope('RNN_Layer'):
				rnn_cell = create_rnn_cell(
					cell_type=self.model_params['cell_type_rnn'],
					num_units=self.model_params['num_units_rnn'],
					num_layers=self.model_params['num_layers_rnn'],
					dp_input_keep_prob=self._dropout_placeholder,
					dp_output_keep_prob=1.0)

				rnn_outputs, rnn_state = tf.nn.dynamic_rnn(
					cell=rnn_cell,
					inputs=self._rnn_input,
					initial_state=self._initial_state,
					dtype=tf.float32)
				self._macro_nn_input = tf.squeeze(rnn_outputs, axis=0)

				if self.model_params['cell_type_rnn'] == 'lstm':
					if self.model_params['num_layers_rnn'] == 1:
						self._rnn_last_state = tf.concat([rnn_state.c, rnn_state.h], axis=1)
					else:
						self._rnn_last_state = tf.concat([tf.concat([state_tuple.c, state_tuple.h], axis=1) for state_tuple in rnn_state], axis=1)
				else:
					if self.model_params['num_layers_rnn'] == 1:
						self._rnn_last_state = rnn_state
					else:
						self._rnn_last_state = tf.concat(rnn_state, axis=1)
		else:
			self._macro_nn_input = self._I_macro_placeholder

		with tf.variable_scope('NN_Layer'):
			I_macro_tile = tf.tile(tf.expand_dims(self._macro_nn_input, axis=1), [1,self._nSize,1]) # T * N * macro_feature_dim
			I_macro_masked = tf.boolean_mask(I_macro_tile, mask=self._mask_placeholder)
			I_masked = tf.boolean_mask(self._I_placeholder, mask=self._mask_placeholder)
			I_concat = tf.concat([I_masked, I_macro_masked], axis=1) # None * (macro_feature_dim + individual_feature_dim)
			R_masked = tf.boolean_mask(self._R_placeholder, mask=self._mask_placeholder)

			h_l = I_concat
			for l in range(self.model_params['num_layers']):
				with tf.variable_scope('dense_layer_%d' %l):
					layer_l = Dense(units=self.model_params['hidden_dim'][l], activation=tf.nn.relu)
					h_l = layer_l(h_l)
					h_l = tf.nn.dropout(h_l, self._dropout_placeholder)

			with tf.variable_scope('last_dense_layer'):
				layer = Dense(units=1)
				w = layer(h_l)
				self._w = tf.reshape(w, shape=[-1])

			weighted_R_masked = R_masked * self._w

		N_i = tf.reduce_sum(tf.to_int32(self._mask_placeholder), axis=1) # len T
		weighted_R_split = tf.split(weighted_R_masked, num_or_size_splits=N_i)
		if 'normalize_w' in self.model_params and self.model_params['normalize_w']:
			deco_print('Normalize weight by N!')
			N_bar = tf.reduce_mean(N_i)
			self._SDF = tf.expand_dims(tf.concat([tf.reduce_sum(item, keepdims=True) for item in weighted_R_split], axis=0) / tf.to_float(N_i) * tf.to_float(N_bar), axis=1) + 1
		else:
			self._SDF = tf.expand_dims(tf.concat([tf.reduce_sum(item, keepdims=True) for item in weighted_R_split], axis=0), axis=1) + 1

	def _add_loss(self, h):
		T_i = tf.reduce_sum(tf.to_float(self._mask_placeholder), axis=0) # len N
		empirical_mean = tf.reduce_sum(self._R_placeholder * tf.to_float(self._mask_placeholder) * self._SDF * h, axis=1) / T_i
		if self.model_params['weighted_loss']:
			loss_weight_normalized = self._loss_weight / tf.reduce_max(self._loss_weight)
			return tf.reduce_mean(tf.square(empirical_mean) * loss_weight_normalized)
		else:
			return tf.reduce_mean(tf.square(empirical_mean))

	### add residual loss
	def _add_loss_residual(self):
		N_i = tf.reduce_sum(tf.to_int32(self._mask_placeholder), axis=1) # len T
		R_masked = tf.boolean_mask(self._R_placeholder, mask=self._mask_placeholder)
		R_masked_list = tf.split(R_masked, num_or_size_splits=N_i)
		w_list = tf.split(self._w, num_or_size_splits=N_i)
		residual_square_list = []
		R_square_list = []
		for R_t, w_t in zip(R_masked_list, w_list):
			R_t_hat = tf.reduce_sum(R_t * w_t) / tf.reduce_sum(w_t * w_t) * w_t
			residual_square_list.append(tf.reduce_mean(tf.square(R_t - R_t_hat)))
			R_square_list.append(tf.reduce_mean(tf.square(R_t)))
		return tf.reduce_mean(residual_square_list) / tf.reduce_mean(R_square_list)
	###

	def train(self, sess, dl, dl_valid, logdir, model_valid, loss_weight=None, loss_weight_valid=None,
			dl_test=None, model_test=None, loss_weight_test=None, 
			printOnConsole=True, printFreq=128, saveLog=True, saveBestFreq=128, ignoreEpoch=64):
		if self._mode != 'train':
			deco_print('ERROR: Model has no train op! ')
		else:
			### validation on loss and sharpe
			logdir_loss = os.path.join(logdir, 'loss')
			logdir_sharpe = os.path.join(logdir, 'sharpe')
			os.system('mkdir -p ' + logdir_loss)
			os.system('mkdir -p ' + logdir_sharpe)
			###

			saver = tf.train.Saver(max_to_keep=100)
			if saveLog:
				sw = tf.summary.FileWriter(logdir, sess.graph)

			best_valid_loss_unc = float('inf')
			best_valid_loss = float('inf')
			best_valid_sharpe_unc = float('-inf')
			best_valid_sharpe = float('-inf')

			sharpe_train = []
			sharpe_valid = []
			### evaluate test data
			evaluate_test_data = False
			if dl_test is not None and model_test is not None:
				evaluate_test_data = True
				sharpe_test = []

			if self.model_params['use_rnn']:
				INITIAL_train = self.getZeroInitialState()
			else:
				INITIAL_train = None

			### train unconditional loss
			time_start = time.time()
			deco_print('Start Training Unconditional Loss...')
			for epoch in range(self.model_params['num_epochs_unc']):
				for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=self.model_params['sub_epoch'])):
					feed_dict = {self._I_macro_placeholder:I_macro, 
								self._I_placeholder:I, 
								self._R_placeholder:R, 
								self._mask_placeholder:mask,
								self._dropout_placeholder:self.model_params['dropout']}
					if self.model_params['weighted_loss']:
						feed_dict[self._loss_weight] = loss_weight
					if self.model_params['use_rnn']:
						feed_dict[self._initial_state_placeholder] = INITIAL_train
					sess.run(fetches=[self._train_model_op_unc], feed_dict=feed_dict)

				### evaluate train loss / sharpe
				train_epoch_loss, INITIAL_valid = self.evaluate_loss(sess, dl, INITIAL_train, loss_weight)
				train_epoch_loss_residual = self.evaluate_loss_residual(sess, dl, INITIAL_train)
				train_epoch_sharpe = self.evaluate_sharpe(sess, dl, INITIAL_train)
				sharpe_train.append(train_epoch_sharpe)

				### evaluate valid loss / sharpe
				valid_epoch_loss, INITIAL_test = model_valid.evaluate_loss(sess, dl_valid, INITIAL_valid, loss_weight_valid)
				valid_epoch_loss_residual = model_valid.evaluate_loss_residual(sess, dl_valid, INITIAL_valid)
				valid_epoch_sharpe = model_valid.evaluate_sharpe(sess, dl_valid, INITIAL_valid)
				sharpe_valid.append(valid_epoch_sharpe)

				### evaluate test loss / sharpe
				if evaluate_test_data:
					test_epoch_loss, _ = model_test.evaluate_loss(sess, dl_test, INITIAL_test, loss_weight_test)
					test_epoch_loss_residual = model_test.evaluate_loss_residual(sess, dl_test, INITIAL_test)
					test_epoch_sharpe = model_test.evaluate_sharpe(sess, dl_test, INITIAL_test)
					sharpe_test.append(test_epoch_sharpe)

				### print loss / sharpe
				if printOnConsole and epoch % printFreq == 0:
					print('\n\n')
					deco_print('Doint epoch %d' %epoch)
					if evaluate_test_data:
						deco_print('Epoch %d train/valid/test loss: %0.4f/%0.4f/%0.4f' %(epoch, train_epoch_loss, valid_epoch_loss, test_epoch_loss))
						deco_print('Epoch %d train/valid/test loss (residual): %0.4f/%0.4f/%0.4f' %(epoch, train_epoch_loss_residual, valid_epoch_loss_residual, test_epoch_loss_residual))
						deco_print('Epoch %d train/valid/test sharpe: %0.4f/%0.4f/%0.4f' %(epoch, train_epoch_sharpe, valid_epoch_sharpe, test_epoch_sharpe))
					else:
						deco_print('Epoch %d train/valid loss: %0.4f/%0.4f' %(epoch, train_epoch_loss, valid_epoch_loss))
						deco_print('Epoch %d train/valid loss (residual): %0.4f/%0.4f' %(epoch, train_epoch_loss_residual, valid_epoch_loss_residual))
						deco_print('Epoch %d train/valid sharpe: %0.4f/%0.4f' %(epoch, train_epoch_sharpe, valid_epoch_sharpe))
				if saveLog:
					value_loss_train = summary_pb2.Summary.Value(tag='Train_epoch_loss', simple_value=train_epoch_loss)
					value_loss_residual_train = summary_pb2.Summary.Value(tag='Train_epoch_loss_residual', simple_value=train_epoch_loss_residual)
					value_loss_valid = summary_pb2.Summary.Value(tag='Valid_epoch_loss', simple_value=valid_epoch_loss)
					value_loss_residual_valid = summary_pb2.Summary.Value(tag='Valid_epoch_loss_residual', simple_value=valid_epoch_loss_residual)
					value_sharpe_train = summary_pb2.Summary.Value(tag='Train_epoch_sharpe', simple_value=train_epoch_sharpe)
					value_sharpe_valid = summary_pb2.Summary.Value(tag='Valid_epoch_sharpe', simple_value=valid_epoch_sharpe)
					if evaluate_test_data:
						value_loss_test = summary_pb2.Summary.Value(tag='Test_epoch_loss', simple_value=test_epoch_loss)
						value_loss_residual_test = summary_pb2.Summary.Value(tag='Test_epoch_loss_residual', simple_value=test_epoch_loss_residual)
						value_sharpe_test = summary_pb2.Summary.Value(tag='Test_epoch_sharpe', simple_value=test_epoch_sharpe)
						summary = summary_pb2.Summary(value=[value_loss_train, value_loss_valid, value_loss_test, value_loss_residual_train, value_loss_residual_valid, value_loss_residual_test, value_sharpe_train, value_sharpe_valid, value_sharpe_test])
					else:
						summary = summary_pb2.Summary(value=[value_loss_train, value_loss_valid, value_loss_residual_train, value_loss_residual_valid, value_sharpe_train, value_sharpe_valid])
					sw.add_summary(summary, global_step=epoch)
					sw.flush()

				### save epoch
				if epoch > ignoreEpoch:
					if valid_epoch_loss < best_valid_loss_unc:
						best_valid_loss_unc = valid_epoch_loss
						if printOnConsole and epoch % printFreq == 0:
							deco_print('Saving current best checkpoint (loss)')
						saver.save(sess, save_path=os.path.join(logdir_loss, 'model-best'))
					if valid_epoch_sharpe > best_valid_sharpe_unc:
						best_valid_sharpe_unc = valid_epoch_sharpe
						if printOnConsole and epoch % printFreq == 0:
							deco_print('Saving current best checkpoint (sharpe)')
						saver.save(sess, save_path=os.path.join(logdir_sharpe, 'model-best'))

				if saveBestFreq > 0 and (epoch+1) % saveBestFreq == 0:
					path_epoch_loss = os.path.join(logdir_loss,'UNC',str(epoch))
					path_best_loss = os.path.join(logdir_loss, 'model-best*')
					path_best_checkpoint_loss = os.path.join(logdir_loss, 'checkpoint')
					os.system('mkdir -p ' + path_epoch_loss)
					os.system('cp %s %s' %(path_best_loss, path_epoch_loss))
					os.system('cp %s %s' %(path_best_checkpoint_loss, path_epoch_loss))

					path_epoch_sharpe = os.path.join(logdir_sharpe,'UNC',str(epoch))
					path_best_sharpe = os.path.join(logdir_sharpe, 'model-best*')
					path_best_checkpoint_sharpe = os.path.join(logdir_sharpe, 'checkpoint')
					os.system('mkdir -p ' + path_epoch_sharpe)
					os.system('cp %s %s' %(path_best_sharpe, path_epoch_sharpe))
					os.system('cp %s %s' %(path_best_checkpoint_sharpe, path_epoch_sharpe))

				### time
				if printOnConsole and epoch % printFreq == 0:
					time_elapse = time.time() - time_start
					time_est = time_elapse / (epoch+1) * self.model_params['num_epochs_unc']
					deco_print('Epoch %d Elapse/Estimate: %0.2fs/%0.2fs' %(epoch, time_elapse, time_est))

			deco_print('Training Unconditional Loss Finished!\n')

			### update moment condition
			deco_print('Start Updating Moment Conditions...')
			self.loadSavedModel(sess, logdir_loss)
			for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=self.model_params['sub_epoch'])):
				best_moment_loss = float('-inf')
				feed_dict = {self._I_macro_placeholder:I_macro, 
							self._I_placeholder:I, 
							self._R_placeholder:R, 
							self._mask_placeholder:mask,
							self._dropout_placeholder:self.model_params['dropout']}
				if self.model_params['weighted_loss']:
					feed_dict[self._loss_weight] = loss_weight
				if self.model_params['use_rnn']:
					feed_dict[self._initial_state_placeholder] = INITIAL_train

				for epoch in range(self.model_params['num_epochs_moment']):
					_, loss = sess.run(fetches=[self._update_moment_op, self._loss], feed_dict=feed_dict)
					if loss > best_moment_loss:
						best_moment_loss = loss
						if printOnConsole and epoch % printFreq == 0:
							deco_print('Saving current best checkpoint (epoch %d)' %epoch)
						saver.save(sess, save_path=os.path.join(logdir_loss, 'model-best'))
			deco_print('Updating Moment Conditions Finished!\n')

			### train conditional loss
			time_start = time.time()
			deco_print('Start Training Conditional Loss...')
			self.loadSavedModel(sess, logdir_loss)
			for epoch in range(self.model_params['num_epochs']):
				for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=self.model_params['sub_epoch'])):
					feed_dict = {self._I_macro_placeholder:I_macro, 
								self._I_placeholder:I, 
								self._R_placeholder:R, 
								self._mask_placeholder:mask,
								self._dropout_placeholder:self.model_params['dropout']}
					if self.model_params['weighted_loss']:
						feed_dict[self._loss_weight] = loss_weight
					if self.model_params['use_rnn']:
						feed_dict[self._initial_state_placeholder] = INITIAL_train
					sess.run(fetches=[self._train_model_op], feed_dict=feed_dict)
				
				### evaluate train loss / sharpe
				train_epoch_loss, INITIAL_valid = self.evaluate_loss(sess, dl, INITIAL_train, loss_weight)
				train_epoch_loss_residual = self.evaluate_loss_residual(sess, dl, INITIAL_train)
				train_epoch_sharpe = self.evaluate_sharpe(sess, dl, INITIAL_train)
				sharpe_train.append(train_epoch_sharpe)

				### evaluate valid loss / sharpe
				valid_epoch_loss, INITIAL_test = model_valid.evaluate_loss(sess, dl_valid, INITIAL_valid, loss_weight_valid)
				valid_epoch_loss_residual = model_valid.evaluate_loss_residual(sess, dl_valid, INITIAL_valid)
				valid_epoch_sharpe = model_valid.evaluate_sharpe(sess, dl_valid, INITIAL_valid)
				sharpe_valid.append(valid_epoch_sharpe)

				### evaluate test loss / sharpe
				if evaluate_test_data:
					test_epoch_loss, _ = model_test.evaluate_loss(sess, dl_test, INITIAL_test, loss_weight_test)
					test_epoch_loss_residual = model_test.evaluate_loss_residual(sess, dl_test, INITIAL_test)
					test_epoch_sharpe = model_test.evaluate_sharpe(sess, dl_test, INITIAL_test)
					sharpe_test.append(test_epoch_sharpe)

				### print loss / sharpe
				if printOnConsole and epoch % printFreq == 0:
					print('\n\n')
					deco_print('Doint epoch %d' %epoch)
					if evaluate_test_data:
						deco_print('Epoch %d train/valid/test loss: %0.4f/%0.4f/%0.4f' %(epoch, train_epoch_loss, valid_epoch_loss, test_epoch_loss))
						deco_print('Epoch %d train/valid/test loss (residual): %0.4f/%0.4f/%0.4f' %(epoch, train_epoch_loss_residual, valid_epoch_loss_residual, test_epoch_loss_residual))
						deco_print('Epoch %d train/valid/test sharpe: %0.4f/%0.4f/%0.4f' %(epoch, train_epoch_sharpe, valid_epoch_sharpe, test_epoch_sharpe))
					else:
						deco_print('Epoch %d train/valid loss: %0.4f/%0.4f' %(epoch, train_epoch_loss, valid_epoch_loss))
						deco_print('Epoch %d train/valid loss (residual): %0.4f/%0.4f' %(epoch, train_epoch_loss_residual, valid_epoch_loss_residual))
						deco_print('Epoch %d train/valid sharpe: %0.4f/%0.4f' %(epoch, train_epoch_sharpe, valid_epoch_sharpe))
				if saveLog:
					value_loss_train = summary_pb2.Summary.Value(tag='Train_epoch_loss', simple_value=train_epoch_loss)
					value_loss_residual_train = summary_pb2.Summary.Value(tag='Train_epoch_loss_residual', simple_value=train_epoch_loss_residual)
					value_loss_valid = summary_pb2.Summary.Value(tag='Valid_epoch_loss', simple_value=valid_epoch_loss)
					value_loss_residual_valid = summary_pb2.Summary.Value(tag='Valid_epoch_loss_residual', simple_value=valid_epoch_loss_residual)
					value_sharpe_train = summary_pb2.Summary.Value(tag='Train_epoch_sharpe', simple_value=train_epoch_sharpe)
					value_sharpe_valid = summary_pb2.Summary.Value(tag='Valid_epoch_sharpe', simple_value=valid_epoch_sharpe)
					if evaluate_test_data:
						value_loss_test = summary_pb2.Summary.Value(tag='Test_epoch_loss', simple_value=test_epoch_loss)
						value_loss_residual_test = summary_pb2.Summary.Value(tag='Test_epoch_loss_residual', simple_value=test_epoch_loss_residual)
						value_sharpe_test = summary_pb2.Summary.Value(tag='Test_epoch_sharpe', simple_value=test_epoch_sharpe)
						summary = summary_pb2.Summary(value=[value_loss_train, value_loss_valid, value_loss_test, value_loss_residual_train, value_loss_residual_valid, value_loss_residual_test, value_sharpe_train, value_sharpe_valid, value_sharpe_test])
					else:
						summary = summary_pb2.Summary(value=[value_loss_train, value_loss_valid, value_loss_residual_train, value_loss_residual_valid, value_sharpe_train, value_sharpe_valid])
					sw.add_summary(summary, global_step=epoch+self.model_params['num_epochs_unc'])
					sw.flush()

				### save epoch
				if epoch > ignoreEpoch:
					if valid_epoch_loss < best_valid_loss:
						best_valid_loss = valid_epoch_loss
						if printOnConsole and epoch % printFreq == 0:
							deco_print('Saving current best checkpoint (loss)')
						saver.save(sess, save_path=os.path.join(logdir_loss, 'model-best'))
					if valid_epoch_sharpe > best_valid_sharpe:
						best_valid_sharpe = valid_epoch_sharpe
						if printOnConsole and epoch % printFreq == 0:
							deco_print('Saving current best checkpoint (sharpe)')
						saver.save(sess, save_path=os.path.join(logdir_sharpe, 'model-best'))

				if saveBestFreq > 0 and (epoch+1) % saveBestFreq == 0:
					path_epoch_loss = os.path.join(logdir_loss,'GAN',str(epoch))
					path_best_loss = os.path.join(logdir_loss, 'model-best*')
					path_best_checkpoint_loss = os.path.join(logdir_loss, 'checkpoint')
					os.system('mkdir -p ' + path_epoch_loss)
					os.system('cp %s %s' %(path_best_loss, path_epoch_loss))
					os.system('cp %s %s' %(path_best_checkpoint_loss, path_epoch_loss))

					path_epoch_sharpe = os.path.join(logdir_sharpe,'GAN',str(epoch))
					path_best_sharpe = os.path.join(logdir_sharpe, 'model-best*')
					path_best_checkpoint_sharpe = os.path.join(logdir_sharpe, 'checkpoint')
					os.system('mkdir -p ' + path_epoch_sharpe)
					os.system('cp %s %s' %(path_best_sharpe, path_epoch_sharpe))
					os.system('cp %s %s' %(path_best_checkpoint_sharpe, path_epoch_sharpe))

				### time
				if printOnConsole and epoch % printFreq == 0:
					time_elapse = time.time() - time_start
					time_est = time_elapse / (epoch+1) * self.model_params['num_epochs']
					deco_print('Epoch %d Elapse/Estimate: %0.2fs/%0.2fs' %(epoch, time_elapse, time_est))

			deco_print('Training Conditional Loss Finished!\n')

			### save last epoch
			deco_print('Saving last checkpoint')
			saver.save(sess, save_path=os.path.join(logdir, 'model-last'))

			if evaluate_test_data:
				return sharpe_train, sharpe_valid, sharpe_test
			else:
				return sharpe_train, sharpe_valid

	def evaluate_loss(self, sess, dl, initial_state=None, loss_weight=None):
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			feed_dict = {self._I_macro_placeholder:I_macro, 
						self._I_placeholder:I, 
						self._R_placeholder:R, 
						self._mask_placeholder:mask, 
						self._dropout_placeholder:1.0}
			if self.model_params['weighted_loss']:
				feed_dict[self._loss_weight] = loss_weight
			if self.model_params['use_rnn']:
				feed_dict[self._initial_state_placeholder] = initial_state
				loss, INITIAL_next = sess.run(fetches=[self._loss_unc, self._rnn_last_state], feed_dict=feed_dict)
				return loss, INITIAL_next
			else:
				loss, = sess.run(fetches=[self._loss_unc], feed_dict=feed_dict)
				return loss, None

	### evaluate residual loss
	def evaluate_loss_residual(self, sess, dl, initial_state=None):
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			feed_dict = {self._I_macro_placeholder:I_macro, 
						self._I_placeholder:I, 
						self._R_placeholder:R, 
						self._mask_placeholder:mask, 
						self._dropout_placeholder:1.0}
			if self.model_params['use_rnn']:
				feed_dict[self._initial_state_placeholder] = initial_state
			loss, = sess.run(fetches=[self._loss_residual], feed_dict=feed_dict)
			return loss
	###

	def evaluate_sharpe(self, sess, dl, initial_state=None, normalized=False):
		if normalized:
			SDF = self.getNormalizedSDF(sess, dl, initial_state=initial_state)[:,0]
		else:
			SDF = self.getSDF(sess, dl, initial_state)[:,0]
		portfolio = 1 - SDF
		return sharpe(portfolio)

	def getSDF(self, sess, dl, initial_state=None):
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			feed_dict = {self._I_macro_placeholder:I_macro,
						self._I_placeholder:I,
						self._R_placeholder:R,
						self._mask_placeholder:mask,
						self._dropout_placeholder:1.0}
			if self.model_params['use_rnn']:
				feed_dict[self._initial_state_placeholder] = initial_state
			SDF, = sess.run(fetches=[self._SDF], feed_dict=feed_dict)
		return SDF

	def getNormalizedSDF(self, sess, dl, initial_state=None):
		w = self.getWeightWithData(sess, dl, initial_state=initial_state, normalized=True)
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			R_weighted_list = np.split(R[mask] * w, np.sum(mask, axis=1).cumsum()[:-1])
			SDF = np.array([[item.sum()] for item in R_weighted_list]) + 1
		return SDF

	def getNextInitialState(self, sess, dl, initial_state):
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			feed_dict = {self._I_macro_placeholder:I_macro,
						self._initial_state_placeholder:initial_state,
						self._dropout_placeholder:1.0}
			INITIAL_next, = sess.run(fetches=[self._rnn_last_state], feed_dict=feed_dict)
		return INITIAL_next

	def getStateMacroVariables(self, sess, dl, initial_state):
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			feed_dict = {self._I_macro_placeholder:I_macro,
						self._initial_state_placeholder:initial_state,
						self._dropout_placeholder:1.0}
			SMV, = sess.run(fetches=[self._macro_nn_input], feed_dict=feed_dict)
		return SMV

	def getZeroInitialState(self):
		return np.zeros(shape=(1,self._state_size))

	def getWeightWithData(self, sess, dl, initial_state=None, normalized=False):
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			feed_dict = {self._I_macro_placeholder:I_macro,
						self._I_placeholder:I,
						self._R_placeholder:R,
						self._mask_placeholder:mask,
						self._dropout_placeholder:1.0}
			if self.model_params['use_rnn']:
				feed_dict[self._initial_state_placeholder] = initial_state
			w, = sess.run(fetches=[self._w], feed_dict=feed_dict)
		if normalized:
			w_list = np.split(w, np.sum(mask, axis=1).cumsum()[:-1])
			w = np.concatenate([item / np.absolute(item).sum() for item in w_list])
		return w

	def calculateStatistics(self, sess, dl, initial_state=None):
		w = self.getWeightWithData(sess, dl, initial_state=initial_state)
		return calculateStatistics(w, dl)

	def getWeight(self, sess, individualFeature, macroFeature):
		### require self._tSize = 1
		I_macro = np.reshape(macroFeature, [1,self._macro_feature_dim])
		I = np.reshape(individualFeature, [1, 1, self._individual_feature_dim])
		R = np.array([[0.0]], dtype=float)
		mask = np.array([[True]], dtype=bool)
		feed_dict = {self._I_macro_placeholder:I_macro, 
					self._I_placeholder:I, 
					self._R_placeholder:R,
					self._mask_placeholder:mask,
					self._dropout_placeholder:1.0}
		if self.model_params['use_rnn']:
			initial_state = self.getZeroInitialState()
			feed_dict[self._initial_state_placeholder] = initial_state
		w, = sess.run(fetches=[self._w], feed_dict=feed_dict)
		return w[0]

	def construct1DNonlinearFunction(self, sess, idx, meanMacroFeature=None, stdMacroFeature=None):
		### require self._tSize = 1
		I_macro = np.zeros(shape=(1, self._macro_feature_dim))
		# I = np.ones(shape=(1, 1, self._individual_feature_dim)) * 0.5
		I = np.zeros(shape=(1, 1, self._individual_feature_dim))
		def f(x):
			if idx < self._individual_feature_dim:
				I[0,0,idx] = x
			else:
				idx_macro = idx - self._individual_feature_dim
				I_macro[0,idx_macro] = (x - meanMacroFeature[idx_macro]) / stdMacroFeature[idx_macro]
			return self.getWeight(sess, I, I_macro)
		return f

	def construct2DNonlinearFunction(self, sess, idx_x, idx_y, idx_z=None, v_z=None, meanMacroFeature=None, stdMacroFeature=None):
		### require self._tSize = 1
		I_macro = np.zeros(shape=(1, self._macro_feature_dim))
		# I = np.ones(shape=(1, 1, self._individual_feature_dim)) * 0.5
		I = np.zeros(shape=(1, 1, self._individual_feature_dim))
		if idx_z and v_z:
			if idx_z < self._individual_feature_dim:
				I[0,0,idx_z] = v_z
			else:
				idx_macro = idx_z - self._individual_feature_dim
				I_macro[0, idx_macro] = (v_z - meanMacroFeature[idx_macro]) / stdMacroFeature[idx_macro]
		def f(x,y):
			if idx_x < self._individual_feature_dim:
				I[0,0,idx_x] = x
			else:
				idx_macro_x = idx_x - self._individual_feature_dim
				I_macro[0, idx_macro_x] = (x - meanMacroFeature[idx_macro_x]) / stdMacroFeature[idx_macro_x]
			if idx_y < self._individual_feature_dim:
				I[0,0,idx_y] = y
			else:
				idx_macro_y = idx_y - self._individual_feature_dim
				I_macro[0, idx_macro_y] = (y - meanMacroFeature[idx_macro_y]) / stdMacroFeature[idx_macro_y]
			return self.getWeight(sess, I, I_macro)
		return f

	def plotWeightWithData(self, sess, dl, plotPath=None, initial_state=None, sample=5000, normalized=False, figsize=(8,6)):
		w = self.getWeightWithData(sess, dl, initial_state=initial_state, normalized=normalized)
		idx_sample = np.arange(len(w))
		np.random.shuffle(idx_sample)
		idx_sample = idx_sample[:sample]
		deco_print('Please enter the idx for the plot you want! ')
		idx_plot = int(input('Please select from 1.1D with Data, 2.2D with Data: '))

		if idx_plot == 1:
			idx = int(input('Enter variable idx for x (0 - %d, %d - %d): ' \
				%(self._individual_feature_dim - 1, self._individual_feature_dim, self._individual_feature_dim + self._macro_feature_dim - 1)))
			xlabel = dl.getFeatureByIdx(idx)

			for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
				if idx < self._individual_feature_dim:
					x = I[:,:,idx][mask]
				else:
					idx_macro = idx - self._individual_feature_dim
					x = np.tile(I_macro[:,[idx_macro]], (1, mask.shape[1]))[mask]

			fig = plt.figure(figsize=figsize)
			plt.scatter(x[idx_sample], w[idx_sample], s=5, alpha=0.5)
			plt.xlabel(xlabel)
			plt.ylabel('weight')
			if plotPath:
				plt.savefig(os.path.join(plotPath, 'w_x_%d_with_data.pdf' %idx))
				plt.savefig(os.path.join(plotPath, 'w_x_%d_with_data.png' %idx))
			plt.show()

		if idx_plot == 2:
			idx_x = int(input('Enter variable idx for x (0 - %d, %d - %d): ' \
				%(self._individual_feature_dim - 1, self._individual_feature_dim, self._individual_feature_dim + self._macro_feature_dim - 1)))
			xlabel = dl.getFeatureByIdx(idx_x)
			idx_y = int(input('Enter variable idx for y (0 - %d, %d - %d): ' \
				%(self._individual_feature_dim - 1, self._individual_feature_dim, self._individual_feature_dim + self._macro_feature_dim - 1)))
			ylabel = dl.getFeatureByIdx(idx_y)

			for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
				if idx_x < self._individual_feature_dim:
					x = I[:,:,idx_x][mask]
				else:
					idx_macro_x = idx_x - self._individual_feature_dim
					x = np.tile(I_macro[:,[idx_macro_x]], (1, mask.shape[1]))[mask]
				if idx_y < self._individual_feature_dim:
					y = I[:,:,idx_y][mask]
				else:
					idx_macro_y = idx_y - self._individual_feature_dim
					y = np.tile(I_macro[:,[idx_macro_y]], (1, mask.shape[1]))[mask]

			fig = plt.figure(figsize=figsize)
			ax = fig.add_subplot(111, projection='3d')
			ax.scatter(x[idx_sample], y[idx_sample], w[idx_sample], s=5, alpha=0.5)
			ax.set_xlabel(xlabel)
			ax.set_ylabel(ylabel)
			ax.set_zlabel('weight')
			if plotPath:
				plt.savefig(os.path.join(plotPath, 'w_x_%d_y_%d_with_data.pdf' %(idx_x, idx_y)))
				plt.savefig(os.path.join(plotPath, 'w_x_%d_y_%d_with_data.png' %(idx_x, idx_y)))
			plt.show()

	def plotWeight(self, sess, dl, plotPath=None, sampleFreqPerAxis=50, figsize=(8,6), xlim=[-0.5,0.5]):
		### require self._tSize = 1
		meanMacroFeature, stdMacroFeature = dl.getMacroFeatureMeanStd()
		deco_print('Please enter the idx for the plot you want! ')
		idx_plot = int(input('Please select from 1.1D, 2.2D Contour 3.2D Contour Slice: '))

		if idx_plot == 1:
			idx = int(input('Enter variable idx for x (0 - %d, %d - %d): ' \
				%(self._individual_feature_dim - 1, self._individual_feature_dim, self._individual_feature_dim + self._macro_feature_dim - 1)))
			xlabel = dl.getFeatureByIdx(idx)
			# x_left = float(input('Enter variate lower bound for %d (%s): ' %(idx, xlabel)))
			# x_right = float(input('Enter variate upper bound for %d (%s): ' %(idx, xlabel)))

			wFun = self.construct1DNonlinearFunction(sess, idx, 
				meanMacroFeature=meanMacroFeature, 
				stdMacroFeature=stdMacroFeature)
			x = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1)
			v = np.zeros_like(x)
			for i in range(len(x)):
				v[i] = wFun(x[i])

			plotWeight1D(x, -v, xlabel, plotPath=plotPath, idx=idx, figsize=figsize)

		if idx_plot == 2:
			idx_x = int(input('Enter variable idx for x (0 - %d, %d - %d): ' \
				%(self._individual_feature_dim - 1, self._individual_feature_dim, self._individual_feature_dim + self._macro_feature_dim - 1)))
			xlabel = dl.getFeatureByIdx(idx_x)
			# x_left = float(input('Enter variate lower bound for %d (%s): ' %(idx_x, xlabel)))
			# x_right = float(input('Enter variate upper bound for %d (%s): ' %(idx_x, xlabel)))
			idx_y = int(input('Enter variable idx for y (0 - %d, %d - %d): ' \
				%(self._individual_feature_dim - 1, self._individual_feature_dim, self._individual_feature_dim + self._macro_feature_dim - 1)))
			ylabel = dl.getFeatureByIdx(idx_y)
			# y_left = float(input('Enter variate lower bound for %d (%s): ' %(idx_y, ylabel)))
			# y_right = float(input('Enter variate upper bound for %d (%s): ' %(idx_y, ylabel)))

			wFun = self.construct2DNonlinearFunction(sess, idx_x, idx_y, 
				meanMacroFeature=meanMacroFeature,
				stdMacroFeature=stdMacroFeature)
			x = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1)
			y = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1)
			x, y = np.meshgrid(x, y)
			v = np.zeros_like(x)
			for i in range(v.shape[0]):
				for j in range(v.shape[1]):
					v[i,j] = wFun(x[i,j], y[i,j])

			plotWeight2D(x, y, -v, xlabel, ylabel, plotPath=plotPath, idx_x=idx_x, idx_y=idx_y, figsize=figsize)

		if idx_plot == 3:
			idx_x = int(input('Enter variable idx for x (0 - %d, %d - %d): ' \
				%(self._individual_feature_dim - 1, self._individual_feature_dim, self._individual_feature_dim + self._macro_feature_dim - 1)))
			xlabel = dl.getFeatureByIdx(idx_x)
			# x_left = float(input('Enter variate lower bound for %d (%s): ' %(idx_x, xlabel)))
			# x_right = float(input('Enter variate upper bound for %d (%s): ' %(idx_x, xlabel)))
			idx_y = int(input('Enter variable idx for y (0 - %d, %d - %d): ' \
				%(self._individual_feature_dim - 1, self._individual_feature_dim, self._individual_feature_dim + self._macro_feature_dim - 1)))
			ylabel = dl.getFeatureByIdx(idx_y)
			# y_left = float(input('Enter variate lower bound for %d (%s): ' %(idx_y, ylabel)))
			# y_right = float(input('Enter variate upper bound for %d (%s): ' %(idx_y, ylabel)))
			idx_z = int(input('Enter variable idx for z (0 - %d, %d - %d): ' \
				%(self._individual_feature_dim - 1, self._individual_feature_dim, self._individual_feature_dim + self._macro_feature_dim - 1)))
			zlabel = dl.getFeatureByIdx(idx_z)
			# z_left = float(input('Enter variate lower bound for %d (%s): ' %(idx_z, zlabel)))
			# z_right = float(input('Enter variate upper bound for %d (%s): ' %(idx_z, zlabel)))

			x = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1)
			y = np.linspace(xlim[0], xlim[1], sampleFreqPerAxis + 1)
			zs = np.linspace(xlim[0], xlim[1], 4)
			v = np.zeros((len(y), len(x), len(zs)))
			x, y = np.meshgrid(x, y)

			for k in range(len(zs)):
				z = zs[k]
				wFun = self.construct2DNonlinearFunction(sess, idx_x, idx_y, idx_z=idx_z, v_z=z, 
					meanMacroFeature=meanMacroFeature,
					stdMacroFeature=stdMacroFeature)
				for i in range(len(y)):
					for j in range(len(x)):
						v[i,j,k] = wFun(x[i,j], y[i,j])

			plotWeight3D(x, y, -v, zs, xlabel, ylabel, zlabel, plotPath=plotPath, idx_x=idx_x, idx_y=idx_y, idx_z=idx_z, figsize=figsize)

	def _saveIndividualFeatureImportance(self, sess, dl, logdir, initial_state=None, delta=1e-6):
		w = self.getWeightWithData(sess, dl, initial_state=initial_state)
		gradients = np.zeros(shape=(self._individual_feature_dim))

		time_start = time.time()
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			for idx in range(self._individual_feature_dim):
				I_copy = copy.deepcopy(I)
				I_copy[mask, idx] += delta

				feed_dict = {self._I_macro_placeholder:I_macro,
							self._I_placeholder:I_copy,
							self._R_placeholder:R,
							self._mask_placeholder:mask,
							self._dropout_placeholder:1.0}
				if self.model_params['use_rnn']:
					feed_dict[self._initial_state_placeholder] = initial_state
				w_idx, = sess.run(fetches=[self._w], feed_dict=feed_dict)
				gradients[idx] = np.mean(np.absolute(w_idx - w))
				time_last = time.time() - time_start
				time_est = time_last / (idx+1) * self._individual_feature_dim
				deco_print('Calculating VI for %s\tElapse / Estimate: %.2fs / %.2fs' %(dl.getIndividualFeatureByIdx(idx), time_last, time_est))

		gradients /= delta
		deco_print('Saving output in %s' %os.path.join(logdir, 'ave_absolute_gradient.npy'))
		np.save(os.path.join(logdir, 'ave_absolute_gradient.npy'), gradients)

	def _saveMacroFeatureImportance(self, sess, dl, logdir, initial_state=None, delta=1e-06):
		w = self.getWeightWithData(sess, dl, initial_state=initial_state)
		gradients = np.zeros(shape=(self._macro_feature_dim))

		time_start = time.time()
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			for idx in range(self._macro_feature_dim):
				I_macro_copy = copy.deepcopy(I_macro)
				I_macro_copy[:, idx] += delta

				feed_dict = {self._I_macro_placeholder:I_macro_copy,
							self._I_placeholder:I,
							self._R_placeholder:R,
							self._mask_placeholder:mask,
							self._dropout_placeholder:1.0}
				if self.model_params['use_rnn']:
					feed_dict[self._initial_state_placeholder] = initial_state
				w_idx, = sess.run(fetches=[self._w], feed_dict=feed_dict)
				gradients[idx] = np.mean(np.absolute(w_idx - w))
				time_last = time.time() - time_start
				time_est = time_last / (idx+1) * self._macro_feature_dim
				deco_print('Calculating VI for %s\tElapse / Estimate: %.2fs / %.2fs' %(dl.getMacroFeatureByIdx(idx), time_last, time_est))

		gradients /= delta
		deco_print('Saving output in %s' %os.path.join(logdir, 'ave_absolute_gradient_macro.npy'))
		np.save(os.path.join(logdir, 'ave_absolute_gradient_macro.npy'), gradients)

	def plotIndividualFeatureImportance(self, sess, dl, logdir, plotPath=None, initial_state=None, top=30, figsize=(8,6)):
		if not os.path.exists(os.path.join(logdir, 'ave_absolute_gradient.npy')):
			self._saveIndividualFeatureImportance(sess, dl, logdir, initial_state=initial_state)
		gradients = np.load(os.path.join(logdir, 'ave_absolute_gradient.npy'))
		gradients_sorted = sorted([(idx, gradients[idx], dl.getIndividualFeatureByIdx(idx)) for idx in range(len(gradients))], key=lambda t:-t[1])
		imp = [item for _, item, _ in gradients_sorted]
		var = [item for _, _, item in gradients_sorted]
		var2color, color2category = dl.getIndividualFeatureColarLabelMap()
		labelColor = [var2color[item] for item in var]
		plot_variable_importance(var, imp, labelColor, plotPath=plotPath, top=top, figsize=figsize, color2category=color2category)

	def plotMacroFeatureImportance(self, sess, dl, logdir, plotPath=None, initial_state=None, top=30, figsize=(8,6)):
		if not os.path.exists(os.path.join(logdir, 'ave_absolute_gradient_macro.npy')):
			self._saveMacroFeatureImportance(sess, dl, logdir, initial_state=initial_state)
		gradients = np.load(os.path.join(logdir, 'ave_absolute_gradient_macro.npy'))
		gradients_sorted = sorted([(idx, gradients[idx], dl.getMacroFeatureByIdx(idx)) for idx in range(len(gradients))], key=lambda t:-t[1])
		imp = [item for _, item, _ in gradients_sorted]
		var = [item for _, _, item in gradients_sorted]
		labelColor = ['red'] * self._macro_feature_dim
		plot_variable_importance(var, imp, labelColor, plotPath=plotPath, top=top, figsize=figsize)

class NNModel_Ensembled:
	def __init__(self, logdirs, model_params, mode, tSize, force_var_reuse=False, global_step=None):
		self._logdirs = logdirs
		self._model = NNModel(model_params, mode, tSize, force_var_reuse=force_var_reuse, global_step=global_step)

	def getSDF(self, sess, dls):
		SDF = []
		for logdir, dl in zip(self._logdirs, dls):
			self._model.loadSavedModel(sess, logdir)
			SDF.append(self._model.getSDF(sess, dl))
		return np.array(SDF).mean(axis=0)

	def getSDFFactor(self, sess, dls):
		return 1 - self.getSDF(sess, dls)

	def getWeightWithData(self, sess, dls, normalized=False):
		w = []
		for logdir, dl in zip(self._logdirs, dls):
			self._model.loadSavedModel(sess, logdir)
			w.append(self._model.getWeightWithData(sess, dl, normalized=normalized))
		return np.array(w).mean(axis=0)

class NNModel(ModelBase):
	def __init__(self, model_params, mode, tSize, force_var_reuse=False, global_step=None):
		super(NNModel, self).__init__(model_params, mode, global_step)
		self._force_var_reuse = force_var_reuse
		self._tSize = tSize
		self._macro_feature_dim = self.model_params['macro_feature_dim']
		self._individual_feature_dim = self.model_params['individual_feature_dim']

		self._I_macro_placeholder = tf.placeholder(dtype=tf.float32, shape=[self._tSize, self._macro_feature_dim], name='macroFeaturePlaceholder')
		self._I_placeholder = tf.placeholder(dtype=tf.float32, shape=[self._tSize, None, self._individual_feature_dim], name='individualFeaturePlaceholder')
		self._R_placeholder = tf.placeholder(dtype=tf.float32, shape=[self._tSize, None], name='returnPlaceholder')
		self._mask_placeholder = tf.placeholder(dtype=tf.bool, shape=[self._tSize, None], name='maskPlaceholder')
		self._dropout_placeholder = tf.placeholder_with_default(1.0, shape=[], name='Dropout')

		self._nSize = tf.shape(self._R_placeholder)[1]

		with tf.variable_scope(name_or_scope='Model_Layer', reuse=self._force_var_reuse):
			self._build_forward_pass_graph()

	def _build_forward_pass_graph(self):
		with tf.variable_scope('NN_Layer'):
			I_macro_tile = tf.tile(tf.expand_dims(self._I_macro_placeholder, axis=1), [1,self._nSize,1]) # T * N * macro_feature_dim
			I_macro_masked = tf.boolean_mask(I_macro_tile, mask=self._mask_placeholder)
			I_masked = tf.boolean_mask(self._I_placeholder, mask=self._mask_placeholder)
			I_concat = tf.concat([I_masked, I_macro_masked], axis=1) # None * (macro_feature_dim + individual_feature_dim)
			R_masked = tf.boolean_mask(self._R_placeholder, mask=self._mask_placeholder)

			h_l = I_concat
			for l in range(self.model_params['num_layers']):
				with tf.variable_scope('dense_layer_%d' %l):
					layer_l = Dense(units=self.model_params['hidden_dim'][l], activation=tf.nn.relu)
					h_l = layer_l(h_l)
					h_l = tf.nn.dropout(h_l, self._dropout_placeholder)

			with tf.variable_scope('last_dense_layer'):
				layer = Dense(units=1)
				w = layer(h_l)
				self._w = tf.reshape(w, shape=[-1])

			weighted_R_masked = R_masked * self._w

		N_i = tf.reduce_sum(tf.to_int32(self._mask_placeholder), axis=1) # len T
		weighted_R_split = tf.split(weighted_R_masked, num_or_size_splits=N_i)
		self._SDF = tf.expand_dims(tf.concat([tf.reduce_sum(item, keepdims=True) for item in weighted_R_split], axis=0), axis=1) + 1

	def getSDF(self, sess, dl):
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			feed_dict = {self._I_macro_placeholder:I_macro,
						self._I_placeholder:I,
						self._R_placeholder:R,
						self._mask_placeholder:mask,
						self._dropout_placeholder:1.0}
			SDF, = sess.run(fetches=[self._SDF], feed_dict=feed_dict)
		return SDF

	def getWeightWithData(self, sess, dl, normalized=False):
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			feed_dict = {self._I_macro_placeholder:I_macro,
						self._I_placeholder:I,
						self._R_placeholder:R,
						self._mask_placeholder:mask,
						self._dropout_placeholder:1.0}
			w, = sess.run(fetches=[self._w], feed_dict=feed_dict)
		if normalized:
			w_list = np.split(w, np.sum(mask, axis=1).cumsum()[:-1])
			w = np.concatenate([item / np.absolute(item).sum() for item in w_list])
		return w

	def _saveAllFeatureImportance(self, sess, dl, logdir, delta=1e-6):
		w = self.getWeightWithData(sess, dl)
		gradients = np.zeros(shape=(self._individual_feature_dim+self._macro_feature_dim))

		time_start = time.time()
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			for idx in range(self._individual_feature_dim):
				I_copy = copy.deepcopy(I)
				I_copy[mask, idx] += delta

				feed_dict = {self._I_macro_placeholder:I_macro,
							self._I_placeholder:I_copy,
							self._R_placeholder:R,
							self._mask_placeholder:mask,
							self._dropout_placeholder:1.0}
				w_idx, = sess.run(fetches=[self._w], feed_dict=feed_dict)
				gradients[idx] = np.mean(np.absolute(w_idx - w))
				time_last = time.time() - time_start
				time_est = time_last / (idx+1) * (self._individual_feature_dim + self._macro_feature_dim)
				deco_print('Calculating VI for %s\tElapse / Estimate: %.2fs / %.2fs' %(dl.getIndividualFeatureByIdx(idx), time_last, time_est))
			for idx in range(self._macro_feature_dim):
				I_macro_copy = copy.deepcopy(I_macro)
				I_macro_copy[:,idx] += delta

				feed_dict = {self._I_macro_placeholder:I_macro_copy,
							self._I_placeholder:I,
							self._R_placeholder:R,
							self._mask_placeholder:mask,
							self._dropout_placeholder:1.0}
				w_idx, = sess.run(fetches=[self._w], feed_dict=feed_dict)
				gradients[self._individual_feature_dim+idx] = np.mean(np.absolute(w_idx - w))
				time_last = time.time() - time_start
				time_est = time_last / (self._individual_feature_dim+idx+1) * (self._individual_feature_dim + self._macro_feature_dim)
				deco_print('Calculating VI for %s\tElapse / Estimate: %.2fs / %.2fs' %(dl.getMacroFeatureByIdx(idx), time_last, time_est))

		gradients /= delta
		deco_print('Saving output in %s' %os.path.join(logdir, 'ave_absolute_gradient_all.npy'))
		np.save(os.path.join(logdir, 'ave_absolute_gradient_all.npy'), gradients)

	def plotIndividualFeatureImportance(self, sess, dl, logdir, plotPath=None, top=30, figsize=(8,6)):
		if not os.path.exists(os.path.join(logdir, 'ave_absolute_gradient_all.npy')):
			self._saveAllFeatureImportance(sess, dl, logdir)
		gradients = np.load(os.path.join(logdir, 'ave_absolute_gradient_all.npy'))
		gradients_sorted = sorted([(idx, gradients[idx], dl.getFeatureByIdx(idx)) for idx in range(len(gradients))], key=lambda t:-t[1])
		imp = [item for _, item, _ in gradients_sorted]
		var = [item for _, _, item in gradients_sorted]
		labelColor = ['red' if item < self._individual_feature_dim else 'green' for item, _, _ in gradients_sorted]
		plot_variable_importance(var, imp, labelColor, plotPath=plotPath, top=top, figsize=figsize)