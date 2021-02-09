import copy
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.python.layers.core import Dense
from tensorflow.core.framework import summary_pb2

from .model_base import ModelBase
from .model_utils import getFactor
from .model_utils import calculateStatistics
from src.utils import deco_print
from src.utils import sharpe

class FeedForwardModelWithNA_Return_Ensembled:
	def __init__(self, logdirs, model_params, mode, force_var_reuse=False, global_step=None):
		self._logdirs = logdirs
		self._model = FeedForwardModelWithNA_Return(model_params, mode, force_var_reuse=force_var_reuse, global_step=global_step)

	def getPrediction(self, sess, dl):
		pred = []
		for logdir in self._logdirs:
			self._model.loadSavedModel(sess, logdir)
			pred.append(self._model.getPrediction(sess, dl))
		return np.array(pred).mean(axis=0)

	def getSDFFactor(self, sess, dl, normalized=False, norm=None):
		beta = self.getPrediction(sess, dl)
		F = getFactor(beta, dl, normalized=normalized, norm=norm)
		return F

	def calculateStatistics(self, sess, dl):
		w = self.getPrediction(sess, dl)
		return calculateStatistics(w, dl)

	def evaluate_sharpe(self, sess, dl):
		R_pred = self.getPrediction(sess, dl)
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			portfolio = construct_long_short_portfolio(R_pred, R[mask], mask)
		return sharpe(portfolio)

class FeedForwardModelWithNA_Return(ModelBase):
	def __init__(self, model_params, mode, force_var_reuse=False, global_step=None):
		super(FeedForwardModelWithNA_Return, self).__init__(model_params, mode, global_step)
		self._force_var_reuse = force_var_reuse
		self._macro_feature_dim = self.model_params['macro_feature_dim']
		self._individual_feature_dim = self.model_params['individual_feature_dim']

		self._I_macro_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, self._macro_feature_dim], name='macroFeaturePlaceholder')
		self._I_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None, self._individual_feature_dim], name='individualFeaturePlaceholder')
		self._R_placeholder = tf.placeholder(dtype=tf.float32, shape=[None, None], name='returnPlaceholder')
		self._mask_placeholder = tf.placeholder(dtype=tf.bool, shape=[None, None], name='maskPlaceholder')
		self._dropout_placeholder = tf.placeholder_with_default(1.0, shape=[], name='Dropout')

		if self.model_params['weighted_loss']:
			self._loss_weight = tf.placeholder(dtype=tf.float32, shape=[None, None], name='weightPlaceholder')

		with tf.variable_scope(name_or_scope='Model_Layer', reuse=self._force_var_reuse):
			self._build_forward_pass_graph()
		if self._mode == 'train':
			self._train_model_op = self._build_train_op(self._loss, scope='Model_Layer')

	def _build_forward_pass_graph(self):
		with tf.variable_scope('NN_Layer'):
			NSize = tf.shape(self._R_placeholder)[1]
			I_macro_tile = tf.tile(tf.expand_dims(self._I_macro_placeholder, axis=1), [1,NSize,1])
			I_macro_masked = tf.boolean_mask(I_macro_tile, mask=self._mask_placeholder)
			I_masked = tf.boolean_mask(self._I_placeholder, mask=self._mask_placeholder)
			I_concat = tf.concat([I_masked, I_macro_masked], axis=1)
			R_masked = tf.boolean_mask(self._R_placeholder, mask=self._mask_placeholder)

			h_l = I_concat
			for l in range(self.model_params['num_layers']):
				with tf.variable_scope('dense_layer_%d' %l):
					layer_l = Dense(units=self.model_params['hidden_dim'][l], activation=tf.nn.relu)
					h_l = layer_l(h_l)
					h_l = tf.nn.dropout(h_l, self._dropout_placeholder)

			with tf.variable_scope('last_dense_layer'):
				layer = Dense(units=1)
				R_pred = layer(h_l)
				self._R_pred = tf.reshape(R_pred, shape=[-1])

		if self.model_params['weighted_loss']:
			loss_weight_masked = tf.boolean_mask(self._loss_weight, mask=self._mask_placeholder)
			loss_weight_masked /= tf.reduce_sum(loss_weight_masked) # normalize weight
			self._loss = tf.reduce_sum(tf.square(R_masked - self._R_pred) * loss_weight_masked)
		else:
			self._loss = tf.reduce_mean(tf.square(R_masked - self._R_pred))

	def train(self, sess, dl, dl_valid, logdir, loss_weight=None, loss_weight_valid=None, 
			dl_test=None, loss_weight_test=None, 
			printOnConsole=True, printFreq=128, saveLog=True):
		saver = tf.train.Saver(max_to_keep=100)
		if saveLog:
			sw = tf.summary.FileWriter(logdir, sess.graph)

		best_valid_loss = float('inf')
		sharpe_train = []
		sharpe_valid = []
		### evaluate test data
		evaluate_test_data = False
		if dl_test is not None:
			evaluate_test_data = True
			sharpe_test = []

		time_start = time.time()
		for epoch in range(self.model_params['num_epochs']):
			for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=self.model_params['sub_epoch'])):
				fetches = [self._train_model_op]
				feed_dict = {self._I_macro_placeholder:I_macro,
							self._I_placeholder:I,
							self._R_placeholder:R,
							self._mask_placeholder:mask,
							self._dropout_placeholder:self.model_params['dropout']}
				if self.model_params['weighted_loss']:
					feed_dict[self._loss_weight] = loss_weight
				sess.run(fetches=fetches, feed_dict=feed_dict)

			### evaluate train loss / sharpe
			train_epoch_loss = self.evaluate_loss(sess, dl, loss_weight)
			train_epoch_sharpe = self.evaluate_sharpe(sess, dl)
			sharpe_train.append(train_epoch_sharpe)

			### evaluate valid loss / sharpe
			valid_epoch_loss = self.evaluate_loss(sess, dl_valid, loss_weight_valid)
			valid_epoch_sharpe = self.evaluate_sharpe(sess, dl_valid)
			sharpe_valid.append(valid_epoch_sharpe)

			### evaluate test loss / sharpe
			if evaluate_test_data:
				test_epoch_loss = self.evaluate_loss(sess, dl_test, loss_weight_test)
				test_epoch_sharpe = self.evaluate_sharpe(sess, dl_test)
				sharpe_test.append(test_epoch_sharpe)

			### print loss / sharpe
			if printOnConsole and epoch % printFreq == 0:
				print('\n\n')
				deco_print('Doing epoch %d' %epoch)
				if evaluate_test_data:
					deco_print('Epoch %d train/valid/test loss: %0.4f/%0.4f/%0.4f' %(epoch, train_epoch_loss, valid_epoch_loss, test_epoch_loss))
					deco_print('Epoch %d train/valid/test sharpe: %0.4f/%0.4f/%0.4f' %(epoch, train_epoch_sharpe, valid_epoch_sharpe, test_epoch_sharpe))
				else:
					deco_print('Epoch %d train/valid loss: %0.4f/%0.4f' %(epoch, train_epoch_loss, valid_epoch_loss))
					deco_print('Epoch %d train/valid sharpe: %0.4f/%0.4f' %(epoch, train_epoch_sharpe, valid_epoch_sharpe))
			if saveLog:
				value_loss_train = summary_pb2.Summary.Value(tag='Train_epoch_loss', simple_value=train_epoch_loss)
				value_loss_valid = summary_pb2.Summary.Value(tag='Valid_epoch_loss', simple_value=valid_epoch_loss)
				value_sharpe_train = summary_pb2.Summary.Value(tag='Train_epoch_sharpe', simple_value=train_epoch_sharpe)
				value_sharpe_valid = summary_pb2.Summary.Value(tag='Valid_epoch_sharpe', simple_value=valid_epoch_sharpe)
				if evaluate_test_data:
					value_loss_test = summary_pb2.Summary.Value(tag='Test_epoch_loss', simple_value=test_epoch_loss)
					value_sharpe_test = summary_pb2.Summary.Value(tag='Test_epoch_sharpe', simple_value=test_epoch_sharpe)
					summary = summary_pb2.Summary(value=[value_loss_train, value_loss_valid, value_loss_test, value_sharpe_train, value_sharpe_valid, value_sharpe_test])
				else:
					summary = summary_pb2.Summary(value=[value_loss_train, value_loss_valid, value_sharpe_train, value_sharpe_valid])
				sw.add_summary(summary, global_step=epoch)
				sw.flush()

			### save epoch
			if valid_epoch_loss < best_valid_loss:
				best_valid_loss = valid_epoch_loss
				if printOnConsole and epoch % printFreq == 0:
					deco_print('Saving current best checkpoint')
				saver.save(sess, save_path=os.path.join(logdir, 'model-best'))

			### time
			if printOnConsole and epoch % printFreq == 0:
				time_elapse = time.time() - time_start
				time_est = time_elapse / (epoch+1) * self.model_params['num_epochs']
				deco_print('Epoch %d Elapse/Estimate: %0.2fs/%0.2fs' %(epoch, time_elapse, time_est))
		if evaluate_test_data:
			return sharpe_train, sharpe_valid, sharpe_test
		else:
			return sharpe_train, sharpe_valid

	def evaluate_loss(self, sess, dl, loss_weight=None):
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			feed_dict = {self._I_macro_placeholder:I_macro,
						self._I_placeholder:I,
						self._R_placeholder:R,
						self._mask_placeholder:mask,
						self._dropout_placeholder:1.0}
			if self.model_params['weighted_loss']:
				feed_dict[self._loss_weight] = loss_weight
			loss, = sess.run([self._loss], feed_dict=feed_dict)
		return loss

	def evaluate_sharpe(self, sess, dl):
		R_pred = self.getPrediction(sess, dl)
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			portfolio = construct_long_short_portfolio(R_pred, R[mask], mask) # equally weighted
		return sharpe(portfolio)

	def getPrediction(self, sess, dl):
		for _, (I_macro, I, R, mask) in enumerate(dl.iterateOneEpoch(subEpoch=False)):
			feed_dict = {self._I_macro_placeholder:I_macro,
						self._I_placeholder:I,
						self._R_placeholder:R,
						self._mask_placeholder:mask,
						self._dropout_placeholder:1.0}
			R_pred, = sess.run(fetches=[self._R_pred], feed_dict=feed_dict)
		return R_pred

	def calculateStatistics(self, sess, dl):
		w = self.getPrediction(sess, dl)
		return calculateStatistics(w, dl)

	def _saveIndividualFeatureImportance(self, sess, dl, logdir, delta=1e-6):
		R_pred = self.getPrediction(sess, dl)
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
				R_pred_idx, = sess.run(fetches=[self._R_pred], feed_dict=feed_dict)
				gradients[idx] = np.mean(np.absolute(R_pred_idx - R_pred))
				time_last = time.time() - time_start
				time_est = time_last / (idx+1) * self._individual_feature_dim
				deco_print('Calculating VI for %s\tElapse / Estimate: %.2fs / %.2fs' %(dl.getIndividualFeatureByIdx(idx), time_last, time_est))

		gradients /= delta
		deco_print('Saving output in %s' %os.path.join(logdir, 'ave_absolute_gradient.npy'))
		np.save(os.path.join(logdir, 'ave_absolute_gradient.npy'), gradients)