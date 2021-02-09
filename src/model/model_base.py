import abc
import six
import tensorflow as tf

from src.utils import deco_print

six.add_metaclass(abc.ABCMeta)
class ModelBase:
	"""Abstract class that defines a model. 
	"""
	def __init__(self, model_params, mode, global_step=None):
		"""Initialize a model. 

		Arguments: 
			model_params: Parameters describing a model. 
			mode: Mode. 
			global_step: Global step. 
		"""
		self._model_params = model_params
		self._mode = mode
		self._global_step = global_step if global_step is not None else tf.contrib.framework.get_or_create_global_step()

	@abc.abstractmethod
	def _build_forward_pass_graph(self):
		"""Abstract method that describes how forward pass graph is constructed. 
		"""
		return

	def _build_train_op(self, loss, scope, loss_factor=1.0):
		"""Construct a training op. 

		Arguments:
			loss: Scalar 'Tensor'
		"""

		### Trainable variables
		deco_print('Trainable variables (scope=%s)' %scope)
		total_params = 0
		trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
		for var in trainable_variables:
			var_params = 1
			for dim in var.get_shape():
				var_params *= dim.value
			total_params += var_params
			print('Name: {} and shape: {}'.format(var.name, var.get_shape()))
		deco_print('Number of parameters: %d' %total_params)

		### Train optimizer
		if self._model_params['optimizer'] == 'Momentum':
			optimizer = lambda lr: tf.train.MomentumOptimizer(lr, momentum=0.9)
		elif self._model_params['optimizer'] == 'AdaDelta':
			optimizer = lambda lr: tf.train.AdadeltaOptimizer(lr, rho=0.95, epsilon=1e-08)
		else:
			optimizer = self._model_params['optimizer']

		### Learning rate decay
		if 'use_decay' in self._model_params and self._model_params['use_decay'] == True:
			learning_rate_decay_fn = lambda lr, global_step: tf.train.exponential_decay(
				learning_rate=lr,
				global_step=global_step,
				decay_steps=self._model_params['decay_steps'],
				decay_rate=self._model_params['decay_rate'],
				staircase=True)
		else:
			learning_rate_decay_fn = None

		return tf.contrib.layers.optimize_loss(
			loss=loss * loss_factor,
			global_step=self._global_step,
			learning_rate=self._model_params['learning_rate'],
			optimizer=optimizer,
			gradient_noise_scale=None,
			gradient_multipliers=None,
			clip_gradients=self._model_params['max_grad_norm'] if 'max_grad_norm' in self._model_params else None,
			learning_rate_decay_fn=learning_rate_decay_fn,
			update_ops=None,
			variables=trainable_variables,
			name=None,
			summaries=None,
			colocate_gradients_with_ops=True,
			increment_global_step=True)

	@property
	def model_params(self):
		"""
		Returns:
			Parameters used to construct the model. 
		"""
		return self._model_params

	def randomInitialization(self, sess):
		sess.run(tf.global_variables_initializer())
		deco_print('Random initialization')

	def loadSavedModel(self, sess, logdir):
		if tf.train.latest_checkpoint(logdir) is not None:
			saver = tf.train.Saver(max_to_keep=100)
			saver.restore(sess, tf.train.latest_checkpoint(logdir))
			deco_print('Restored checkpoint')
		else:
			deco_print('WARNING: Checkpoint not found! Use random initialization! ')
			self.randomInitialization(sess)