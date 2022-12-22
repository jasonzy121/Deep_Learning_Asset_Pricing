import os
import json
import numpy as np
import tensorflow as tf

from src.data import data_layer
from src.model.model_RtnFcst import FeedForwardModelWithNA_Return
from src.utils import deco_print

tf.flags.DEFINE_string('config', '', 'Path to the file with configurations')
tf.flags.DEFINE_string('logdir', '', 'Path to save logs and checkpoints')
tf.flags.DEFINE_string('task_id', '', 'ID of task')

tf.flags.DEFINE_boolean('printOnConsole', True, 'Print on console or not')
tf.flags.DEFINE_boolean('saveLog', True, 'Save log or not')
tf.flags.DEFINE_integer('printFreq', 128, 'Frequency to print on console')
tf.flags.DEFINE_integer('trial_id', 0, 'ID of trials')

FLAGS = tf.flags.FLAGS

def main(_):
	with open(FLAGS.config + '/config_RF_' + FLAGS.task_id + '.json', 'r') as file:
		config = json.load(file)
	deco_print('Read the following in config: ')
	print(json.dumps(config, indent=4))

	deco_print('Creating data layer')
	dl = data_layer.DataInRamInputLayer(
		config['individual_feature_file'],
		pathMacroFeature=config['macro_feature_file'],
		macroIdx=config['macro_idx'])
	meanMacroFeature, stdMacroFeature = dl.getMacroFeatureMeanStd()
	dl_valid = data_layer.DataInRamInputLayer(
		config['individual_feature_file_valid'],
		pathMacroFeature=config['macro_feature_file_valid'],
		macroIdx=config['macro_idx'],
		meanMacroFeature=meanMacroFeature,
		stdMacroFeature=stdMacroFeature)
	dl_test = data_layer.DataInRamInputLayer(
		config['individual_feature_file_test'],
		pathMacroFeature=config['macro_feature_file_test'],
		macroIdx=config['macro_idx'],
		meanMacroFeature=meanMacroFeature,
		stdMacroFeature=stdMacroFeature)
	if config['weighted_loss']:
		loss_weight = dl.getDateCountList()
		loss_weight_valid = dl_valid.getDateCountList()
		loss_weight_test = dl_test.getDateCountList()
	else:
		loss_weight = None
		loss_weight_valid = None
		loss_weight_test = None
	deco_print('Data layer created')

	tf.reset_default_graph()
	global_step = tf.train.get_or_create_global_step()
	model = FeedForwardModelWithNA_Return(config, 'train', global_step=global_step)
	gpu_options = tf.GPUOptions(allow_growth=True)
	sess_config = tf.ConfigProto(gpu_options=gpu_options)
	sess_config.gpu_options.allow_growth = True
	sess = tf.Session(config=sess_config)
	model.randomInitialization(sess)

	logdir_trial = os.path.join(FLAGS.logdir, 'RF_%s_Trial_%d'%(FLAGS.task_id, FLAGS.trial_id))
	os.system('mkdir -p ' + logdir_trial)
	sharpe_train, sharpe_valid, sharpe_test = model.train(sess, dl, dl_valid, logdir_trial, 
		loss_weight=loss_weight, loss_weight_valid=loss_weight_valid, 
		dl_test=dl_test, loss_weight_test=loss_weight_test, 
		printOnConsole=FLAGS.printOnConsole, printFreq=FLAGS.printFreq, saveLog=FLAGS.saveLog)

	### best model on sharpe
	idxBestEpoch = np.array(sharpe_valid).argmax()
	sharpe_train_best_sharpe = sharpe_train[idxBestEpoch]
	sharpe_valid_best_sharpe = sharpe_valid[idxBestEpoch]
	sharpe_test_best_sharpe = sharpe_test[idxBestEpoch]
	deco_print('SDF Portfolio Sharpe Ratio (Evaluated on Sharpe): Train %0.3f\tValid %0.3f\tTest %0.3f' %(sharpe_train_best_sharpe, sharpe_valid_best_sharpe, sharpe_test_best_sharpe))

if __name__ == '__main__':
	tf.app.run()