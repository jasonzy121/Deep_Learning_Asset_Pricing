import pandas as pd
import numpy as np
import os

UNK = -99.99
task_id = 1

tmp_train = np.load('datasets/char/Char_train.npz')
tmp_valid = np.load('datasets/char/Char_valid.npz')
tmp_test = np.load('datasets/char/Char_test.npz')

log_dir = 'sample_checkpoints'

sdf_path = os.path.join(log_dir, 'SDF_normalized_ensemble.npy')
F = np.load(sdf_path)[:,0]

train_data = tmp_train['data']
for i in range(train_data.shape[1]):
    mask = train_data[:, i, 0] != UNK
    train_data[mask, i, 0] *= F[0:240][mask] * 50

valid_data = tmp_valid['data']
for i in range(valid_data.shape[1]):
    mask = valid_data[:, i, 0] != UNK
    valid_data[mask, i, 0] *= F[240:300][mask] * 50

test_data = tmp_test['data']
for i in range(test_data.shape[1]):
    mask = test_data[:, i, 0] != UNK
    test_data[mask, i, 0] *= F[300:][mask] * 50

path_train = os.path.join('datasets', 'RF', 'RF_train_normalized_task_%d.npz' % (task_id))
np.savez(path_train, date = tmp_train['date'], variable = tmp_train['variable'], data = train_data)

path_valid = os.path.join('datasets', 'RF', 'RF_valid_normalized_task_%d.npz' % (task_id))
np.savez(path_valid, date = tmp_valid['date'], variable = tmp_valid['variable'], data = valid_data)

path_test = os.path.join('datasets', 'RF', 'RF_test_normalized_task_%d.npz' % (task_id))
np.savez(path_test, date = tmp_test['date'], variable = tmp_test['variable'], data = test_data)

