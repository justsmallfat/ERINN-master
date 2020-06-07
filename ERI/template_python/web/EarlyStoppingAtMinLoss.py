import numpy as np
import tensorflow as tf
from erinn.python.utils.io_utils import read_config_file


class EarlyStoppingAtMinLoss(tf.keras.callbacks.Callback):
  rootConfigPath = ''
  def __init__(self, configPath, patience=0):
    super(EarlyStoppingAtMinLoss, self).__init__()
    self.patience = patience
    self.rootConfigPath = configPath

    # best_weights to store the weights at which the minimum loss occurs.
    self.best_weights = None

  def on_train_begin(self, logs=None):
    # The number of epoch it has waited when loss is no longer minimum.
    print('EarlyStoppingAtMinLoss on_train_begin')
    self.wait = 0
    # The epoch the training stops at.
    self.stopped_epoch = 0
    # Initialize the best as infinity.
    self.best = np.Inf

  def on_epoch_end(self, epoch, logs=None):
    print(f'EarlyStoppingAtMinLoss on_epoch_end {self.rootConfigPath}')
    current = logs.get('loss')
    config = read_config_file(self.rootConfigPath)
    result = config['trainingStop']
    print(f'on_epoch_end result : {result}')
    if 'true' == result:
      self.stopped_epoch = epoch
      self.model.stop_training = True
      print('Stop by user.')
    # if np.less(current, self.best):
    #   self.best = current
    #   self.wait = 0
    #   # Record the best weights if current results is better (less).
    #   self.best_weights = self.model.get_weights()
    #   print('np.less.')
    # else:
    #   self.wait += 1
    #   if self.wait >= self.patience:
    #     self.stopped_epoch = epoch
    #     self.model.stop_training = True
    #     print('Restoring model weights from the end of the best epoch.')
    #     self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0:
      print('EarlyStoppingAtMinLoss Epoch %05d: early stopping' % (self.stopped_epoch + 1))
