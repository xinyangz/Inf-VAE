import math

import tensorflow as tf
from tensorflow.python.layers.core import Dense

flags = tf.compat.v1.flags
FLAGS = flags.FLAGS


class Model(object):

  def __init__(self, **kwargs):
    allowed_kwargs = {'name', 'logging'}
    for kwarg in kwargs.keys():
      assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
    name = kwargs.get('name')
    if not name:
      name = self.__class__.__name__.lower()
    self.name = name

    logging = kwargs.get('logging', False)
    self.logging = logging
    self.vars = {}
    self.loss = 0
    self.optimizer = None
    self.opt_op = None

  def _build(self):
    raise NotImplementedError

  def build(self):
    """ Wrapper for _build() """
    with tf.compat.v1.variable_scope(self.name):
      self._build()
    # Store model variables for easy access
    #variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
    #self.vars = {var.name: var for var in variables}

  def predict(self):
    pass

  def _loss(self):
    raise NotImplementedError

  def _accuracy(self):
    raise NotImplementedError

  def save(self, sess=None):
    if not sess:
      raise AttributeError("TensorFlow session not provided.")
    saver = tf.train.Saver(self.vars)
    save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
    print("Model saved in file: %s" % save_path)

  def load(self, sess=None):
    if not sess:
      raise AttributeError("TensorFlow session not provided.")
    saver = tf.train.Saver(self.vars)
    save_path = "tmp/%s.ckpt" % self.name
    saver.restore(sess, save_path)
    print("Model restored from file: %s" % save_path)
