import math
import numpy as np
import tensorflow as tf

from tensorflow.python.framework import ops

from utils import *

import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import initializers

from functools import partial
import warnings

##  ------ SN GAN plug in ----------------------
## from : https://github.com/minhnhat93/tf-SNDCGAN/blob/master/libs/ops.py

# from libs.sn import spectral_normed_weight

def scope_has_variables(scope):
  return len(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope.name)) > 0


def sn_conv2d(input_, output_dim,
           k_h=4, k_w=4, d_h=2, d_w=2, stddev=None,
           name="conv2d", spectral_normed=False, update_collection=None, with_w=False, padding="SAME"):
  # Glorot intialization
  # For RELU nonlinearity, it's sqrt(2./(n_in)) instead
  fan_in = k_h * k_w * input_.get_shape().as_list()[-1]
  fan_out = k_h * k_w * output_dim
  if stddev is None:
    stddev = np.sqrt(2. / (fan_in))

  with tf.variable_scope(name) as scope:
    if scope_has_variables(scope):
      scope.reuse_variables()
    w = tf.get_variable("w", [k_h, k_w, input_.get_shape()[-1], output_dim],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    if spectral_normed:
      conv = tf.nn.conv2d(input_, spectral_normed_weight(w, update_collection=update_collection),
                          strides=[1, d_h, d_w, 1], padding=padding)
    else:
      conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding=padding)

    biases = tf.get_variable("b", [output_dim], initializer=tf.constant_initializer(0.0))
    conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    if with_w:
      return conv, w, biases
    else:
      return conv


def sn_deconv2d(input_, output_shape,
             k_h=4, k_w=4, d_h=2, d_w=2, stddev=None,
             name="deconv2d", spectral_normed=False, update_collection=None, with_w=False, padding="SAME"):
  # Glorot initialization
  # For RELU nonlinearity, it's sqrt(2./(n_in)) instead
  fan_in = k_h * k_w * input_.get_shape().as_list()[-1]
  fan_out = k_h * k_w * output_shape[-1]
  if stddev is None:
    stddev = np.sqrt(2. / (fan_in))

  with tf.variable_scope(name) as scope:
    if scope_has_variables(scope):
      scope.reuse_variables()
    # filter : [height, width, output_channels, in_channels]
    w = tf.get_variable("w", [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
    if spectral_normed:
      deconv = tf.nn.conv2d_transpose(input_, spectral_normed_weight(w, update_collection=update_collection),
                                      output_shape=output_shape,
                                      strides=[1, d_h, d_w, 1], padding=padding)
    else:
      deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                      strides=[1, d_h, d_w, 1], padding=padding)

    biases = tf.get_variable("b", [output_shape[-1]], initializer=tf.constant_initializer(0))
    deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
    if with_w:
      return deconv, w, biases
    else:
      return deconv


def sn_lrelu(x, leak=0.1):
  return tf.maximum(x, leak * x)


def sn_linear(input_, output_size, name="linear", spectral_normed=False, update_collection=None, stddev=None, bias_start=0.0, with_biases=True,
           with_w=False):
  shape = input_.get_shape().as_list()

  if stddev is None:
    stddev = np.sqrt(1. / (shape[1]))
  with tf.variable_scope(name) as scope:
    if scope_has_variables(scope):
      scope.reuse_variables()
    weight = tf.get_variable("w", [shape[1], output_size], tf.float32,
                             tf.truncated_normal_initializer(stddev=stddev))
    if with_biases:
      bias = tf.get_variable("b", [output_size],
                             initializer=tf.constant_initializer(bias_start))
    if spectral_normed:
      mul = tf.matmul(input_, spectral_normed_weight(weight, update_collection=update_collection))
    else:
      mul = tf.matmul(input_, weight)
    if with_w:
      if with_biases:
        return mul + bias, weight, bias
      else:
        return mul, weight, None
    else:
      if with_biases:
        return mul + bias
      else:
        return mul


def sn_batch_norm(input, is_training=True, momentum=0.9, epsilon=2e-5, in_place_update=True, name="batch_norm"):
  if in_place_update:
    return tf.contrib.layers.batch_norm(input,
                                        decay=momentum,
                                        center=True,
                                        scale=True,
                                        epsilon=epsilon,
                                        updates_collections=None,
                                        is_training=is_training,
                                        scope=name)
  else:
    return tf.contrib.layers.batch_norm(input,
                                        decay=momentum,
                                        center=True,
                                        scale=True,
                                        epsilon=epsilon,
                                        is_training=is_training,
                                        scope=name)

import warnings

NO_OPS = 'NO_OPS'

def _l2normalize(v, eps=1e-12):
  return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def spectral_normed_weight(W, u=None, num_iters=1, update_collection=None, with_sigma=False):
  # Usually num_iters = 1 will be enough
  W_shape = W.shape.as_list()
  W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
  if u is None:
    u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)
  def power_iteration(i, u_i, v_i):
    v_ip1 = _l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
    u_ip1 = _l2normalize(tf.matmul(v_ip1, W_reshaped))
    return i + 1, u_ip1, v_ip1
  _, u_final, v_final = tf.while_loop(
    cond=lambda i, _1, _2: i < num_iters,
    body=power_iteration,
    loop_vars=(tf.constant(0, dtype=tf.int32),
               u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
  )
  if update_collection is None:
    warnings.warn('Setting update_collection to None will make u being updated every W execution. This maybe undesirable'
                  '. Please consider using a update collection instead.')
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
    W_bar = W_reshaped / sigma
    with tf.control_dependencies([u.assign(u_final)]):
      W_bar = tf.reshape(W_bar, W_shape)
  else:
    sigma = tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final))[0, 0]
    # sigma = tf.reduce_sum(tf.matmul(u_final, tf.transpose(W_reshaped)) * v_final)
    W_bar = W_reshaped / sigma
    W_bar = tf.reshape(W_bar, W_shape)
    # Put NO_OPS to not update any collection. This is useful for the second call of discriminator if the update_op
    # has already been collected on the first call.
    if update_collection != NO_OPS:
      tf.add_to_collection(update_collection, u.assign(u_final))
  if with_sigma:
    return W_bar, sigma
  else:
    return W_bar


# dense net code comes from https://github.com/taki0112/Densenet-Tensorflow
def Batch_Normalization(x, training, scope):
        with arg_scope([batch_norm],
                       scope=scope,
                       updates_collections=None,
                       decay=0.9,
                       center=True,
                       scale=True,
                       zero_debias_moving_mean=True) :
            return tf.cond(training,
                           lambda : batch_norm(inputs=x, is_training=training, reuse=None),
                           lambda : batch_norm(inputs=x, is_training=training, reuse=True))