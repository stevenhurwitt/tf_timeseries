from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

NUM_OUTPUT = 15

IMAGE_WIDTH = 195
IMAGE_HEIGHT = 20
IMAGE_PIXELS = IMAGE_WIDTH * IMAGE_HEIGHT

def inference(images, num_samples, keep_prob, reuse=None):

    with tf.variable_scope('conv1', reuse=reuse):
        kernel = tf.get_variable(name='weights', shape=[3, 30, 1, 5], initializer=tf.contrib.layers.xavier_initializer(uniform=False))        
        weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.001, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        conv = tf.nn.conv2d(images, kernel, [1, 1, 5, 1], padding='VALID')
        # output dim: 18x34
        biases = tf.Variable(tf.constant(0.0, name='biases', shape=[5]))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name='conv1')

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')    
    #output dim: 9x17

    with tf.variable_scope('conv2', reuse=reuse):
        kernel = tf.get_variable(name='weights', shape=[2, 2, 5, 5], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        weight_decay = tf.mul(tf.nn.l2_loss(kernel), 0.001, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='VALID')
        #output dim: 8x16
        biases = tf.Variable(tf.constant(0.1, name='biases', shape=[5]))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name='conv2')


    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')
    #output dim: 4x8

    h_fc1_drop = tf.nn.dropout(pool2, keep_prob)

    with tf.variable_scope('fully_connected', reuse=reuse):
        reshape = tf.reshape(h_fc1_drop, [num_samples, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable(name='weights', shape=[dim, 20], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        weight_decay = tf.mul(tf.nn.l2_loss(weights), 0.004, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        biases = tf.Variable(tf.zeros([20], name='biases'))
        fully_connected = tf.nn.relu(tf.matmul(reshape, weights) + biases, name='fully_connected')

    with tf.variable_scope('identity', reuse=reuse):
        weights = tf.get_variable(name='weights', shape=[20,NUM_OUTPUT], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        weight_decay = tf.mul(tf.nn.l2_loss(weights), 0.004, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
        biases = tf.Variable(tf.zeros([NUM_OUTPUT], name='biases'))
        output = tf.matmul(fully_connected, weights) + biases

    return output


def loss(outputs, labels):

    rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(labels, outputs))), name="rmse")
    loss_list = tf.get_collection('losses')
    loss_list.append(rmse)
    rmse_tot = tf.add_n(loss_list, name='total_loss')  
    return rmse_tot


def training(loss, starter_learning_rate, global_step):

      tf.scalar_summary(loss.op.name, loss)
#      optimizer = tf.train.AdamOptimizer()
      learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 200, 0.8, staircase=True)
      optimizer = tf.train.MomentumOptimizer(learning_rate, 0.8)
      grads_and_vars = optimizer.compute_gradients(loss)
      grad_norms = [tf.nn.l2_loss(g[0]) for g in grads_and_vars]      
      grad_norm = tf.add_n(grad_norms)
      train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
#      train_op = optimizer.minimize(loss, global_step=global_step)
      return train_op, grad_norm