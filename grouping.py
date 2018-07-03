import tensorflow as tf
from tools import gauss
slim = tf.contrib.slim

def SE_layer(net, layer_name,dropout_keep_prob,  is_training):

  net_buffer = net
  net_shapes = net.get_shape().as_list()

  with tf.variable_scope('regroup'):

    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID', scope='AvgPool_1a_8x8' + layer_name)

    excitation = slim.fully_connected(net, int(net_shapes[3]/2.0), activation_fn=None, scope=layer_name+'_fully_connected1')
    excitation = tf.nn.relu(excitation, name = layer_name + '_relu')
    excitation = slim.dropout(excitation, dropout_keep_prob, is_training=is_training, scope='Dropout_1' + layer_name)
    excitation = slim.fully_connected(excitation, int(net_shapes[3]),  activation_fn=None, scope=layer_name+'_fully_connected2')
    excitation = slim.dropout(excitation, dropout_keep_prob, is_training=is_training, scope='Dropout_2' + layer_name)
    excitation = tf.nn.sigmoid(excitation, name = layer_name + '_sigmoid')

    excitation = tf.reshape(excitation, shape=(-1,1,1,int(net.get_shape()[3])))
    
    net = tf.multiply(net_buffer, excitation)

  return net

def logits_group(net, end_points, num_classes, dropout_keep_prob, is_training, layer_name):

  net = SE_layer(net, layer_name, dropout_keep_prob=dropout_keep_prob, is_training= is_training)
  net_temp = tf.reduce_mean(net,axis=3, keep_dims=True)
  end_points['group' + layer_name] = gauss(net_temp, layer_name='gauss_log'+layer_name)

  with tf.variable_scope('Logits'):
    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                        scope='AvgPool_1a_8x8' + layer_name)
    net = slim.flatten(net)

    net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                     scope='Dropout' + layer_name)

    logits = slim.fully_connected(net, num_classes, activation_fn=None,
                                scope='Logits' + layer_name)
    end_points['Logits'+layer_name] = logits
    end_points['Predictions'+layer_name] = tf.nn.softmax(logits, name='Predictions' + layer_name)

  return end_points


