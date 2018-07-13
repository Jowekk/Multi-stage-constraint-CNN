import tensorflow as tf
from tools import gauss
from param import group_num
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

def logits_group(net, end_points, dropout_keep_prob, is_training, layer_name):

  net = SE_layer(net, layer_name, dropout_keep_prob=dropout_keep_prob, is_training= is_training)
  net_temp = tf.reduce_mean(net,axis=3, keep_dims=True)
  end_points['group' + layer_name] = net_temp
  end_points['net' + layer_name] = net

  return end_points

def backend_group(net, end_points,  dropout_keep_prob, is_training):

  quart_num_log = int(net.get_shape().as_list()[3]/ 4.0)
  for i in range(group_num):
    end_points = logits_group(net[:,:,:,i*quart_num_log:(i+1)*quart_num_log], end_points, dropout_keep_prob, is_training, '_'+str(i))
  net = tf.concat([end_points['net_0'], end_points['net_1'], end_points['net_2'], end_points['net_3']], axis=3)

  return net

def frontend_group(net, dropout_keep_prob, is_training):

  net_list = list()
  quart_num = net.get_shape().as_list()[3] / 4
  for i in range(group_num):
    temp_quart = net[:,:,:,i*quart_num:(i+1)*quart_num]
    quart_num_quart = quart_num/ 4
    for j in range(group_num):
      q_to_q_net = temp_quart[:,:,:,j*quart_num_quart:(j+1)*quart_num_quart]
      q_to_q_net = SE_layer(q_to_q_net, layer_name='front_' + str(i) + str(j), dropout_keep_prob=dropout_keep_prob, is_training=is_training)
      net_list.append(q_to_q_net)
  
  net = tf.concat(net_list, axis=3)

  return net


