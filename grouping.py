import tensorflow as tf
slim = tf.contrib.slim


def group_function(net, end_points, num_classes, dropout_keep_prob, is_training, layer_name):

  with tf.variable_scope('group') :

    net = slim.avg_pool2d(net, net.get_shape()[1:3], padding='VALID',
                        scope='AvgPool_1a_8x8' + layer_name)

    excitation = slim.fully_connected(net, int(net.get_shape()[3]/2), activation_fn=None, scope=layer_name+'_fully_connected1')
    excitation = tf.nn.relu(excitation, name = layer_name + '_relu')
    excitation = slim.dropout(excitation, dropout_keep_prob, is_training=is_training, scope='Dropout_1' + layer_name)
    excitation = slim.fully_connected(excitation, int(net.get_shape()[3]),  activation_fn=None, scope=layer_name+'_fully_connected2')
    excitation = slim.dropout(excitation, dropout_keep_prob, is_training=is_training, scope='Dropout_2' + layer_name)
    excitation = tf.nn.sigmoid(excitation, name = layer_name + '_sigmoid')

    excitation = tf.reshape(excitation, shape=(-1,1,1,36))
    
    net = tf.multiply(net, excitation)
    end_points['group'] = net

  with tf.variable_scope('Log') :
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
