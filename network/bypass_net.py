import tensorflow as tf
slim = tf.contrib.slim

def bypass(inputs):

  with tf.variable_scope("bypass"):

    net = slim.conv2d(inputs, 32, 3, stride=2, padding='VALID', scope='Conv2d_1a_3x3') 
    net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_1a_3x3')
    net = slim.conv2d_transpose(net, 32, 3,stride=2, padding='VALID', scope='Deconv_1a_3x3')

    net = slim.conv2d(net, 32, 3, stride=1, padding='VALID', scope='Conv2d_2a_3x3')
    net = slim.max_pool2d(net, 2, stride=2, padding='VALID', scope='MaxPool_2a_2x2')
    net = slim.conv2d_transpose(net, 32, 3, stride=2, padding='VALID', scope='Deconv_2a_2x2')

    net = slim.conv2d(net, 64, 3, scope='Conv2d_2b_3x3')
    net = slim.max_pool2d(net, 3, stride=2, padding='VALID', scope='MaxPool_2b_3x3')

    net = slim.conv2d(net, 80, 1, padding='VALID', scope='Conv2d_3b_1x1')
    net = slim.max_pool2d(net, 2, stride=2, padding='VALID', scope='MaxPool_3b_2x2')
    net = slim.conv2d_transpose(net, 80, 3, stride=2, padding='VALID', scope='Deconv_3b_2x2')

    net = slim.conv2d(net, 192, 3, padding='VALID', scope='Conv2d_4a_3x3')

  return net


  
