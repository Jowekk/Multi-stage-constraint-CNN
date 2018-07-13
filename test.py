import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from load_data.load_data import get_split, load_batch
from network.inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2
from loss import dis_loss
from param import *


import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
slim = tf.contrib.slim

if not os.path.exists(log_dir):
    os.mkdir(log_dir)

with tf.Graph().as_default() as graph:
    tf.logging.set_verbosity(tf.logging.INFO)

    dataset = get_split('train', dataset_dir, file_pattern)
    images, _, labels = load_batch(dataset, batch_size=batch_size, height=image_size, width=image_size)

    with slim.arg_scope(inception_resnet_v2_arg_scope()):
        end_points = inception_resnet_v2(images, num_classes=dataset.num_classes, is_training=True)
        net = tf.concat([end_points['group_0'], end_points['group_1'], end_points['group_2'], end_points['group_3']], axis=3)
    
    d_loss = dis_loss(net)

    variables_to_restore = slim.get_variables_to_restore()

    saver = tf.train.Saver(variables_to_restore)
    def restore_fn(sess):
        return saver.restore(sess, '../data/MSCNN/log/model.ckpt-10727')

    sv = tf.train.Supervisor(logdir = None, summary_op = None, init_fn = restore_fn)

    with sv.managed_session() as sess:
        raw_images, imgs_matrix = sess.run([images, d_loss])

    print imgs_matrix.shape, imgs_matrix
