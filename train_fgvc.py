import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
from load_data.load_data import get_split, load_batch
from network.inception_resnet_v2 import inception_resnet_v2_arg_scope, inception_resnet_v2
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

    variables_to_restore = slim.get_variables_to_restore(exclude = exclude_list)

    accuracy_0, accuracy_update_0 = tf.contrib.metrics.streaming_accuracy(tf.argmax(end_points['Predictions_0'], 1), labels)
    metrics_op_0 = tf.group(accuracy_update_0, end_points['Predictions_0'])

    accuracy_1, accuracy_update_1 = tf.contrib.metrics.streaming_accuracy(tf.argmax(end_points['Predictions_1'], 1), labels)
    metrics_op_1 = tf.group(accuracy_update_1, end_points['Predictions_1'])

    accuracy_2, accuracy_update_2 = tf.contrib.metrics.streaming_accuracy(tf.argmax(end_points['Predictions_2'], 1), labels)
    metrics_op_2 = tf.group(accuracy_update_2, end_points['Predictions_2'])

    accuracy_3, accuracy_update_3 = tf.contrib.metrics.streaming_accuracy(tf.argmax(end_points['Predictions_3'], 1), labels)
    metrics_op_3 = tf.group(accuracy_update_3, end_points['Predictions_3'])

    one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

# TODO
    loss_0 = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = end_points['Logits_0'])
    loss_1 = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = end_points['Logits_1'])
    loss_2 = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = end_points['Logits_2'])
    loss_3 = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = end_points['Logits_3'])

    loss = loss_0 + loss_1 + loss_2 + loss_3

    dif_loss = end_points['dif_loss']
    dis_loss = end_points['dis_loss']

    limit_loss = tf.add(dif_loss, dis_loss)

    global_step = get_or_create_global_step()

    num_batches_per_epoch = int(train_img_num / batch_size)
    num_steps_per_epoch = num_batches_per_epoch

    decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)


    lr = tf.train.exponential_decay(
            learning_rate = initial_learning_rate,
            global_step = global_step,
            decay_steps = decay_steps,
            decay_rate = learning_rate_decay_factor,
            staircase = True)

    tf.summary.scalar('learning_rate', lr)
    tf.summary.scalar('dif_loss', dif_loss)
    tf.summary.scalar('dis_loss', dis_loss)
    tf.summary.scalar('loss_0', loss_0)
    tf.summary.scalar('loss_1', loss_1)
    tf.summary.scalar('loss_2', loss_2)
    tf.summary.scalar('loss_3', loss_3)
    tf.summary.scalar('accuracy_0', accuracy_0)
    tf.summary.scalar('accuracy_0', accuracy_0)
    tf.summary.scalar('accuracy_0', accuracy_0)
    tf.summary.scalar('accuracy_0', accuracy_0)
    my_summary_op = tf.summary.merge_all()

    all_vars =  tf.trainable_variables()
    first_train_vars = [var for var in all_vars if not var.name.startswith(mask_vars)]
    second_train_vars = [var for var in all_vars if var.name.startswith(mask_vars)]


    optimizer_1 = tf.train.AdamOptimizer(learning_rate = 5e-4)
    train_op_1 = slim.learning.create_train_op(loss, optimizer_1, variables_to_train = first_train_vars)

    optimizer_2 = tf.train.AdamOptimizer(learning_rate = lr)
    train_op_2 = slim.learning.create_train_op(limit_loss, optimizer_2 , variables_to_train = second_train_vars)

    def train_step(sess, train_op, global_step, flag):
        start_time = time.time()
        total_loss, global_step_count, _,_,_,_ = sess.run([train_op, global_step, metrics_op_0, metrics_op_1, metrics_op_2, metrics_op_3])
        time_elapsed = time.time() - start_time

        if flag == 1:
            loss_flag = 'MLP'
        else:
            loss_flag = 'limit'
        logging.info('global step %s: %s loss: %.4f (%.2f sec/step)', global_step_count, loss_flag, total_loss, time_elapsed)
        return total_loss, global_step_count


    saver = tf.train.Saver(variables_to_restore)
    def restore_fn(sess):
        return saver.restore(sess, checkpoint_file)

    sv = tf.train.Supervisor(logdir = log_dir, summary_op = None, init_fn = restore_fn)

    with sv.managed_session() as sess:
        global_step_ = sess.run(sv.global_step) 

        for step in xrange(global_step_, num_epochs):
            if step % 100 == 0:
                #loss, _ = train_step(sess, train_op, sv.global_step)
	        summaries = sess.run(my_summary_op)
	        sv.summary_computed(sess, summaries)

            if (step % 2000) < 1500:

                Loss_, _ = train_step(sess, train_op_1, sv.global_step, flag=1)
            else:
                Loss_, _ = train_step(sess, train_op_2, sv.global_step, flag=2)


        logging.info('Final Loss: %s', loss)
        logging.info('Finished training! Saving model to disk now.')
        sv.saver.save(sess, sv.save_path, global_step = sv.global_step)

