import tensorflow as tf
import numpy as np
from param import batch_size, group_num
from tools import gauss

minimun_val = 1e-7

def clc_angle(A, B):

    A_dot_B = tf.reduce_sum(tf.multiply(A, B))
    norm_A = tf.norm(A, axis=[1,2])
    norm_B = tf.norm(A, axis=[1,2])
    norm_A_dot_B = tf.reduce_sum(tf.multiply(norm_A, norm_B))
    
    angle = tf.divide(A_dot_B, norm_A_dot_B + minimun_val)
    #angle = tf.acos(angle/batch_size)

    return angle

def max_raw_and_col(img):

    img_size = img.get_shape()[1]
    img_flat = tf.reshape(img, [batch_size, img_size*img_size, group_num])
    img_max_index = tf.argmax(img_flat, axis=1)

    img_x_index = img_max_index % img_size
    img_y_index = img_max_index / img_size
    
    return  img_x_index, img_y_index

def dis_loss(net):

    # distance loss
    image_size = net.get_shape()[1]
    x_index, y_index = max_raw_and_col(net)

    x_index = tf.expand_dims(tf.expand_dims(x_index, 1), 1)
    y_index = tf.expand_dims(tf.expand_dims(y_index, 1), 1)
    x_max = tf.image.resize_images(x_index, (image_size, image_size))
    y_max = tf.image.resize_images(y_index, (image_size, image_size))

    x_temp = np.zeros([batch_size, image_size, image_size, group_num])   
    y_temp = np.zeros([batch_size, image_size, image_size, group_num])   
    for index in range(image_size):
      x_temp[:,:,index,:] = index
      y_temp[:,index,:,:] = index
    x_p = tf.constant(x_temp, dtype=tf.float32)
    y_p = tf.constant(y_temp, dtype=tf.float32)

    x_y_diff = tf.cast(tf.pow(tf.subtract(x_max, x_p), 2) + tf.pow(tf.subtract(y_max, y_p), 2), tf.float32)
    #x_y_diff = tf.subtract(tf.maximum(x_y_diff,100.0), 100.0)
    dis_sum = tf.reduce_sum(tf.multiply(net, x_y_diff), [0,1,2,3])
    dis_sum = tf.divide(dis_sum, tf.reduce_sum(net) + minimun_val)
    dis_sum = tf.divide(dis_sum, batch_size)

    return dis_sum

def div_loss(net):

    # div loss
    margin = tf.constant(2e-4, shape=[1])
    list_0 = [tf.expand_dims(net[:,:,:,1], axis=3), tf.expand_dims(net[:,:,:,2], axis=3), tf.expand_dims(net[:,:,:,3], axis=3)]
    list_1 = [tf.expand_dims(net[:,:,:,0], axis=3), tf.expand_dims(net[:,:,:,2], axis=3), tf.expand_dims(net[:,:,:,3], axis=3)]
    list_2 = [tf.expand_dims(net[:,:,:,0], axis=3), tf.expand_dims(net[:,:,:,1], axis=3), tf.expand_dims(net[:,:,:,3], axis=3)]
    list_3 = [tf.expand_dims(net[:,:,:,0], axis=3), tf.expand_dims(net[:,:,:,1], axis=3), tf.expand_dims(net[:,:,:,2], axis=3)]
    diff0 = tf.subtract(tf.reduce_max(tf.concat(list_0, 3), axis=3), margin)
    diff1 = tf.subtract(tf.reduce_max(tf.concat(list_1, 3), axis=3), margin)
    diff2 = tf.subtract(tf.reduce_max(tf.concat(list_2, 3), axis=3), margin)
    diff3 = tf.subtract(tf.reduce_max(tf.concat(list_3, 3), axis=3), margin)

    div_temp = tf.add(clc_angle(net[:,:,:,0], diff0), tf.add(clc_angle(net[:,:,:,1], diff1), tf.add(clc_angle(net[:,:,:,2], diff2), clc_angle(net[:,:,:,3], diff3))))
    div_sum = tf.multiply(div_temp, 1)

    return div_sum

def bind_loss(net):

    dis_sum = dis_loss(net)
    
    div_sum = div_loss(net)

    d_loss = dis_sum + div_sum 

    tf.summary.scalar('dis_sum', dis_sum)   #TODO
    tf.summary.scalar('div_sum', div_sum)   #TODO

    return d_loss

def similar_loss(net, img):

    img_shape = net.get_shape()[1]
    img = tf.expand_dims(img, axis=3)
    img = tf.image.resize_images(img, (img_shape, img_shape))
    net = tf.reduce_sum(net, axis=3, keep_dims=True)

    sm_loss = clc_angle(net, img)

    return sm_loss

def get_constraint_loss(end_points):

    bind_net = tf.concat([end_points['group_0'], end_points['group_1'], end_points['group_2'], end_points['group_3']], axis=3)
    b1_loss = bind_loss(bind_net)

    b_loss_list = list()
    sm_loss_list = list()
    net = end_points['by_pass']
    quart_num = net.get_shape().as_list()[3] / 4
    for i in range(group_num):
        temp_quart = net[:,:,:,i*quart_num:(i+1)*quart_num]
        quart_num_quart = quart_num / 4
        quart_list = list()
        for j  in range(group_num):
            q_to_q_net = temp_quart[:,:,:,j*quart_num_quart:(j+1)*quart_num_quart]
            q_to_q_net = tf.reduce_mean(q_to_q_net, axis=3, keep_dims=True)
            #q_to_q_net = gauss(q_to_q_net, layer_name ='gauss_4a_' + str(i) + str(j))
            quart_list.append(q_to_q_net)
        quart_net = tf.concat(quart_list, axis=3)
        sm_loss_list.append(similar_loss(quart_net, bind_net[:,:,:,i]))
        b_loss_list.append(bind_loss(quart_net))
    sm_loss = tf.reduce_sum(tf.stack(sm_loss_list, axis=0), axis=0)
    sm_loss = tf.cast(sm_loss, dtype=tf.float32)
    sm_loss = tf.multiply(sm_loss, 2e5)
    b2_loss = tf.reduce_sum(tf.stack(b_loss_list, axis=0), axis=0)

    constraint_loss = b1_loss  
    #constraint_loss = -tf.divide(sm_loss, 1e5)

    tf.summary.scalar('b1_loss',b1_loss)
    tf.summary.scalar('b2_loss',b2_loss)
    tf.summary.scalar('sm_loss',sm_loss)

    return constraint_loss

'''
    image_size = net.get_shape()[1]
    net_flat = tf.reshape(net, [batch_size, image_size*image_size, group_num])
    max_index = tf.argmax(net_flat, axis=1)

    x_index = max_index % image_size
    y_index = max_index / image_size
'''



