# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 16:25:44 2017

@author: Administrator
"""

# -*- coding: utf-8 -*-
import os
import numpy as np
import tensorflow as tf
import codecs
from PIL import Image
import pylab as plt
from sklearn.cross_validation import train_test_split   
import skimage.io as io
from skimage import data_dir

        
def get_files(filename):
    
     class_train = []
     label_train = []
     for train_class in os.listdir(filename):
         for pic in os.listdir(filename+train_class):
             class_train.append(filename+train_class+'/'+pic)
             label_train.append(train_class)
     temp = np.array([class_train,label_train])
     print(temp)
     temp = temp.transpose()#转置
     np.random.shuffle(temp)#打乱序列中的顺序
     image_list = list(temp[:,0])#去第一维中的所有数据，即取所有行中的第0个数据
     label_list = list(temp[:,1])#去第二维中的所有数据，即取所有行中的第1个数据
     label_list = [int(i) for i in label_list]
     print(label_list)
     return image_list,label_list#由此可见，但结果数据量很大时，输出列表，结果是完整的，输出数组结果是会隐藏大量信息
 

def get_batches(image,label,resize_w,resize_h,batch_size,capacity):
    #convert the list of images and labels to tensor
    image = tf.cast(image,tf.string)
    label = tf.cast(label,tf.int64)
    queue = tf.train.slice_input_producer([image,label])
    label = queue[1]
    image_c = tf.read_file(queue[0])
    image = tf.image.decode_jpeg(image_c,channels = 3)#解码JPEG格式图像
    image = tf.image.resize_image_with_crop_or_pad(image,resize_w,resize_h)#resize
    #(x - mean) / adjusted_stddev
    image = tf.image.per_image_standardization(image)
    image_batch,label_batch = tf.train.batch([image,label],batch_size = batch_size,num_threads = 64,capacity = capacity)
    images_batch = tf.cast(image_batch,tf.float32)
    labels_batch = tf.reshape(label_batch,[batch_size])
    print(image_batch)
    print(label_batch)
    return images_batch,labels_batch

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape,stddev = 0.01))
 #init weights
weights = {
      "w1":init_weights([3,3,3,16]),
      "w2":init_weights([3,3,16,128]),
      "w3":init_weights([3,3,128,256]),
      "w4":init_weights([4096,4096]),
      "wo":init_weights([4096,6])
          }
 
 #init biases
biases = {
     "b1":init_weights([16]),
     "b2":init_weights([128]),
     "b3":init_weights([256]),
     "b4":init_weights([4096]),
     "bo":init_weights([6])
         }

def conv2d(x,w,b):
    x = tf.nn.conv2d(x,w,strides = [1,1,1,1],padding = "SAME")
    x = tf.nn.bias_add(x,b)
    return tf.nn.relu(x)

def pooling(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = "SAME")
 
def norm(x,lsize = 4):
    return tf.nn.lrn(x,depth_radius = lsize,bias = 1,alpha = 0.001/9.0,beta = 0.75)

def mmodel(images):
    print("begin to build model")
    l1 = conv2d(images,weights["w1"],biases["b1"])
    l2 = pooling(l1)
    l2 = norm(l2)
    l3 = conv2d(l2,weights["w2"],biases["b2"])
    l4 = pooling(l3)
    l4 = norm(l4)
    l5 = conv2d(l4,weights["w3"],biases["b3"])
    #same as the batch size
    l6 = pooling(l5)
    l6 = tf.reshape(l6,[-1,weights["w4"].get_shape().as_list()[0]])
    l7 = tf.nn.relu(tf.matmul(l6,weights["w4"])+biases["b4"])
    soft_max = tf.add(tf.matmul(l7,weights["wo"]),biases["bo"])
    return soft_max
def loss(logits,label_batches):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=label_batches)
    cost = tf.reduce_mean(cross_entropy)
    return cost
def get_accuracy(logits,labels):
    acc = tf.nn.in_top_k(logits,labels,1)
    acc = tf.cast(acc,tf.float32)
    acc = tf.reduce_mean(acc)
    return acc
def training(loss,lr):
    train_op = tf.train.RMSPropOptimizer(lr,0.9).minimize(loss)
    return train_op
def get_one_image(img_dir):
    
    image = Image.open(img_dir)
    plt.imshow(image)
    image = image.resize([32, 32])
    image_arr = np.array(image)
    return image_arr
def test(test_file):
    
    log_dir = "/home/lab326/songpeng/log/"
    image_arr = get_one_image(test_file)
    
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32,shape = [32,32,3])
        image = tf.cast(x, tf.float32)
        image = tf.image.per_image_standardization(image)
        image = tf.reshape(image, [1,32, 32, 3])
        print(image.shape)
        p = mmodel1(image,1)
        logits = tf.nn.softmax(p)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(log_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Loading success')
            else:
                print('No checkpoint')
            prediction = sess.run(logits, feed_dict={x: image_arr})
            max_index = np.argmax(prediction)
            print(max_index)
def mmodel1(images,batch_size):
    
    with tf.variable_scope('conv1') as scope:
        weights = tf.get_variable('weights', 
                                  shape = [3,3,3, 16],
                                  dtype = tf.float32, 
                                  initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                  shape=[16],
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1,1,1,1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name= scope.name)
    with tf.variable_scope('pooling1_lrn') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1,2,2,1],strides=[1,2,2,1],
                                padding='SAME', name='pooling1')
        norm1 = tf.nn.lrn(pool1, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                           beta=0.75,name='norm1')
    with tf.variable_scope('conv2') as scope:
        weights = tf.get_variable('weights',
                                   shape=[3,3,16,128],
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                  shape=[128], 
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm1, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')    
    with tf.variable_scope('pooling2_lrn') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,1,1,1],
                                padding='SAME',name='pooling2')
        norm2 = tf.nn.lrn(pool2, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                           beta=0.75,name='norm2')
       
    with tf.variable_scope('conv3') as scope:
        weights = tf.get_variable('weights',
                                   shape=[3,3,128,256],
                                   dtype=tf.float32,
                                   initializer=tf.truncated_normal_initializer(stddev=0.1,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                  shape=[256], 
                                  dtype=tf.float32,
                                  initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(norm2, weights, strides=[1,1,1,1],padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(pre_activation, name='conv3')    
    with tf.variable_scope('pooling3_lrn') as scope:
        norm3 = tf.nn.lrn(conv3, depth_radius=4, bias=1.0, alpha=0.001/9.0,
                           beta=0.75,name='norm2')
        pool3 = tf.nn.max_pool(norm3, ksize=[1,2,2,1], strides=[1,1,1,1],
                                padding='SAME',name='pooling3')
    with tf.variable_scope('local3') as scope:
        reshape = tf.reshape(pool3, shape=[batch_size, -1])
        dim = reshape.get_shape()[1].value
        weights = tf.get_variable('weights',
                                  shape=[dim,4096],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[4096],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name) 
    with tf.variable_scope('softmax_linear') as scope:
        weights = tf.get_variable('softmax_linear',
                                  shape=[4096, 6],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.005,dtype=tf.float32))
        biases = tf.get_variable('biases', 
                                 shape=[6],
                                 dtype=tf.float32, 
                                 initializer=tf.constant_initializer(0.1))
        soft_max = tf.add(tf.matmul(local3, weights), biases, name='softmax_linear')
    return soft_max

out_list=[]
def run_training():
    
    
    data ="/home/lab326/songpeng/dataset/"
    log_dir = "/home/lab326/songpeng/log/"
    image,label = get_files(data)
    X_train,X_test,Y_train,Y_test=train_test_split(image,label,test_size = 0.2)
    image_batches,label_batches = get_batches(X_train,Y_train,32,32,len(X_train),1200)
    print(len(X_train),len(X_test))
    p = mmodel1(image_batches,len(X_train))
    cost = loss(p,label_batches)
    train_op = training(cost,0.001)
    acc = get_accuracy(p,label_batches)
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    saver = tf.train.Saver()
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess = sess,coord = coord)
    count=0
    #flag=0
    try:
        while not coord.should_stop():
            sess.run(train_op)
            for step in np.arange(1000):
                
                _,train_acc,train_loss = sess.run([train_op,acc,cost])
                if train_acc>0.98:
                    count+=1
                if step % 100 == 0:
                    print(step)
                    check = os.path.join(log_dir,"model.ckpt")
                    saver.save(sess,check,global_step = step)
                if count>10:
                    #flag=1
                    coord.request_stop()
                    break
                    
                   
                print("loss:{} accuracy:{}".format(train_loss,train_acc))
                outresult = np.array([train_loss,train_acc])        
                out_list.append(outresult)
                if train_acc>0.98:
                    count+=1
          
    except tf.errors.OutOfRangeError:
        print("Done!!!")
    finally:
        coord.request_stop()
    coord.join(threads)
    '''
    if flag==1:
        image_batches1,label_batches1 = get_batches(X_test,Y_test,32,32,len(X_test),300)
        p1 = mmodel1(image_batches1,len(X_test))
        #cost1 = loss(p1,label_batches1)
        #train_op1 = training(cost1,0.001)
        acc1 = get_accuracy(p1,label_batches1)
        train_acc1 = sess.run([acc1])
        print("测试集的识别率")
        print("accuracy:{}".format(train_acc1))
    '''
    sess.close()
    
if __name__ == '__main__':
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    #run_training()
    '''
    print(len(out_list)) 
    f = codecs.open("/home/lab326/songpeng/result.txt",'w','utf-8')
    for i in range(len(out_list)):
        f.write(str(i)+'\r\n')
        f.write(str(out_list[i])+'\r\n')
    f.close()
    '''
    test("/home/lab326/songpeng/test.jpg")
    
    
    
    
    