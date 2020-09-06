import os
import time

import tensorflow as tf
import math
import numpy as np 
import scipy.io as sio
import ntpath
from random import shuffle
import random
import re
from scipy import ndimage
import imutils

from custom_vgg16 import * # code from https://github.com/antlerros/tensorflow-fast-neuralstyle
from videomaker import *
from utils_robot import *
from ops import *
from ops_sn import *

class DCGAN(object):
    def __init__(self, sess,
                 batch_size=256,gf_dim=64, df_dim=64,c_dim=3, is_train=True):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.model_name = "DCGAN.model"
        self.sess = sess
        self.batch_size = batch_size
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim

        self.sample_size = 256
        self.input_size = 256
        self.dis_dim = 64

        self.build_model(is_train)


    def build_model(self, is_train):
        
        self.local_size = self.input_size/2

        self.fnum = 17 #amount of frames of one video clip
        self.nb_lys = 2**3
        self.feature_dim = self.df_dim*4

        self.img_t = tf.placeholder(tf.float32, shape=(self.batch_size, self.sample_size, self.sample_size, 3),name='img_t')
        self.img_0 = tf.placeholder(tf.float32, shape=(self.batch_size, self.sample_size, self.sample_size, 3),name='img_0')
        self.img_gt = tf.placeholder(tf.float32, shape=(self.batch_size, self.sample_size, self.sample_size, 3),name='img_gt')
        self.img_next_gt = tf.placeholder(tf.float32, shape=(self.batch_size, self.sample_size, self.sample_size, 3),name='img_next_gt')
        
        self.ms = tf.placeholder(tf.float32, shape=(self.batch_size, self.sample_size, self.sample_size, 1),name='ms')
        self.ms_t = tf.placeholder(tf.float32, shape=(self.batch_size, self.sample_size, self.sample_size, 1),name='ms_t')
        self.rep_img = tf.placeholder(tf.float32, shape=(self.batch_size, self.sample_size/self.nb_lys, self.sample_size/self.nb_lys, self.feature_dim),name='rep_img')
        self.rep_flag = tf.placeholder(tf.int32, shape=(self.batch_size, self.sample_size/self.nb_lys, self.sample_size/self.nb_lys, self.feature_dim),name='rep_flag')
        self.img_gt_mask = tf.placeholder(tf.float32, shape=(self.batch_size, self.sample_size, self.sample_size, 3),name='img_gt_mask')
        self.img_0_mask = tf.placeholder(tf.float32, shape=(self.batch_size, self.sample_size, self.sample_size, 3),name='img_0_mask')
        

        with tf.variable_scope('generator') as scope: 

            self.rep_img_0 = self.encoder_f(self.img_0)
            self.rep_ms = self.encoder_m(self.ms)
            self.rep_ms_t = self.encoder_m(self.ms_t,reuse=True)

            self.rep_img_t = tf.where(tf.equal(self.rep_flag,0),self.rep_img_0,self.rep_img)

            self.rep_for_AE = tf.concat(axis=3,values=[self.rep_img_t,self.rep_img_0])
            self.rep_for_AE = tf.concat(axis=3,values=[self.rep_for_AE,self.rep_ms])
            self.rep_for_AE = tf.concat(axis=3,values=[self.rep_for_AE,self.rep_ms_t])
            
            self.rep_img_next = self.AE_predict(self.rep_for_AE)

            self.img_next = self.decoder_f(self.rep_img_next)

            self.rep_img_t_gen = self.encoder_f(self.img_next_gt,reuse=True)
            self.img_t_gen = self.decoder_f(self.rep_img_t_gen,reuse=tf.AUTO_REUSE)

            self.img_next_local = tf.reshape(tf.boolean_mask(self.img_next,self.img_gt_mask),(self.batch_size,self.local_size,self.local_size,3))
            self.img_next_gt_local = tf.reshape(tf.boolean_mask(self.img_next_gt,self.img_gt_mask),(self.batch_size,self.local_size,self.local_size,3))
            self.img_0_local = tf.reshape(tf.boolean_mask(self.img_0,self.img_0_mask),(self.batch_size,self.local_size,self.local_size,3))
            self.img_t_local = tf.reshape(tf.boolean_mask(self.img_t,self.img_gt_mask),(self.batch_size,self.local_size,self.local_size,3))
            self.img_gt_local = tf.reshape(tf.boolean_mask(self.img_gt,self.img_gt_mask),(self.batch_size,self.local_size,self.local_size,3))

            self.ms_resize = tf.image.resize_images(self.ms,[self.local_size,self.local_size])
            self.ms_t_resize = tf.image.resize_images(self.ms_t,[self.local_size,self.local_size])
            
            data_dict = loadWeightsData('vgg16.npy') 
            # ground truth feature
            vgg_s = custom_Vgg16(self.img_next_local, data_dict=data_dict)
            self.feature_ = [vgg_s.conv1_2, vgg_s.conv2_2, vgg_s.conv3_3, vgg_s.conv4_3, vgg_s.conv5_3]
            
            # prediction feature
            vgg_s = custom_Vgg16(self.img_next_gt_local, data_dict=data_dict)
            self.feature = [vgg_s.conv1_2, vgg_s.conv2_2, vgg_s.conv3_3, vgg_s.conv4_3, vgg_s.conv5_3]
            


        with tf.variable_scope('discriminator_global_local') as scope:
            img =  tf.concat(axis=3, values=[self.img_next, self.img_0])
            img = tf.concat(axis=3,values=[img,self.ms_t])
            
            crop =  tf.concat(axis=3, values=[self.img_next_local, self.img_0_local])
            crop = tf.concat(axis=3,values=[crop,self.ms_t_resize]) 
            d_fake_local = self.discriminator_global_local(self.img_next,self.img_next_local, update_collection=None) 

            img =  tf.concat(axis=3, values=[self.img_next_gt, self.img_0])
            
            img = tf.concat(axis=3,values=[img,self.ms_t])
            crop =  tf.concat(axis=3, values=[self.img_next_gt_local, self.img_0_local])
            crop = tf.concat(axis=3,values=[crop,self.ms_t_resize])
            d_real_local = self.discriminator_global_local(self.img_next_gt,self.img_next_gt_local, reuse=tf.AUTO_REUSE,update_collection="NO_OPS")

            self.real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_real_local), logits=d_real_local))
            self.fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_fake_local), logits=d_fake_local))
            self.d_loss = self.real_loss + self.fake_loss
            self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_fake_local), logits=d_fake_local))
            
            self.loss_f = tf.zeros_like(self.g_loss)
            ii = 5
            for f, f_ in zip(self.feature, self.feature_):
                self.loss_f +=  ((0.5)**(ii))*tf.losses.absolute_difference(f,f_)
                ii = ii-1

        
        # Discriminator on (I_t,I_t+1) vs (I_t_estimated, I_t+1_estimated)
        with tf.variable_scope('discriminator_sequence') as scope:
            img = tf.concat(axis=3,values=[self.img_next,self.img_t])
            img = tf.concat(axis=3,values=[img,self.ms_t])
            d_seq_fake = self.discriminator_sequence(img,update_collection=None) 

            img = tf.concat(axis=3, values=[self.img_next_gt, self.img_gt])
            img = tf.concat(axis=3,values=[img,self.ms_t])
            d_seq_real = self.discriminator_sequence(img, reuse=tf.AUTO_REUSE,update_collection="NO_OPS")

            self.real_seq_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_seq_real), logits=d_seq_real))
            self.fake_seq_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(d_seq_fake), logits=d_seq_fake))
            self.d_seq_loss = self.real_seq_loss + self.fake_seq_loss
            self.g_seq_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(d_seq_fake), logits=d_seq_fake))
            

        with tf.variable_scope('L2') as scope:

            self.rec_loss = tf.reduce_mean(tf.square(self.img_next - self.img_next_gt))

            self.rec_true_loss = tf.reduce_mean(tf.square(self.img_t_gen-self.img_next_gt))

            self.rec_local_loss = tf.reduce_mean(tf.square(self.img_next_local-self.img_next_gt_local))


        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'discriminator_global_local' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.s_vars = [var for var in t_vars if 'discriminator_sequence' in var.name]
        

        self.saver = tf.train.Saver(self.d_vars + self.g_vars,
                                    max_to_keep=0)


    def train(self, config, run_string="???"):
        """Train DCGAN"""

        if config.continue_from_iteration:
            counter = config.continue_from_iteration
        else:
            counter = 0

        global_step = tf.Variable(counter, name='global_step', trainable=False)
        
        # Learning rate of generator is gradually decreasing.
        self.g_lr = tf.train.exponential_decay(0.0002,global_step=global_step,decay_steps=20000,decay_rate=0.9,staircase=True)
        self.d_lr = tf.train.exponential_decay(0.0002,global_step=global_step,decay_steps=20000,decay_rate=0.9,staircase=True)
        
        g_optim = tf.train.AdamOptimizer(learning_rate=self.g_lr, beta1=config.beta1) \
                          .minimize(30*self.rec_loss+20*self.rec_true_loss+50*self.rec_local_loss+self.g_loss+self.g_seq_loss+10*self.loss_f, var_list=self.g_vars)

        d_optim = tf.train.AdamOptimizer(learning_rate=self.d_lr, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars, global_step=global_step)
        s_optim = tf.train.AdamOptimizer(learning_rate=self.d_lr, beta1=config.beta1) \
                          .minimize(self.d_seq_loss, var_list=self.s_vars)

        tf.global_variables_initializer().run()
        if config.continue_from:
            checkpoint_dir = os.path.join(os.path.dirname(config.checkpoint_dir), config.continue_from)
            print('Loading variables from ' + checkpoint_dir)
            self.load(checkpoint_dir, config.continue_from_iteration)

        start_time = time.time()

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord) 
        self.make_summary_ops()
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(config.summary_dir, graph_def=self.sess.graph_def)

        folder_path = 'train_examples/'
        subfolders = os.listdir(folder_path)
        subfolders.sort(reverse=False)

        fnum = self.fnum # frame number
        local_size = self.input_size/2

        for _ in xrange(200000):  
            counter = counter+1 
            sp_data = []
            ms_all_points = []
            ms_drawing = []
            print(counter) 

            video_idx = random.sample(range(len(subfolders)),self.batch_size) # randomly pick up 64 video segment
            for s in video_idx: # loop for batch_size
                f = subfolders[s]
                tmp_folder = os.path.join(folder_path,f)

                fms = os.listdir(tmp_folder)
                fms.sort(reverse=False)

                frames = [os.path.join(tmp_folder,x) for x in fms]

                a = random.randint(1,len(frames)-fnum-2)
                file_idx = random.sample(range(a,a+fnum),fnum)
                file_idx.sort()
                frames_fnum = [frames[x] for x in file_idx]
                sp_data.append(frames_fnum)

                ms_file = os.path.join(tmp_folder,'stroke.txt')
                ms = open(ms_file,'r').readlines()
                if len(ms)>25:
                    ms = ms[1:]
                ms = [ms[x-1] for x in file_idx]
                ms_points = [[int(x.strip().split(',')[0]),int(x.strip().split(',')[1])] for x in ms]
                ms_all_points.append(ms_points)
                

            img_0_batch = [row[0] for row in sp_data]
            img_0 = [get_img(x,self.input_size) for x in img_0_batch]
            
            img_batch_files = [row[1] for row in sp_data]
            img_next_gt = [get_img(x,self.input_size) for x in img_batch_files]
            
            img_gt = img_0
            
            ms = [get_ms_img_robot(point_list,self.input_size,0) for point_list in ms_all_points] # c=0 if get whole ms
            ms_t = [get_ms_img_robot([x[0],x[1]],self.input_size,0) for x in ms_all_points]

            img_0_mask = [get_local_mask_robot([x_ms[0],x_ms[1]],self.input_size,local_size) for x_ms in ms_all_points]
            img_gt_mask = img_0_mask

            rep_flag = np.zeros((self.batch_size, self.sample_size/self.nb_lys, self.sample_size/self.nb_lys, self.feature_dim))
            rep_img_0 = np.random.normal(0,1,(self.batch_size, self.sample_size/self.nb_lys, self.sample_size/self.nb_lys, self.feature_dim))
            
            feed_dict = {self.img_next_gt:img_next_gt,self.img_gt:img_gt,self.img_0:img_0,self.rep_img:rep_img_0,\
                 self.ms_t:ms_t,self.ms:ms,self.rep_flag:rep_flag,\
                 self.img_gt_mask:img_gt_mask,self.img_0_mask:img_0_mask,self.img_t:img_0}

            self.sess.run([g_optim,d_optim,s_optim],feed_dict=feed_dict)
            
            I_next,rep_img = self.sess.run([self.img_next,self.rep_img_next],feed_dict=feed_dict)
            
            if np.mod(counter, 300) == 1:
                I_next_store = []
                img_next_gt_store = []
                I_next_store.append(img_0)
                img_next_gt_store.append(img_0)
                I_next_store.append(I_next)
                img_next_gt_store.append(img_next_gt)


            rep_flag = np.ones((self.batch_size, self.sample_size/self.nb_lys, self.sample_size/self.nb_lys, self.feature_dim))

            for i in xrange(1,fnum-1):

                img_gt = img_next_gt

                img_batch_files = [row[i+1] for row in sp_data]
                img_next_gt = [get_img(x,self.input_size) for x in img_batch_files]
                ms_t = [get_ms_img_robot([x[i],x[i+1]],self.input_size,i) for x in ms_all_points]
                
                img_gt_mask = [get_local_mask_robot([x_ms[i],x_ms[i+1]],self.input_size,local_size) for x_ms in ms_all_points]
                
               
                feed_dict = {self.img_next_gt:img_next_gt,self.img_gt:img_gt,self.img_0:img_0,self.rep_img:rep_img,\
                 self.ms_t:ms_t,self.ms:ms,self.rep_flag:rep_flag,\
                 self.img_gt_mask:img_gt_mask,self.img_0_mask:img_0_mask,self.img_t:I_next}
                         
                self.sess.run([g_optim,d_optim,s_optim],feed_dict=feed_dict)
                

                I_next,rep_img = self.sess.run([self.img_next,self.rep_img_next],feed_dict=feed_dict)

                if np.mod(counter, 300) == 1:
                    I_next_store.append(I_next)
                    img_next_gt_store.append(img_next_gt)
                    

            if np.mod(counter, 300) == 1: 
                show_num = 4
                grid_size = np.ceil(np.sqrt(self.batch_size))
                grid= [show_num, 9,self.input_size]
                image_gen = np.array(I_next_store)
                ref = np.array(img_next_gt_store)
                ms = np.array(ms)
                
                grid_size = np.ceil(np.sqrt(self.batch_size))
                save_img_video(ref,image_gen,grid,os.path.join(config.summary_dir,'%s_train_img.png' % (counter)))
                save_images(ms,[grid_size,grid_size],os.path.join(config.summary_dir,'%s_train_ms.png' % (counter)))

                ref = np.swapaxes(ref,0,1)
                image_gen = np.swapaxes(image_gen,0,1)
                filename = os.path.join(config.summary_dir,'%s_train_stroke_img.avi' % (counter))
                make_video_with_stroke_robot(ref, image_gen, strokes=ms_all_points, filename=filename)


            if np.mod(counter, 300) == 1:
                self.save(config.checkpoint_dir, counter)

            if np.mod(counter, 50) == 1:
                summary_str = self.sess.run(summary_op,feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, counter)


    def discriminator_sequence(self, image1,reuse=False, update_collection=tf.GraphKeys.UPDATE_OPS):
        with tf.variable_scope('discriminator_seq_sn'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            c0_0 = sn_lrelu(sn_conv2d(image1,  self.dis_dim, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='s_c0_0'))
            c0_1 = sn_lrelu(sn_conv2d(c0_0, self.dis_dim*2, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='s_c0_1'))
            c1_0 = sn_lrelu(sn_conv2d(c0_1, self.dis_dim*2, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='s_c1_0'))
            c1_1 = sn_lrelu(sn_conv2d(c1_0, self.dis_dim*4, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='s_c1_1'))
            c2_0 = sn_lrelu(sn_conv2d(c1_1, self.dis_dim*4, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='s_c2_0'))
            c2_1 = sn_lrelu(sn_conv2d(c2_0, self.dis_dim*8, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='s_c2_1'))
            c3_0 = sn_lrelu(sn_conv2d(c2_1, self.dis_dim*8, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='s_c3_0'))
            c3_0 = tf.reshape(c3_0, [self.batch_size, -1])
            l4 = sn_linear(c3_0, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='s_l4')
            return tf.reshape(l4, [-1])


    def discriminator_global_local(self, image,crop,reuse=False, update_collection=tf.GraphKeys.UPDATE_OPS):
        with tf.variable_scope('discriminator_global_local'):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            # global 256
            c0_0 = sn_lrelu(sn_conv2d(image,  self.dis_dim, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_c0_0'))
            c0_1 = sn_lrelu(sn_conv2d(c0_0, self.dis_dim*2, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_c0_1'))
            c1_0 = sn_lrelu(sn_conv2d(c0_1, self.dis_dim*2, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_c1_0'))
            c1_1 = sn_lrelu(sn_conv2d(c1_0, self.dis_dim*4, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_c1_1'))
            c2_0 = sn_lrelu(sn_conv2d(c1_1, self.dis_dim*4, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_c2_0'))
            c2_1 = sn_lrelu(sn_conv2d(c2_0, self.dis_dim*8, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_c2_1'))
            c3_0 = sn_lrelu(sn_conv2d(c2_1, self.dis_dim*8, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_c3_0'))
            c3_0 = tf.reshape(c3_0, [self.batch_size, -1])
            
            #local 128
            y_l = sn_lrelu(sn_conv2d(crop,  self.dis_dim/2, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_l0_0'))
            y_l = sn_lrelu(sn_conv2d(y_l, self.dis_dim, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_l0_1'))
            y_l = sn_lrelu(sn_conv2d(y_l, self.dis_dim, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_l1_0'))
            y_l = sn_lrelu(sn_conv2d(y_l, self.dis_dim*2, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_l1_1'))
            y_l = sn_lrelu(sn_conv2d(y_l, self.dis_dim*2, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_l2_0'))
            y_l = sn_lrelu(sn_conv2d(y_l, self.dis_dim*4, 4, 4, 2, 2, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_l2_1'))
            y_l = sn_lrelu(sn_conv2d(y_l, self.dis_dim*4, 3, 3, 1, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_l3_0'))
            y_l = tf.reshape(y_l, [self.batch_size, -1])
            
            y = tf.concat(values=[c3_0,y_l],axis=-1)
            l4 = sn_linear(y, 1, spectral_normed=True, update_collection=update_collection, stddev=0.02, name='d_l4')
            return tf.reshape(l4, [-1])

   

    def encoder_m(self,image,reuse=False,update_collection=tf.GraphKeys.UPDATE_OPS):
        with tf.variable_scope('encoder_m') as scope: 
            if reuse:
                tf.get_variable_scope().reuse_variables()
            s = lrelu(instance_norm(conv2d(image, self.df_dim, name='g_s0_conv')))
            s = lrelu(instance_norm(conv2d(s, self.df_dim * 2, name='g_s1_conv')))
            s = lrelu(instance_norm(conv2d(s, self.df_dim * 4, name='g_s2_conv')))

            return s

    def encoder_f(self,motion_of,reuse=False,update_collection=tf.GraphKeys.UPDATE_OPS):
        with tf.variable_scope('encoder_f') as scope: 
            if reuse:
                tf.get_variable_scope().reuse_variables() 
            f = lrelu(instance_norm(conv2d(motion_of, self.df_dim, name='g_f0_conv')))
            f = lrelu(instance_norm(conv2d(f, self.df_dim * 2, name='g_f1_conv')))
            f = lrelu(instance_norm(conv2d(f, self.gf_dim * 4, name='g_f2_conv')))
            
            return f


    def decoder_f(self, d, reuse=False,update_collection=tf.GraphKeys.UPDATE_OPS): 
        with tf.variable_scope('decoder_f') as scope:
            if reuse:
                tf.get_variable_scope().reuse_variables()            

            relu = tf.nn.relu
            d = relu(sn_deconv2d(d, [self.batch_size, self.input_size/4, self.input_size/4, self.gf_dim*4],spectral_normed=True, name='g_f1', stddev=0.02))
            d = relu(sn_deconv2d(d, [self.batch_size, self.input_size/2, self.input_size/2, self.gf_dim*2],spectral_normed=True, name='g_f2', stddev=0.02))
            d = relu(sn_deconv2d(d, [self.batch_size, self.input_size, self.input_size, self.gf_dim*1],spectral_normed=True, name='g_f3', stddev=0.02))
            d = sn_deconv2d(d, [self.batch_size, self.input_size, self.input_size, 3], 3, 3, 1, 1, name='g_f4',spectral_normed=True, stddev=0.02)

            return tf.nn.tanh(d)

    def AE_predict(self,x,reuse=False,update_collection=tf.GraphKeys.UPDATE_OPS):
        with tf.variable_scope('AE_predict') as scope: 
            if reuse:
                tf.get_variable_scope().reuse_variables()
        
            x = self.dense_block(input_x=x, nb_layers=5, layer_name='g_dense_1')
            
            x = self.dense_block(input_x=x, nb_layers=5, layer_name='g_dense_2')
            
            x = self.dense_block(input_x=x, nb_layers=5, layer_name='g_dense_3')
            
            x = self.dense_block(input_x=x, nb_layers=5, layer_name='g_dense_4')

            x = self.dense_block(input_x=x, nb_layers=5, layer_name='g_dense_5')
            
            x = self.dense_block(input_x=x, nb_layers=5, layer_name='g_dense_final') 
            x = self.transition_layer(x, scope='g_trans_3')

            return x
            

    def bottleneck_layer(self, x, scope):
        with tf.name_scope(scope):
            Relu = tf.nn.relu
            x = Relu(x)
            x = conv2d(x, self.df_dim*2, k_w=1,k_h=1,d_w=1,d_h=1, name=scope+'_conv1')
            x = Relu(x)
            x = conv2d(x, self.df_dim*4, k_w=3,k_h=3,d_w=1,d_h=1, name=scope+'_conv2')
            
            return x

    def transition_layer(self, x, scope):
        with tf.name_scope(scope):
            Relu = tf.nn.relu
            x = Relu(x)
            x = conv2d(x, self.feature_dim, k_w=1,k_h=1,d_w=1,d_h=1, name=scope+'_conv1')
        
            return x

    def dense_block(self, input_x, nb_layers, layer_name):
        with tf.name_scope(layer_name):
            layers_concat = list()
            layers_concat.append(input_x)

            x = self.bottleneck_layer(input_x, scope=layer_name + '_bottleN_' + str(0))

            layers_concat.append(x)

            for i in range(nb_layers - 1):
                x = tf.concat(axis=-1,values=layers_concat)
                x = self.bottleneck_layer(x, scope=layer_name + '_bottleN_' + str(i + 1))
                layers_concat.append(x)

            x = tf.concat(axis=-1,values=layers_concat)

            return x
         
    def make_summary_ops(self):
        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('d_loss', self.d_loss)
        tf.summary.scalar('rec_loss', self.rec_loss)
        tf.summary.scalar('rec_true_loss', self.rec_true_loss)
        tf.summary.scalar('rec_loss_local', self.rec_local_loss)
        tf.summary.scalar('g_seq_loss', self.g_seq_loss)
        tf.summary.scalar('d_seq_loss', self.d_seq_loss)

    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir) 

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)

    def load(self, checkpoint_dir, iteration=None):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and iteration:
            # Restores dump of given iteration
            ckpt_name = self.model_name + '-' + str(iteration)
        elif ckpt and ckpt.model_checkpoint_path:
            # Restores most recent dump
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
        else:
            raise Exception(" [!] Testing, but %s not found" % checkpoint_dir)

        ckpt_file = os.path.join(checkpoint_dir, ckpt_name)
        print('Reading variables to be restored from ' + ckpt_file)
        self.saver.restore(self.sess, ckpt_file)
        return ckpt_name

def make_project_dir(project_dir):
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)
        os.makedirs(os.path.join(project_dir, 'models'))
        os.makedirs(os.path.join(project_dir, 'result'))
        os.makedirs(os.path.join(project_dir, 'result_test'))

