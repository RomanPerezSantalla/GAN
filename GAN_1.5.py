#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 02:33:34 2019

@author: roman

"""

import sys
import numpy as np
from keras.layers import Dense, Reshape, Input, BatchNormalization, Concatenate
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D, Convolution2D, MaxPooling2D,Deconvolution2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD, Nadam,Adamax
from keras import initializers
from keras.utils import plot_model
from astropy.io import fits
import tensorflow as tf
from copy import deepcopy
from random import randint
from PIL import Image

with tf.device('/gpu:0'):
    f = open("Losses.txt","w+")
    f.close()
    num_imag = 937
    num_imag_train = 800
    num_imag_test = num_imag-num_imag_train
    size_im = 256
    epochs = 150
    batch_size = 1
    
    class Generator(object):
        def __init__(self, width = size_im, height = size_im, channels = 1):
            
            self.W = width
            self.H = height
            self.C = channels
            self.SHAPE = (width,height,channels)
    
            self.Generator = self.model()
            self.OPTIMIZER = Adam(lr=2e-4, beta_1=0.5,decay=1e-5)
            self.Generator.compile(loss='binary_crossentropy', 
                 optimizer=self.OPTIMIZER,metrics=['accuracy'])
    
            #self.save_model()
            self.summary()
            
        def model(self):
            input_layer = Input(shape=self.SHAPE)
            down_1 = Convolution2D(64 , kernel_size=4, strides=2,  
                     padding='same',activation=LeakyReLU(alpha=0.2))(input_layer)

            down_2 = Convolution2D(64*2, kernel_size=4, strides=2, 
                     padding='same',activation=LeakyReLU(alpha=0.2))(down_1)
            norm_2 = BatchNormalization()(down_2)
            
            down_3 = Convolution2D(64*4, kernel_size=4, strides=2, 
                     padding='same',activation=LeakyReLU(alpha=0.2))(norm_2)
            norm_3 = BatchNormalization()(down_3)
            
            down_4 = Convolution2D(64*8, kernel_size=4, strides=2, 
                     padding='same',activation=LeakyReLU(alpha=0.2))(norm_3)
            norm_4 = BatchNormalization()(down_4)
            
            down_5 = Convolution2D(64*8, kernel_size=4, strides=2, 
                     padding='same',activation=LeakyReLU(alpha=0.2))(norm_4)
            norm_5 = BatchNormalization()(down_5)
            
            down_6 = Convolution2D(64*8, kernel_size=4, strides=2, 
                     padding='same',activation=LeakyReLU(alpha=0.2))(norm_5)
            norm_6 = BatchNormalization()(down_6)
            
            down_7 = Convolution2D(64*8, kernel_size=4, strides=2, 
                     padding='same',activation=LeakyReLU(alpha=0.2))(norm_6)
            norm_7 = BatchNormalization()(down_7)
            
            upsample_1 = UpSampling2D(size=2)(norm_7)
            up_conv_1 = Convolution2D(64*8, kernel_size=4, strides=1,  
                        padding='same',activation='relu')(upsample_1)
            norm_up_1 = BatchNormalization(momentum=0.8)(up_conv_1)
            add_skip_1 = Concatenate()([norm_up_1,norm_6])
            
            upsample_2 = UpSampling2D(size=2)(add_skip_1)
            up_conv_2 = Convolution2D(64*8, kernel_size=4, strides=1, 
                        padding='same',activation='relu')(upsample_2)
            norm_up_2 = BatchNormalization(momentum=0.8)(up_conv_2)
            add_skip_2 = Concatenate()([norm_up_2,norm_5])
            
            upsample_3 = UpSampling2D(size=2)(add_skip_2)
            up_conv_3 = Convolution2D(64*8, kernel_size=4, strides=1, 
                        padding='same',activation='relu')(upsample_3)
            norm_up_3 = BatchNormalization(momentum=0.8)(up_conv_3)
            add_skip_3 = Concatenate()([norm_up_3,norm_4])
            
            upsample_4 = UpSampling2D(size=2)(add_skip_3)
            up_conv_4 = Convolution2D(64*4, kernel_size=4, strides=1, 
                        padding='same',activation='relu')(upsample_4)
            norm_up_4 = BatchNormalization(momentum=0.8)(up_conv_4)
            add_skip_4 = Concatenate()([norm_up_4,norm_3])
            
            upsample_5 = UpSampling2D(size=2)(add_skip_4)
            up_conv_5 = Convolution2D(64*2, kernel_size=4, strides=1, 
                        padding='same',activation='relu')(upsample_5)
            norm_up_5 = BatchNormalization(momentum=0.8)(up_conv_5)
            add_skip_5 = Concatenate()([norm_up_5,norm_2])
            
            upsample_6 = UpSampling2D(size=2)(add_skip_5)
            up_conv_6 = Convolution2D(64, kernel_size=4, strides=1, 
                        padding='same',activation='relu')(upsample_6)
            norm_up_6 = BatchNormalization(momentum=0.8)(up_conv_6)
            add_skip_6 = Concatenate()([norm_up_6,down_1])
            
            last_upsample = UpSampling2D(size=2)(add_skip_6)
            output_layer = Convolution2D(self.C, kernel_size=4, strides=1, 
                           padding='same',activation='tanh')(last_upsample)
             
            return Model(input_layer,output_layer)
        
        def summary(self):
            return self.Generator.summary()
        
    class Discriminator(object):
        def __init__(self, width = 256, height= 256, channels = 1, starting_filters=64):
            self.W = width
            self.H = height
            self.C = channels
            self.CAPACITY = width*height*channels
            self.SHAPE = (width,height,channels)
            self.FS = starting_filters #FilterStart
            
            self.Discriminator = self.model()
            self.OPTIMIZER = Adam(lr=2e-4, beta_1=0.5,decay=1e-5)
            self.Discriminator.compile(loss='mse', 
                 optimizer=self.OPTIMIZER, metrics=['accuracy'] )
    
            #self.save_model()
            self.summary()
            
        def model(self):
            input_A = Input(shape=self.SHAPE)
            input_B = Input(shape=self.SHAPE)
            input_layer = Concatenate(axis=-1)([input_A, input_B])
            
            up_layer_1 = Convolution2D(self.FS, kernel_size=4, strides=2,  
                  padding='same',activation=LeakyReLU(alpha=0.2))(input_layer)
            
            up_layer_2 = Convolution2D(self.FS*2, kernel_size=4, strides=2,   
                 padding='same',activation=LeakyReLU(alpha=0.2))(up_layer_1)
            leaky_layer_2 = BatchNormalization(momentum=0.8)(up_layer_2)
            
            up_layer_3 = Convolution2D(self.FS*4, kernel_size=4, strides=2, 
                  padding='same',activation=LeakyReLU(alpha=0.2))(leaky_layer_2)
            leaky_layer_3 = BatchNormalization(momentum=0.8)(up_layer_3)
            
            up_layer_4 = Convolution2D(self.FS*8, kernel_size=4, strides=2, 
                 padding='same',activation=LeakyReLU(alpha=0.2))(leaky_layer_3)
            leaky_layer_4 = BatchNormalization(momentum=0.8)(up_layer_4)
                 
            output_layer = Convolution2D(1, kernel_size=4, strides=1, 
                 padding='same')(leaky_layer_4)
    
            return Model([input_A, input_B],output_layer)
        
        def summary(self):
            return self.Discriminator.summary()            
        
    class GAN(object):
        def __init__(self,model_inputs=[],model_outputs=[]):
            self.inputs = model_inputs
            self.outputs = model_outputs
            self.gan_model = Model(inputs = self.inputs, outputs = self.outputs)
            self.OPTIMIZER = Adam(lr=2e-4, beta_1=0.5)
            self.gan_model.compile(loss=['mse','mae'],loss_weights=[1, 100], optimizer=self.OPTIMIZER)
            #self.save_model()
            self.summary()
        def model(self):
            model = Model()
            return model
        def summary(self):
            return self.gan_model.summary()
        
    class Trainer:
        def __init__(self, height = size_im, width = size_im, channels = 1, epochs = 5000, batch = 1, checkpoint = 50):
            self.EPOCHS = epochs
            self.BATCH = batch
            self.H = height
            self.W = width
            self.C = channels
            self.CHECKPOINT = checkpoint
            self.X_train_A, self.X_train_B = self.load_data(0)
            self.X_test_A, self.X_test_B = self.load_data(1)
            self.generator = Generator(height = self.H, width = self.W, channels = self.C)
            self.orig_A = Input(shape=(self.W, self.H, self.C))
            self.orig_B = Input(shape=(self.W, self.H, self.C))
            self.fake_A = self.generator.Generator(self.orig_B)
            self.discriminator = Discriminator(height = self.H, width = self.W, channels = self.C)
            self.discriminator.trainable = False
            self.valid = self.discriminator.Discriminator([self.fake_A,self.orig_B])
            model_inputs=[self.orig_A,self.orig_B]
            model_outputs=[self.valid,self.fake_A]
            self.gan = GAN(model_inputs=model_inputs,model_outputs=model_outputs)
            
        def norm_and_expand(self,arr):
            arr = (arr.astype(np.float32) - 127.5)/127.5
            normed = np.expand_dims(arr, axis=1)
            return normed
        
        def load_data(self, train_or_test):
            imageArr = []
            imgs_A = []
            imgs_B = []
            if train_or_test == 0:
                for i in range(num_imag_train):
                    im = Image.open('Imagen_Calar_Alto_Train_%d.jpg' % i).convert("L")
                    imData = np.asarray(im)
                    imageArr.append(imData)
            if train_or_test == 1:
                for i in range(num_imag_test):
                    im = Image.open('Imagen_Calar_Alto_Test_%d.jpg' % i).convert("L")
                    imData = np.asarray(im)
                    imageArr.append(imData)
            imgs_temp = np.array(imageArr)
            del imageArr
            for img in imgs_temp:
                imgs_A.append(img[:,:self.H])
                imgs_B.append(img[:,self.H:])
            
            del imgs_temp
            imgs_A_out = self.norm_and_expand(np.array(imgs_A))
            imgs_B_out = self.norm_and_expand(np.array(imgs_B))
            del imgs_A, imgs_B
            return imgs_A_out, imgs_B_out
            
        def train(self):
            for e in range(self.EPOCHS):
                X_train_A_temp = deepcopy(self.X_train_A)
                X_train_B_temp = deepcopy(self.X_train_B)
                print(e)
                number_of_batches = len(self.X_train_A)
                for b in range(number_of_batches):
                    starting_ind = randint(0, (len(X_train_A_temp)-1))
                    real_images_raw_A = X_train_A_temp[ starting_ind: (starting_ind + 1)]
                    real_images_raw_B = X_train_B_temp[ starting_ind: (starting_ind + 1)]
                    X_train_A_temp = np.delete(X_train_A_temp,range(starting_ind,(starting_ind + 1)), 0)
                    X_train_B_temp = np.delete(X_train_B_temp,range(starting_ind,(starting_ind + 1)), 0)
                    batch_A = real_images_raw_A.reshape(1, self.W, self.H, self.C)
                    batch_B = real_images_raw_B.reshape(1, self.W, self.H, self.C)
                    
                    y_valid = np.ones((1,)+(int(self.W / 2**4), int(self.W / 2**4), 1))
                    y_fake = np.zeros((1,)+(int(self.W / 2**4), int(self.W / 2**4), 1))
                    
                    fake_A = self.generator.Generator.predict(batch_B)
                    
                    discriminator_loss_real = self.discriminator.Discriminator.train_on_batch([batch_A,batch_B],y_valid)[0]
                    discriminator_loss_fake = self.discriminator.Discriminator.train_on_batch([fake_A,batch_B],y_fake)[0]
                    full_loss = 0.5 * np.add(discriminator_loss_real, discriminator_loss_fake)
                    generator_loss = self.gan.gan_model.train_on_batch([batch_A, batch_B],[y_valid,batch_A]) 
                    if b % self.CHECKPOINT == 0:
                        f = open("Losses.txt","a+")
                        f.write('Epoch: '+str(int(e))+', Batch: '+str(int(b))+', [Full Discriminator :: Loss: '+str(full_loss)+'], [ Generator :: Loss: '+str(generator_loss)+']\n')
                        f.close()
        def test(self):
            starting_ind = 0
            for z in range(num_imag_test):
                X_test_A_temp = deepcopy(self.X_test_A)
                X_test_B_temp = deepcopy(self.X_test_B)
                real_images_raw_A = X_test_A_temp[ starting_ind: (starting_ind + 1)]
                real_images_raw_B = X_test_B_temp[ starting_ind: (starting_ind + 1)]
                X_test_A_temp = np.delete(X_test_A_temp,range(starting_ind,(starting_ind + 1)), 0)
                X_test_B_temp = np.delete(X_test_B_temp,range(starting_ind,(starting_ind + 1)), 0)
                batch_A = real_images_raw_A.reshape(1, self.W, self.H, self.C)
                batch_B = real_images_raw_B.reshape(1, self.W, self.H, self.C)
                fake_A = self.generator.Generator.predict(batch_B)
                img_A = np.array(batch_A).reshape(size_im,size_im)
                img_B = np.array(batch_B).reshape(size_im,size_im)
                img_fake = np.array(fake_A).reshape(size_im,size_im)
                gen_imgs = np.concatenate([img_B, img_fake, img_A], axis = 1)
                gen_imgs = gen_imgs * 127.5 + 127.5
                gen_imgs = Image.fromarray(gen_imgs)
                gen_imgs = gen_imgs.convert("L")
                gen_imgs.save('/home/roman/Documents/TFG propio 2/Test con mas ruido/Imagen_Calar_Alto_Rec_%d.jpg' % z)
                starting_ind = starting_ind + 1
                
    trainer = Trainer(height = size_im, width = size_im, channels = 1, epochs = epochs, batch = batch_size, checkpoint = 200)
    trainer.train()
    trainer.test()