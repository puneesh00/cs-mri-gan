import tensorflow as tf 
import os
from keras.utils import multi_gpu_model 
from keras.models import Model, Input
from keras.layers import Conv2D, Conv2DTranspose
from keras.layers import Flatten, Add
from keras.layers import Concatenate, Activation
from keras.layers import LeakyReLU, BatchNormalization, Lambda
import numpy as np
from metrics import metrics
from keras.initializers import constant
import pickle
import time
from keras import backend as K
from tensorflow.python.ops import array_ops
from keras.initializers import RandomUniform

data_path='/home/cs-mri-gan/testing_gt.pickle'
usam_path='/home/cs-mri-gan/testing_usamp_1dg_a5.pickle'

df=open(data_path,'rb')
uf=open(usam_path,'rb')

dataset_real=pickle.load(df)
u_sampled_data=pickle.load(uf)

data = np.asarray(dataset_real[0:2000], dtype = 'float32')
usp_data = np.expand_dims(u_sampled_data[0:2000], axis = -1)

inp_shape = (256,256,2)
trainable = False

usp_img = usp_data.imag 
usp_real = usp_data.real

data_gen = np.concatenate((usp_real, usp_img), axis =-1)

def resden(x,fil,gr,beta,gamma_init,trainable):    
    x1=Conv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x)
    x1=BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x1)
    x1=LeakyReLU(alpha=0.2)(x1)
    
    x1=Concatenate(axis=-1)([x,x1])
    
    x2=Conv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x1)
    x2=BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x2)
    x2=LeakyReLU(alpha=0.2)(x2)

    x2=Concatenate(axis=-1)([x1,x2])
        
    x3=Conv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x2)
    x3=BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x3)
    x3=LeakyReLU(alpha=0.2)(x3)

    x3=Concatenate(axis=-1)([x2,x3])
    
    x4=Conv2D(filters=gr,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x3)
    x4=BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(x4)
    x4=LeakyReLU(alpha=0.2)(x4)

    x4=Concatenate(axis=-1)([x3,x4])
    
    x5=Conv2D(filters=fil,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x4)
    x5=Lambda(lambda x:x*beta)(x5)
    xout=Add()([x5,x])
    
    return xout

def resresden(x,fil,gr,betad,betar,gamma_init,trainable):
    x1=resden(x,fil,gr,betad,gamma_init,trainable)
    x2=resden(x1,fil,gr,betad,gamma_init,trainable)
    x3=resden(x2,fil,gr,betad,gamma_init,trainable)
    x3=Lambda(lambda x:x*betar)(x3)
    xout=Add()([x3,x])
    
    return xout


def generator(inp_shape, trainable = True): 
   gamma_init = tf.random_normal_initializer(1., 0.02) 

   fd=512
   gr=32
   nb=12
   betad=0.2
   betar=0.2

   inp_real_imag = Input(inp_shape)
   lay_128dn = Conv2D(64, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(inp_real_imag)
   
   lay_128dn = LeakyReLU(alpha = 0.2)(lay_128dn)
    
   lay_64dn = Conv2D(128, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_128dn)
   lay_64dn = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_64dn)
   lay_64dn = LeakyReLU(alpha = 0.2)(lay_64dn)
    
   lay_32dn = Conv2D(256, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_64dn)
   lay_32dn = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_32dn)
   lay_32dn = LeakyReLU(alpha=0.2)(lay_32dn)
    
   lay_16dn = Conv2D(512, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_32dn)
   lay_16dn = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_16dn)
   lay_16dn = LeakyReLU(alpha=0.2)(lay_16dn)  #16x16 
    
   lay_8dn = Conv2D(512, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_16dn)
   lay_8dn = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_8dn)
   lay_8dn = LeakyReLU(alpha=0.2)(lay_8dn) #8x8


   xc1=Conv2D(filters=fd,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_8dn) #8x8
   xrrd=xc1
   for m in range(nb):
     xrrd=resresden(xrrd,fd,gr,betad,betar,gamma_init,trainable)

   xc2=Conv2D(filters=fd,kernel_size=3,strides=1,padding='same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(xrrd)
   lay_8upc=Add()([xc1,xc2])

  
   lay_16up = Conv2DTranspose(1024, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_8upc) 
   lay_16up = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_16up)
   lay_16up = Activation('relu')(lay_16up) #16x16
    
   lay_16upc = Concatenate(axis = -1)([lay_16up,lay_16dn])
    
   lay_32up = Conv2DTranspose(256, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_16upc) 
   lay_32up = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_32up)
   lay_32up = Activation('relu')(lay_32up) #32x32
    
   lay_32upc = Concatenate(axis = -1)([lay_32up,lay_32dn])
     
   lay_64up = Conv2DTranspose(128, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_32upc) 
   lay_64up = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_64up)
   lay_64up = Activation('relu')(lay_64up) #64x64
    
   lay_64upc = Concatenate(axis = -1)([lay_64up,lay_64dn])
     
   lay_128up = Conv2DTranspose(64, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_64upc) 
   lay_128up = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_128up)
   lay_128up = Activation('relu')(lay_128up) #128x128
    
   lay_128upc = Concatenate(axis = -1)([lay_128up,lay_128dn])
     
   lay_256up = Conv2DTranspose(64, (4,4), strides = (2,2), padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_128upc) 
   lay_256up = BatchNormalization(gamma_initializer = gamma_init, trainable = trainable)(lay_256up)
   lay_256up = Activation('relu')(lay_256up) #256x256
    
   out =  Conv2D(1, (1,1), strides = (1,1), activation = 'tanh', padding = 'same', use_bias = True, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(lay_256up)

   model = Model(inputs = inp_real_imag, outputs = out)

   return model


#to infer all the models after a run
gen4 = generator(inp_shape = inp_shape, trainable = False)

f = open('/home/cs-mri-gan/cs_mri_a5_metrics.txt', 'x')
f = open('/home/cs-mri-gan/cs_mri_a5_metrics.txt', 'a')

for i in range(300):
   filename = '/home/cs-mri-gan/gen_weights_a5_%04d.h5' % (i+1)   
   gen4.load_weights(filename)
   out4 = gen4.predict(data_gen)
   psnr, ssim = metrics(data, out4[:,:,:,0],2.0)
   f.write('psnr = %.5f, ssim = %.7f' %(psnr, ssim))
   f.write('\n')
   print(psnr, ssim)


'''
#to infer one model
gen16 = generator(inp_shape = inp_shape, trainable = False)
gen16.load_weights('/home/cs-mri-gan/gen_weights_a5_best.h5')
out16 = gen16.predict(data_gen)
psnr, ssim = metrics(data, out16[:,:,:,0], 2.0)
print(psnr,ssim)
'''
