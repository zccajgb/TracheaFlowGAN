from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import tensorflow as tf
import image_gen_v2 as dg
import argparse
import sys
import os
from pprint import pprint
from datetime import datetime as dt
from itertools import chain,repeat,cycle
#from pprint import pprint
parser = argparse.ArgumentParser()
parser.parse_args()
FLAGS=None
BATCH_SIZE=100
BUFFER_SIZE=1000
#batch_size=tf.placeholder(tf.int64,name="batch_size")

class network:
	def __init__(self,name,Num_points,use_z=True,y_points=None,buffer_size=BUFFER_SIZE):
		
		self.name,self.Num_points,self.y_points,self.buffer_size=name,Num_points,y_points,buffer_size
		self.batch_size=tf.placeholder(tf.int64,name=self.name+"batch_size")
		self.training=tf.placeholder(tf.bool,name=self.name+"training")
		self.y_=tf.placeholder(tf.float32,shape=[None],name=self.name+"y_") #labels
		self.x_=tf.placeholder(tf.float32,shape=[None,self.Num_points],name=self.name+"x_") #area
		#~ self.y_fake=tf.placeholder(tf.float32,shape=[None,1],name="y_fake")
		#~ self.y_real=tf.placeholder(tf.float32,shape=[None,1],name="y_real")
		if use_z==True:
			self.z_=tf.placeholder(tf.float32,shape=[None,self.Num_points],name=self.name+"z_") #pressure
			self.x_2=tf.stack([self.x_,self.z_],axis=2)
		
		#	
		# DEFINE ITERATOR
		#
		with tf.name_scope(self.name+'Data_processing'):
		#	self.dataset=tf.data.Dataset.from_tensor_slices((self.y_,self.x_)) #creates dataset
			if use_z==True:
				self.dataset=tf.data.Dataset.from_tensor_slices((self.x_2)) #creates dataset
			else:
				self.dataset=tf.data.Dataset.from_tensor_slices((self.x_))				
			self.dataset=self.dataset.shuffle(buffer_size=self.buffer_size) #shuffles
			self.dataset=self.dataset.repeat().batch(self.batch_size) #batches
		with tf.name_scope(self.name+'Iterator'):
			self.iterator=tf.data.Iterator.from_structure(self.dataset.output_types,self.dataset.output_shapes) #creates iterators	
			#elf.y,self.x=self.iterator.get_next() #sets P as next batch of pressure, and A as next batch of area	
			self.x=self.iterator.get_next()
			self.dataset_init_op = self.iterator.make_initializer(self.dataset,name=self.name+'_dataset_init')
	def disc_init(self,FAKE=False,xf=None,zf=None):	
		self.FAKE=FAKE
		self.disc_training=tf.placeholder(tf.bool,name='disc_training')
		#self.x_fake_=tf.placeholder(tf.float32,shape=[None,Num_points],name="x_fake") #area
		#self.z_fake_=tf.placeholder(tf.float32,shape=[None,Num_points],name="y_fake") #pressure
		#~ with tf.name_scope(self.name+'Fake_Data_processing'):			
			#~ self.x_fake_=tf.stack([self.x_fake_,self.z_fake_],axis=2)
			#~ self.fake_dataset=tf.data.Dataset.from_tensor_slices((self.x_fake_)) #creates dataset
			#~ self.fake_dataset=self.fake_dataset.shuffle(buffer_size=BUFFER_SIZE) #shuffles
			#~ self.fake_dataset=self.fake_dataset.repeat().batch(self.batch_size) #batches
		#~ with tf.name_scope(self.name+'Fake_Iterator'):
			#~ self.fake_iterator=tf.data.Iterator.from_structure(self.fake_dataset.output_types,self.fake_dataset.output_shapes) #creates iterators	
			#~ self.x_fake=self.fake_iterator.get_next() #sets P as next batch of pressure, and A as next batch of area
			#~ self.fake_dataset_init_op = self.fake_iterator.make_initializer(self.fake_dataset,name=self.name+'_dataset_init')
		#~ with tf.name_scope(self.name+'Disc_Iterator'):
			#~ self._iterator=chain.from_iterable(repeat((self.iterator, self.fake_iterator)))
			#~ #self.disc_in=chain.from_iterable(repeat((self.x,x_FAKE)))
#			self._iterator=cycle([self.iterator,self.fake_iterator])

		if self.FAKE:
			self.x=tf.concat([xf,zf],axis=2)
		
	def disc_next_batch(self):	
		return next(self._iterator).get_next()
	#~ def Fake_Label(self):	
		#~ FAKE=cycle([False,True])
		#~ return next(FAKE)
		
	def conv_layer(x,FILTERS=32,ACT=tf.nn.relu,POOL_SIZE=[2,2],P_STRIDES=2,name='Name',K_SIZE=[5,5],K_STRIDES=(1,1)):
		""" Generates convolutional layer, returns "h_pool"
		Args:
		x: Input Data
		FILTERS: Integer, the dimensionality of the output space (i.e. the number of filters in the convolution) (DEFAULT=32).
		K_SIZE: An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window. Can be a single integer to specify the same value for all spatial dimensions. (DEFAULT=5x5)
		K_STRIDES: An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. Can be a single integer to specify the same value for all spatial dimensions. Specifying any stride value != 1 is incompatible with specifying any dilation_rate value != 1. (DEFAULT=1x1)
		ACT: Activation Function (default=ReLU)
		POOL_SIZE: An integer or tuple/list of 2 integers: (pool_height, pool_width) specifying the size of the pooling window. Can be a single integer to specify the same value for all spatial dimensions. (DEFAULT=2x2)
		P_STRIDES: An integer or tuple/list of 2 integers, specifying the strides of the pooling operation. Can be a single integer to specify the same value for all spatial dimensions. (DEFAULT=2)"""
	#	h_conv=tf.layers.conv2d(inputs=x_image,filters=FILTERS,kernel_size=K_SIZE,padding="same",activation=ACT,name=name)
		h_conv=tf.layers.conv2d(inputs=x,filters=FILTERS,kernel_size=K_SIZE,padding="same",activation=ACT,name=name)	
		h_pool=tf.layers.average_pooling2d(inputs=h_conv, pool_size=POOL_SIZE, strides=P_STRIDES)
		return h_pool
		
	def build_conv(self,Conv_shape,x=None,z=None,BATCH_NORMALISE=True):
		''' Builds Convolutional Part of Net'''
	
		self.x.set_shape([None,self.y_points,self.Num_points])
		self.x_image=tf.reshape(self.x,[-1,self.x.shape[1],self.x.shape[2],1])
		self.c=[]
		with tf.name_scope(self.name+'_'+Conv_shape[0][4]):
			self.c.append(conv_layer(self.x_image,Conv_shape[0]))
			if BATCH_NORMALISE:
				self.c[0]=tf.layers.batch_normalization(self.c[0],training=self.training)
			#tf.summary.histogram('c0',self.c[0])

		for i in range(1,len(Conv_shape)):
			with tf.name_scope(self.name+'_'+Conv_shape[i][4]):
				self.c.append(conv_layer(self.c[i-1],*Conv_shape[i]))
				if BATCH_NORMALISE:
					self.c[i]=tf.layers.batch_normalization(self.c[i],training=self.training)				
			#	tf.summary.histogram('c'+str(i),self.c[i])

		self.dense_input=tf.layers.flatten(self.c[-1])		
	def disc_1d_conv(self,Conv_shape,BATCH_NORMALISE=True):
		''' Builds De-Convolutional Part of Net'''
		#~ if disc_training=True:
			#~ self.disc_1d_input=self.disc_next_batch()
		#~ else:
			#~ self.disc_1d_input=tf.stack([x,z],axis=2)
		#print(type(self.x))
		self.disc_1d_input_image=self.x
		self.disc_1d_input_image.set_shape([None,self.Num_points,2])
		#print(type(self.x_image))
		#print('a',self.x_image)
		#self.x_image=tf.reshape(self.x,[-1,self.Num_points,2])
		self.c=[]
		with tf.name_scope(self.name+'_'+Conv_shape[0][-2]):
			#self.c.append(self.conv_layer(self.x_image,*Conv_shape[0]))
			self.c.append(tf.layers.conv1d(self.disc_1d_input_image,*Conv_shape[0]))
			if BATCH_NORMALISE:
				self.c[0]=tf.layers.batch_normalization(self.c[0],training=self.training,name=self.name+'_'+Conv_shape[0][-2]+'_BN',reuse=Conv_shape[0][-1])
			#tf.summary.histogram('c0',self.c[0])

		for i in range(1,len(Conv_shape)):
			with tf.name_scope(self.name+'_'+Conv_shape[i][-2]):
				self.c.append(tf.layers.conv1d(self.c[i-1],*Conv_shape[i]))
				if BATCH_NORMALISE:
					self.c[i]=tf.layers.batch_normalization(self.c[i],training=self.training,name=self.name+'_'+Conv_shape[i][-2]+'_BN',reuse=Conv_shape[i][-1])				
			#	tf.summary.histogram('c'+str(i),self.c[i])
		self.dense_input=tf.layers.flatten(self.c[-1])
	def generator_conv(self,x,Conv_shape,BATCH_NORMALISE=True):
		
		self.c=[]
		with tf.name_scope(self.name+'_'+Conv_shape[0,4]):
			self.c.append(tf.layers.conv2_transpose(x_image,*Conv_shape[0]))
			if BATCH_NORMALISE:
				self.c[0]=tf.layers.batch_normalization(self.c[0],training=self.training)
			#tf.summary.histogram('c0',self.c[0])

		for i in range(1,len(self.Conv_shape)):
			with tf.name_scope(self.name+'_'+Conv_shape[i,4]):
				self.c.append(tf.layers.conv2_transpose(self.c[i-1],*Conv_shape[i]))
				if BATCH_NORMALISE:
					self.c[i]=tf.layers.batch_normalization(self.c[i],training=self.training)				
			#	tf.summary.histogram('c'+str(i),self.c[i])

		self.generator_1d_conv_out=c[-1]
	def generator_1d_conv(self,x,Conv_shape,BATCH_NORMALISE=True):
		self.c=[]
		self.gen_1d_image=tf.expand_dims(x,axis=2)
		self.gen_1d_image.set_shape([None,self.Num_points,None])
		with tf.name_scope(self.name+'_'+Conv_shape[0][-1]):
			self.c.append(tf.layers.conv1d(self.gen_1d_image,*Conv_shape[0]))
			if BATCH_NORMALISE:
				self.c[0]=tf.layers.batch_normalization(self.c[0],training=self.training,name=self.name+'_'+Conv_shape[0][-1]+'_BN')
			tf.summary.histogram('c0',self.c[0])

		for i in range(1,len(Conv_shape)):
			with tf.name_scope(self.name+'_'+Conv_shape[i][-1]):
				self.c.append(tf.layers.conv1d(self.c[i-1],*Conv_shape[i]))
				if BATCH_NORMALISE:
					self.c[i]=tf.layers.batch_normalization(self.c[i],training=self.training,name=self.name+'_'+Conv_shape[i][-1]+'_BN')				
			#	tf.summary.histogram('c'+str(i),self.c[i])
		self.generator_conv_out=tf.layers.flatten(self.c[-1])	
	def generator_1d_deconv(self,x,deconv_shape,readout,BATCH_NORMALISE=True):
		#self.x_image=tf.reshape(self.x,[None,1,None])
		self.dc=[]
		with tf.name_scope(self.name+'_'+deconv_shape[0][-1]):
			batch_size=tf.shape(x)[0]
			if deconv_shape[0][3]=='VALID':
				out_h=(x.shape[1].value-1)*deconv_shape[0][2]+deconv_shape[0][0][0]
				out_h=x.shape[1].value
			else:
				out_h=(x.shape[1].value-1)*deconv_shape[0][2]+1
			deconv_shape[0][1]=tf.stack([batch_size,out_h,deconv_shape[0][1][2]])
			deconv_shape[0][0]=tf.get_variable(self.name+'_'+deconv_shape[0][-1]+'_filters', deconv_shape[0][0], initializer=tf.random_normal_initializer(stddev=0.02))
			self.dc.append(tf.nn.leaky_relu(tf.contrib.nn.conv1d_transpose(x,*deconv_shape[0])))
			if BATCH_NORMALISE:
				self.dc[0]=tf.layers.batch_normalization(self.c[0],training=self.training,name=self.name+'_'+deconv_shape[0][-1]+'_BN')
			#tf.summary.histogram('dc0',self.dc[0])

		for i in range(1,len(deconv_shape)):
			with tf.name_scope(self.name+'_'+deconv_shape[i][-1]):
				batch_size=tf.shape(self.dc[i-1])[0]
			
				if deconv_shape[i][3]=='VALID':
					out_h=(self.dc[i-1].shape[1].value-1)*deconv_shape[i][2]+deconv_shape[i][0][0]
					#out_h=253
					print(out_h)
				else:
					out_h=(self.dc[i-1].shape[1].value-1)*deconv_shape[i][2]+1
				deconv_shape[i][1]=tf.stack([batch_size,out_h,deconv_shape[i][1][2]])
				deconv_shape[i][0]=tf.get_variable(self.name+'_'+deconv_shape[i][-1]+'_filters', deconv_shape[i][0], initializer=tf.random_normal_initializer(stddev=0.02))
				self.dc.append(tf.leaky_relu(tf.contrib.nn.conv1d_transpose(self.dc[i-1],*deconv_shape[i])))
				if BATCH_NORMALISE:
					self.dc[i]=tf.layers.batch_normalization(self.dc[i],training=self.training,name=self.name+'_'+deconv_shape[i][-1]+'_BN')				
			#	tf.summary.histogram('dc'+str(i),self.dc[i])
		with tf.name_scope(self.name+'_'+readout[-1]):
			#print(self.name+'_'+readout[-1])
			batch_size=tf.shape(self.dc[-1])[0]
			readout[1]=tf.stack([batch_size,readout[1][1],readout[1][2]])
			readout[0]=tf.get_variable(self.name+'_'+readout[-1]+'filters', readout[0], initializer=tf.random_normal_initializer(stddev=0.02))
			self.generator_output=(tf.tanh(tf.contrib.nn.conv1d_transpose(self.dc[-1],*readout)))
			#self.generator_output=((tf.contrib.nn.conv1d_transpose(self.dc[-1],*readout)))
		#	if BATCH_NORMALISE:
		#		self.generator_output=tf.layers.batch_normalization(self.generator_output,training=self.training,name=self.name+'_'+readout[-1]+'_BN')
			tf.summary.histogram('generator_output',self.generator_output)
			return self.gen_1d_image, self.generator_output
	def no_conv(self):
		self.dense_input=self.x
		self.dense_input.set_shape([None,self.Num_points])

	def build_dense(self,Dense,dropout,Readout,BATCH_NORMALISE=True):

		
		''' Builds Dense part'''
		with tf.name_scope(self.name+'_'+Dense[0][4]):
			self.d=[]
			self.drop=[]
			#self.d.append(nn.dense_layer(self.dense_input,*Dense[0]))
			self.d.append(tf.layers.dense(self.dense_input,*Dense[0]))
			self.drop.append(tf.layers.dropout(self.d[0],dropout,training=self.training))
			if BATCH_NORMALISE:
				self.d[0]=tf.layers.batch_normalization(self.drop[0],training=self.training)			
			#tf.summary.histogram('d0',self.d[0])
		for i in range(1,len(Dense)):
			with tf.name_scope(self.name+'_'+Dense[i][4]):
				self.d.append(tf.layers.dense(self.d[i-1],*Dense[i]))
				self.drop.append(tf.layers.dropout(self.d[i],dropout,training=self.training))
				if BATCH_NORMALISE:
					self.d[i]=tf.layers.batch_normalization(self.drop[i],training=self.training)				
			#	tf.summary.histogram('d'+str(i),self.d[i])	
		with tf.name_scope(self.name+'_Dropout'):
			self.d_drop=tf.layers.dropout(self.d[-1],dropout,training=self.training,name="d_drop")
			tf.summary.histogram('dropout',self.d_drop)
		with tf.name_scope(self.name+'_Output_Layer'):
		#	self.y_est=nn.dense_layer(self.d_drop,Readout)
			self.y_est=tf.layers.dense(self.d_drop,*Readout)
			#tf.summary.histogram(self.name+'_y_est', self.y_est)
	def disc_dense(self,x,Dense,Readout,dropout,BATCH_NORMALISE=True):
		''' Builds Dense part'''
		#~ fake=self.Fake_Label()
		with tf.name_scope(self.name+'_'+Dense[0][-2]):
			self.d=[]
			self.drop=[]
			#self.d.append(nn.dense_layer(self.dense_input,*Dense[0]))
			self.d.append(tf.layers.dense(x,*Dense[0]))
			self.drop.append(tf.layers.dropout(self.d[0],dropout,training=self.training))
			if BATCH_NORMALISE:
				self.d[0]=tf.layers.batch_normalization(self.drop[0],training=self.training,name=self.name+'_'+Dense[0][-2]+'_BN',reuse=Dense[0][-1])
			#tf.summary.histogram('d0',self.d[0])
		for i in range(1,len(Dense)):
			with tf.name_scope(self.name+'_'+Dense[i][-2]):
				#self.d.append(nn.dense_layer(self.d[i-1],*Dense[i]))
				self.d.append(tf.layers.dense(self.d[i-1],*Dense[i]))
				self.drop.append(tf.layers.dropout(self.d[i],dropout,training=self.training))
				if BATCH_NORMALISE:
					self.d[i]=tf.layers.batch_normalization(self.drop[i],training=self.training,name=self.name+'_'+Dense[i][-2]+'_BN',reuse=Dense[i][-1])				
			#	tf.summary.histogram('d'+str(i),self.d[i])	
		#with tf.name_scope(self.name+'_Dropout'):
		#	self.d_drop=tf.layers.dropout(self.d[-1],dropout,training=self.training,name="d_drop")
		#	tf.summary.histogram('droupout',self.d_drop)
		with tf.name_scope(self.name+'_Output_Layer'):
			if self.FAKE:
				#self.y_fake=nn.dense_layer(self.d_drop,Readout)
				self.y_fake=tf.layers.dense(self.d[-1],*Readout)
			#	tf.summary.histogram(self.name+'_y_fake', self.y_fake)
				return self.y_fake
			else:
				#self.y_real=nn.dense_layer(self.d_drop,Readout)
				self.y_real=tf.layers.dense(self.d[-1],*Readout)
			#	tf.summary.histogram(self.name+'_y_real', self.y_real)
				return self.y_real
	def generator_dense(self,Dense,BATCH_NORMALISE=True):
		with tf.name_scope(self.name+'_'+Dense[0,4]):
			self.d=[]
			self.drop=[]
			#self.d.append(nn.dense_layer(self.dense_input,*Dense[0]))
			self.d.append(tf.layers.dense(self.dense_input,*Dense[0]))
			self.drop.append(tf.layers.dropout(self.d[0],dropout,training=self.training))
			if BATCH_NORMALISE:
				self.d[0]=tf.layers.batch_normalization(self.drop[0],training=self.training,name='gen_batch_norm_0')
			#tf.summary.histogram('d0',self.d[0])
			
		for i in range(1,len(self.Dense)):
			with tf.name_scope(self.name+'_'+Dense[i,]):
				#self.d.append(nn.dense_layer(self.d[i-1],*Dense[i]))
				self.d.append(tf.layers.dense(self.d[i-1],*Dense[i]))
				self.drop.append(tf.layers.dropout(self.d[i],dropout,training=self.training))
				if BATCH_NORMALISE and i<len(self.Dense):
					self.d[i]=tf.layers.batch_normalization(self.drop[i],training=self.training)
			#	tf.summary.histogram('d'+str(i),self.d[i])			
	def generator_conv_deconv_dense(self,x,Dense,dropout,BATCH_NORMALISE=True):
		with tf.name_scope(self.name+'_'+Dense[0][-2]):
			self.d=[]
			self.drop=[]
			#self.d.append(nn.dense_layer(self.dense_input,*Dense[0]))
			self.d.append(tf.layers.dense(x,*Dense[0]))
			self.drop.append(tf.layers.dropout(self.d[0],dropout,training=self.training))
			if BATCH_NORMALISE:
				self.d[0]=tf.layers.batch_normalization(self.drop[0],training=self.training,name=self.name+'_'+Dense[0][-2]+'_BN')
			#tf.summary.histogram('d0',self.d[0])
			
		for i in range(1,len(Dense)):
			with tf.name_scope(self.name+'_'+Dense[i][-2]):
				#self.d.append(nn.dense_layer(self.d[i-1],*Dense[i]))
				self.d.append(tf.layers.dense(self.d[i-1],*Dense[i]))
				self.drop.append(tf.layers.dropout(self.d[i],dropout,training=self.training))
				if BATCH_NORMALISE:
					self.d[i]=tf.layers.batch_normalization(self.drop[i],training=self.training,name=self.name+'_'+Dense[i][-2]+'_BN')
			#	tf.summary.histogram('d'+str(i),self.d[i])
		self.deconv_input=tf.expand_dims(self.d[-1],axis=2)

	def train(self,LEARNING_RATE):

		with tf.name_scope(self.name+'_Loss'):
			self.loss = tf.reduce_mean(tf.squared_difference(self.y_est,self.y),name=self.name+'_loss')
		with tf.name_scope(self.name+'_Train'):
			self.train_op=tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.loss)	
	def gen_train(self,LEARNING_RATE,y_fake):
		#y_fake=tf.placeholder(tf.float32,shape=[None],name="y_fake")
		self.g_vars=[var for var in tf.trainable_variables() if 'gen' in var.name]
		#pprint(self.g_vars)
		#print('/n',len(self.g_vars),'/n')
		with tf.name_scope(self.name+'_Loss'):
			self.gen_loss = -tf.reduce_mean(tf.log(y_fake),name=self.name+'_loss')
			self.gen_loss = tf.contrib.gan.losses.wargs.wasserstein_generator_loss(y_fake)
		with tf.name_scope(self.name+'_Train'):
			self.gen_train_op=tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.gen_loss,var_list=self.g_vars)
	def disc_train(self,LEARNING_RATE,y_real,y_fake):
		self.d_vars=[var for var in tf.trainable_variables() if 'disc' in var.name]
		#pprint(self.d_vars)
		#print('/n',len(self.d_vars),'/n')
		with tf.name_scope(self.name+'_Loss'):
			self.disc_loss = -tf.reduce_mean(tf.log(y_real)+tf.log(1.-y_fake),name=self.name+'_loss')
			self.disc_loss = tf.contrib.gan.losses.wargs.wasserstein_discriminator_loss(self.y_real,self.y_fake)
		with tf.name_scope(self.name+'_Train'):
			self.disc_train_op=tf.train.AdamOptimizer(LEARNING_RATE).minimize(self.disc_loss,var_list=self.d_vars)
			
def build_disc(discriminator,D_Conv_shape,D_Dense,Readout,DROPOUT=0.5,FAKE=False,xf=None,zf=None,reuse=None):
	discriminator.disc_init(FAKE,xf,zf)
	for i in range(len(D_Conv_shape)):
		D_Conv_shape[i][-1]=reuse
	for i in range(len(D_Dense)):
		D_Dense[i][-1]=reuse
	Readout[-1]=reuse
	discriminator.disc_1d_conv(D_Conv_shape)
	y=discriminator.disc_dense(discriminator.dense_input,D_Dense,Readout,DROPOUT)
	return y

def build_gen(generator,G_Conv_shape,G_Dense,DROPOUT,G_Deconv_shape,G_Deconv_out):
	generator.generator_1d_conv(generator.x,G_Conv_shape)
	generator.generator_conv_deconv_dense(generator.generator_conv_out,G_Dense,DROPOUT)
	xf,zf=generator.generator_1d_deconv(generator.deconv_input,G_Deconv_shape,G_Deconv_out)
	return xf,zf

