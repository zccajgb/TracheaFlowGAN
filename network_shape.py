from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf

def gen_shape(num_layers=2,filters=32,strides=1,BATCH_SIZE=100,kernel_size=4,Num_points=256):
	G_conv_shape=[]
	G_deconv_shape=[]
	for i in range(num_layers-1):
		G_conv_shape.append([filters,kernel_size,strides,'VALID','channels_last',1,tf.nn.leaky_relu,True,None,tf.zeros_initializer(),None,None,None,None,None,True,"gen_conv"+str(i+1)])
		if i==0:
			G_deconv_shape.append([[kernel_size,filters,1],[BATCH_SIZE,1,filters],strides,'VALID','NWC','gen_Deconv'+str(i+1)])
		else:
			G_deconv_shape.append([[kernel_size,filters,filters],[BATCH_SIZE,1,filters],strides,'VALID','NWC','gen_Deconv'+str(i+1)])
	G_deconv_out=[[kernel_size,1,filters],[BATCH_SIZE,Num_points,1],strides,'VALID','NWC','Generator_Output']
	return G_conv_shape, G_deconv_shape, G_deconv_out
	
def gen_dense_shape(num_layers=1,Units=32):
	G_Dense=[]
	for i in range(num_layers):
		G_Dense.append([Units,tf.nn.leaky_relu,True,None,tf.zeros_initializer(),None,None,None,None,None,True,'gen_Dense_'+str(i+1),None])
	return G_Dense
	
def disc_shape(num_layers=2,filters=32,kernel_size=3,strides=1):
	D_conv_shape=[]
	for i in range(num_layers):
		D_conv_shape.append([filters,kernel_size,strides,'VALID','channels_last',1,tf.nn.leaky_relu,True,None,tf.zeros_initializer(),None,None,None,None,None,True,"disc_conv"+str(i+1),None])
	return D_conv_shape
	
def disc_dense_shape(num_layers=1,Units=32):
	D_Dense=[]
	for i in range(num_layers):
		D_Dense.append([Units,tf.nn.leaky_relu,True,None,tf.zeros_initializer(),None,None,None,None,None,True,'disc_Dense_'+str(i+1),None])
	Readout=[1,tf.nn.sigmoid,True,None,tf.zeros_initializer(),None,None,None,None,None,True,'disc_Readout',None]
	return D_Dense, Readout
