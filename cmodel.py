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
parser = argparse.ArgumentParser()
parser.parse_args()
FLAGS=None

Num_points=256
y_points=64
PO=1
#define function to create data
z_test,x_test,_=dg.bernoulli(5,Num_points,poly_order=PO) 
#x_test=np.expand_dims(x_test,axis=1)
#print(y_test.shape)
loss_vector=[]

export_dir="./"

	
with tf.Session() as sess:
	#RESTORE MODEL
	saver = tf.train.import_meta_graph('./save.meta')
	ckpt = tf.train.get_checkpoint_state(os.path.dirname('./checkpoint'))
	saver.restore(sess, ckpt.model_checkpoint_path)
	graph=tf.get_default_graph()
	#for op in tf.get_default_graph().get_operations():
	#	pprint(str(op.name))
	#LOAD VARIABLES
	dataset_init_op = graph.get_operation_by_name("gIterator/g_dataset_init")
	z_=graph.get_tensor_by_name('dz_:0')
	x_=graph.get_tensor_by_name('gx_:0')
	z_est=graph.get_tensor_by_name('g_Generator_Output/Generator_Output:0')
	batch_size=graph.get_tensor_by_name('gbatch_size:0')
	#b_size=graph.get_tensor_by_name('batch_size:0')
	training=graph.get_tensor_by_name('gtraining:0')
	#loss=graph.get_tensor_by_name('Loss/loss:0')
	
	#RUN SESSION
	sess.run(dataset_init_op,feed_dict={z_:z_test,x_:x_test,batch_size:z_test.shape[0]}) #Initialise Iterator
	#losses=sess.run(loss,feed_dict={training:False})
	z_calc=sess.run(z_est,feed_dict={training:False})

#print(losses)
#y_test=np.reshape(y_test,(-1,1))
#y=np.concatenate((y_calc,y_test,abs(y_calc-y_test)),1)

#print('Estimates \t Labels\t    Difference')
##plt.show()
#print(y)
z_calc=np.squeeze(z_calc)
loss=np.mean((z_test-z_calc)**2)
print(loss)
zc=(z_calc-np.mean(z_calc))/np.std(z_calc)
zt=(z_test-np.mean(z_test))/np.std(z_test)
plt.plot(zc[0])
plt.plot(zt[0])
plt.legend(('Estimates','Labels'))
plt.show()
