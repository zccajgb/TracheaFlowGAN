from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import tensorflow as tf
import argparse
import sys
from PIL import Image
parser = argparse.ArgumentParser()
parser.parse_args()

def my_distribution(min_val, max_val, mean, std):
	"""Defines scaled beta distribution from which random data can be generated from
	ARGS:
	min_val = minimum value of distribution
	max_val = maximum value of distribution
	mean = mean value of distribution
	std = standard deviation of distribution"""
	scale = max_val - min_val
	location = min_val
	# Mean and standard deviation of the unscaled beta distribution
	unscaled_mean = (mean - min_val) / scale
	unscaled_var = (std / scale) ** 2
	# Computation of alpha and beta can be derived from mean and variance formulas
	t = unscaled_mean / (1 - unscaled_mean)
	beta = ((t / unscaled_var) - (t * t) - (2 * t) - 1) / ((t * t * t) + (3 * t * t) + (3 * t) + 1)
	alpha = beta * t
	# Not all parameters may produce a valid distribution
	if alpha <= 0 or beta <= 0:
		raise ValueError('Cannot create distribution for the given parameters.')
	# Make scaled beta distribution with computed parameters
	return scipy.stats.beta(alpha, beta, scale=scale, loc=location)

def trachea(Num_trachea,Num_points,poly_order=10):

	mydist=my_distribution(min_val=1.55,max_val=34.57,mean=9.87,std=3.1)
	lengthdist=my_distribution(min_val=6.4597,max_val=11.887,mean=9.318,std=1.066)
	length=lengthdist.rvs(size=(Num_trachea)) #randomly generate length of trachea
	#feature_loc=np.random.normal(0,0.5,(Num_trachea,poly_order))*length #randomly generate location of features
	features=mydist.rvs(size=(Num_trachea,poly_order*4))#randomly generate features
	
	A,x=np.zeros([Num_trachea,Num_points]),np.zeros([Num_trachea,Num_points])	
	for i in range(Num_trachea):
		x[i,:]=np.linspace(0,length[i],Num_points)
		feature_loc=np.linspace(0,length[i],poly_order*4)
		z=(np.polyfit(feature_loc,features[i,:],poly_order))
		p=np.poly1d(z)
		A[i,:]=p(x[i,:])
	return A, x

def bernoulli(Num_trachea,Num_points,poly_order=10):
	""" Uses the beroullli relation to create P and A
	ARGS: 
	Num_trachea = Number of tracheas to be simulatated
	Num_points = Number of points in discretsation of trachea
	poly_order = order of polynomial used to simulate trachea
	"""
	
	rho=1.225e-6
#	P0=101325
	P0=0
	Q2=250
	mu=1.86e-7
	st_dev=3.1
	minimum_area=1.55
	mn=9.87
	mydist=my_distribution(min_val=1.55,max_val=34.57,mean=9.87,std=st_dev)
	A,x = trachea(Num_trachea,Num_points+10,poly_order=poly_order)#returns A[Num_trachea,Num_points]
	#L=x[:,1]-x[:,0]
	L=9.318/Num_points
	ind=5
	for i in range(ind):
		A[:,i]=A[:,ind]
		A[:,-i]=A[:,-ind]
	A[:,0]=A[:,ind]	
	A=A[:,5:-5]
	dP,dP_jet,Pb,P,V=np.zeros(A.shape),np.zeros(A.shape),np.zeros(A.shape),np.zeros(A.shape),np.zeros(A.shape)
#	A=9.87*np.ones(A.shape)
	Q=Q2*np.ones(A.shape)
	V[:,0]=Q[:,0]/A[:,0]
	Pb[:,0]=-rho/2*V[:,0]**2+P0
	P[:,0]=Pb[:,0]
	#dP_jet1=np.zeros(A.shape)
	for i in range(1,A.shape[1]):
		if i+1<A.shape[1]:
			ind2=np.nonzero(A[:,i]<A[:,i-1])
			dP_jet[ind2,i]=0.5*rho*(Q[ind2,i-1]/A[ind2,i-1])*(((A[ind2,i-1]*A[ind2,i-1])/(A[ind2,i]*A[ind2,i+1]))-2*((A[ind2,i-1]/A[ind2,i]))**2-2*(A[ind2,i-1]/A[ind2,i])+1)

		V[:,i]=(Q[:,i-1]/A[:,i])
		dP[:,i]=8*mu*Q[:,i-1]*L*np.pi/(A[:,i]**2)
		Pb[:,i]=-rho/2*V[:,i]**2+P0
		ind=np.nonzero(A[:,i]>A[:,i-1])
		dP_jet[ind,i]=rho*(Q[ind,i-1]/A[ind,i-1])*((A[ind,i-1]/A[ind,i])**2-(A[ind,i-1]/A[ind,i]))
		P[:,i]=Pb[:,i]-dP[:,i]+dP_jet[:,i]
		#P[:,i]=Pb[:,i]-dP[:,i]
		Q[:,i]=np.sqrt(2*(P0-P[:,i])*(A[:,i]**2)/rho)

	#plt.plot(dP_jet.T)
	#plt.plot(dP_jet1.T)
	#plt.show()
	

	A=(A-np.mean(A))/np.std(A)#standardisation
#	P=(P-np.mean(P))/np.std(P)#standardisation
	#A=A/11.887


	return P, A,x 
def scalar_bernoulli(Num_trachea,Num_points,poly_order=10):
	P,A,x=bernoulli(Num_trachea,Num_points,poly_order)
	P_diff=P[:,1]-P[:,-1]
	P=(P_diff-np.mean(P_diff))/np.std(P_diff)#standardisation
	return P_diff,A,x
def categorisation(Num_trachea,Num_points,poly_order=10):
	P,A,x=scalar_bernoulli(Num_trachea,Num_points,poly_order=10)	
	P_binary=np.zeros(P.shape)
	ind=np.nonzero(P>2.5)
	P_binary[ind]=1
	return P_binary,A,x
	 
def loss_fcn(A,Q,mu,L):
	dP=np.zeros((A.shape))
	for i in range(A.shape[0]):
		dP[i,:]=8*mu*Q*L[i]*np.pi/(A[i,:]**2)
	return dP
def loss_fcn2(A,Q,mu,L):
	dP=8*mu*Q*L*np.pi/(A**2)
	return dP

def jet_loss(A,Q,rho):
	dP_jet=np.zeros((A.shape))
	if A[i+1]>A[i]:
		for i in range(A.shape[0]-1):
			dP_jet[i+1,:]=rho*(Q/A[i])*((A[i]/A[i+1])**2-(A[i]/A[i+1]))
	elif A[i+1]<A[i]:
		for i in range(A.shape[0]-2):
			dP_jet[i+1,:]=0.5*rho*(Q/A[i])*((A[i]*A[i]/(A[i+1]*A[i+2]))-2*((A[i]/A[i+1]))**2-2(A[i]/A[i+1])+1)


def make_image(Num_trachea,Num_points,y_points,poly_order=10):

#	create	matrix
	img=np.zeros((Num_trachea,y_points,Num_points))
	P_final,A,x=bernoulli(Num_trachea,Num_points,poly_order=10)
	for j in range(Num_trachea):
		Amax=y_points//2+(A[j,:]*y_points//2)//2
		Amin=y_points//2-(A[j,:]*y_points//2)//2
		for i in range(Num_points):
			a,b=Amin[i].astype(int),Amax[i].astype(int)
			img[j,a:b,i]=256
	return P_final,img,x


if __name__=='__main__':
	#P_final,img,x=make_image(1,512,256)
	#nam=str(P_final)
	#name='img_'+nam+'.png'
#	print(type(img))
#	print(img.shape)
	#for i in range(2):
	#imag=Image.fromarray(img[0,:,:])
	#imag=imag.convert('RGB')
	#imag.save(name)
	#imag.show()
#	z,x,_=bernoulli(5,32,poly_order=1)
	z,_,_=categorisation(10,32)
	#print(np.mean(z))
	#print(np.median(z))
	#print(x.shape)
	#plt.plot(z)
	#plt.show()
	#plt.plot(x.T)
	#plt.axis([0,max(x),0,max(x)])
	plt.plot(z.T)
	#plt.show()
	#plt.plot(x.T)
	#plt.axis([0,max(z),0,max(z)])
	plt.show()
