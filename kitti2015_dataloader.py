from __future__ import division
import tensorflow as tf
import numpy as np

import scipy
from sklearn.feature_extraction import image
import matplotlib.pyplot as plt

import os
import time
import random

class Dataloader(object):

	def __init__(self,params):
		self.__params=params
		self.__contents = os.listdir(self.__params.disp_path)
		self.__contents.sort()
		self.__training_samples=len(self.__contents)
		self.__sample_index=0
		self.epoch=0
		self.maxwidth=0
		self.maxheight=0
		self.configure_input_size()
		self.__init_index=0
		self.__widthresize =self.maxwidth+ (self.__params.down_sample_ratio - self.maxwidth%self.__params.down_sample_ratio)%self.__params.down_sample_ratio
		self.__heightresize =self.maxheight+( self.__params.down_sample_ratio - self.maxheight%self.__params.down_sample_ratio)%self.__params.down_sample_ratio
		self.max_disp=256
		self.__val_num=40


	def init_sample_index(self,val):
		 self.__sample_index=val

	def get_sample_size(self):
		return self.__training_samples

	def get_sample_index(self):
		return self.__sample_index
			
	def get_data_size(self):
		return self.__heightresize,self.__widthresize,2

	def get_training_data_size(self):
		return 256,256,2		

	def configure_input_size(self):
		for i in range(len(self.__contents)):
			img = scipy.misc.imread( self.__params.left_path+self.__contents[i]).astype(float);
			s = img.shape
			if self.maxheight < s[0]:
				self.maxheight = s[0]

			if self.maxwidth < s[1]:
				self.maxwidth = s[1]	

	def fold_1(self):
		self.__training_samples = int(self.__training_samples/2)
		self.__contents = self.__contents[self.__init_index:self.__training_samples]

	def fold_2(self):
		self.__init_index = int(self.__training_samples/2)
		self.__contents = self.__contents[self.__init_index:self.__training_samples]
		self.__training_samples = int(self.__training_samples/2)
				

	def shuffle_data(self):
		millis = int(round(time.time()))
		np.random.seed(millis)
		np.random.shuffle(self.__contents)				

	def load_training_sample(self):
		if self.__sample_index >= self.__training_samples-self.__val_num:
			self.__sample_index=0
			self.epoch+=1
			self.shuffle_data()

		img = scipy.misc.imread( self.__params.left_path+self.__contents[self.__sample_index]).astype(float);
		img = img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114
		disp = scipy.misc.imread( self.__params.disp_path+self.__contents[self.__sample_index]).astype(float)/256;
		gt = scipy.misc.imread( self.__params.gt_path+self.__contents[self.__sample_index]).astype(float)/256;
		gt_noc = scipy.misc.imread( self.__params.gt_path_noc+self.__contents[self.__sample_index]).astype(float)/256;

		s = img.shape

		s = img.shape
		maxheight = s[0]-256
		maxwidth = s[1]-256
		x = random.randint(0,maxheight)
		y = random.randint(0,maxwidth)		
		disp = disp[x:x+256,y:y+256]
		img = img[x:x+256,y:y+256]
		gt = gt[x:x+256,y:y+256]
		gt_noc = gt_noc[x:x+256,y:y+256]		

		data = np.stack([disp,img],axis=2)
		data = np.reshape(data,[1,data.shape[0],data.shape[1],data.shape[2]])
		gt = np.reshape(gt,[1,gt.shape[0],gt.shape[1],1])
		gt_noc = np.reshape(gt_noc,[1,gt_noc.shape[0],gt_noc.shape[1],1])

		self.__sample_index+=1

		return data,gt,gt_noc,self.__sample_index


	def load_validation_sample(self):
		if self.__sample_index >= self.__training_samples:
			self.__sample_index=self.__training_samples-40

		img = scipy.misc.imread( self.__params.left_path+self.__contents[self.__sample_index]).astype(float);
		img = img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114
		disp = scipy.misc.imread( self.__params.disp_path+self.__contents[self.__sample_index]).astype(float)/256;
		gt = scipy.misc.imread( self.__params.gt_path+self.__contents[self.__sample_index]).astype(float)/256;
		gt_noc = scipy.misc.imread( self.__params.gt_path_noc+self.__contents[self.__sample_index]).astype(float)/256;

		s = img.shape
		if s[0] <self.__heightresize:
			padding= self.__heightresize - s[0]
			img = np.lib.pad(img,[(padding,0),(0,0)],'edge')
			disp = np.lib.pad(disp,[(padding,0),(0,0)],'edge')
			gt = np.lib.pad(gt,[(padding,0),(0,0)],'edge')
			gt_noc = np.lib.pad(gt_noc,[(padding,0),(0,0)],'edge')
		if s[1] <self.__widthresize:
			padding= self.__widthresize-s[1]
			img = np.lib.pad(img,[(0,0),(padding,0)],'edge')
			disp = np.lib.pad(disp,[(0,0),(padding,0)],'edge')
			gt = np.lib.pad(gt,[(0,0),(padding,0)],'edge')
			gt_noc = np.lib.pad(gt_noc,[(0,0),(padding,0)],'edge')

		data = np.stack([disp,img],axis=2)
		data = np.reshape(data,[1,data.shape[0],data.shape[1],data.shape[2]])
		gt = np.reshape(gt,[1,gt.shape[0],gt.shape[1],1])
		gt_noc = np.reshape(gt_noc,[1,gt_noc.shape[0],gt_noc.shape[1],1])

		self.__sample_index+=1

		return data,gt,gt_noc,self.__sample_index

	def load_verify_sample(self):
		if self.__sample_index >= self.__training_samples:
			self.__sample_index=self.__training_samples-40

		img = scipy.misc.imread( self.__params.left_path+self.__contents[self.__sample_index]).astype(float);
		img = img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114
		disp = scipy.misc.imread( self.__params.disp_path+self.__contents[self.__sample_index]).astype(float)/256;
		gt = scipy.misc.imread( self.__params.gt_path+self.__contents[self.__sample_index]).astype(float)/256;
		gt_noc = scipy.misc.imread( self.__params.gt_path_noc+self.__contents[self.__sample_index]).astype(float)/256;

		s = img.shape
		height,width= img.shape;
		if s[0] <self.__heightresize:
			padding= self.__heightresize - s[0]
			img = np.lib.pad(img,[(padding,0),(0,0)],'edge')
			disp = np.lib.pad(disp,[(padding,0),(0,0)],'edge')
			gt = np.lib.pad(gt,[(padding,0),(0,0)],'edge')
			gt_noc = np.lib.pad(gt_noc,[(padding,0),(0,0)],'edge')
		if s[1] <self.__widthresize:
			padding= self.__widthresize-s[1]
			img = np.lib.pad(img,[(0,0),(padding,0)],'edge')
			disp = np.lib.pad(disp,[(0,0),(padding,0)],'edge')
			gt = np.lib.pad(gt,[(0,0),(padding,0)],'edge')
			gt_noc = np.lib.pad(gt_noc,[(0,0),(padding,0)],'edge')

		data = np.stack([disp,img],axis=2)
		data = np.reshape(data,[1,data.shape[0],data.shape[1],data.shape[2]])
		gt = np.reshape(gt,[1,gt.shape[0],gt.shape[1],1])
		gt_noc = np.reshape(gt_noc,[1,gt_noc.shape[0],gt_noc.shape[1],1])
		name = self.__contents[self.__sample_index]	
		self.__sample_index+=1

		return data,gt,gt_noc,self.__sample_index,height,width,name		


	def load_test_sample(self):
		if self.__sample_index >= self.__training_samples:
			self.__sample_index=0

		img = scipy.misc.imread( self.__params.left_path+self.__contents[self.__sample_index]).astype(float);
		img = img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114
		disp = scipy.misc.imread( self.__params.disp_path+self.__contents[self.__sample_index]).astype(float)/256;

		height,width= img.shape;

		s = img.shape
		if s[0] <self.__heightresize:
			padding= self.__heightresize - s[0]
			img = np.lib.pad(img,[(padding,0),(0,0)],'edge')
			disp = np.lib.pad(disp,[(padding,0),(0,0)],'edge')
		if s[1] <self.__widthresize:
			padding= self.__widthresize-s[1]
			img = np.lib.pad(img,[(0,0),(padding,0)],'edge')
			disp = np.lib.pad(disp,[(0,0),(padding,0)],'edge')

		data = np.stack([disp,img],axis=2)
		data = np.reshape(data,[1,data.shape[0],data.shape[1],data.shape[2]])

		name = self.__contents[self.__sample_index]	
		self.__sample_index+=1

		return data,self.__sample_index,height,width, name	

