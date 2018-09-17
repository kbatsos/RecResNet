from __future__ import division
import numpy as np
import scipy
from sklearn.feature_extraction import image
from PIL import Image

import matplotlib.pyplot as plt

import os
import sys

def get_dispaity(filename):
	im = scipy.misc.imread(filename)
	d_r = im[:,:,0].astype('float64')
	d_g = im[:,:,1].astype('float64')
	d_b = im[:,:,2].astype('float64')
	disp = d_r*4 + d_g/(2**6) + d_b/(2**14)
	return disp

def disparity_read(filename):
    """ Return disparity read from filename. """
    f_in = np.array(Image.open(filename))
    d_r = f_in[:,:,0].astype('float64')
    d_g = f_in[:,:,1].astype('float64')
    d_b = f_in[:,:,2].astype('float64')

    depth = d_r * 4 + d_g / (2**6) + d_b / (2**14)
    return depth	

sintel_disp_path="/media/kbatsos/Data2/datasets/Sintel/MPI-Sintel-stereo-training-20150305/training/disparities/"

sintel_sets = os.listdir(sintel_disp_path)

max_disp=0

for setname in sintel_sets:
	depth_files = os.listdir(sintel_disp_path+setname+"/")
	for depth_file in depth_files:
		disp = disparity_read(sintel_disp_path+setname+"/"+depth_file)
		d_m = np.max(disp)
		print "######################" + sintel_disp_path+setname+"/"+depth_file + "################################"
		print d_m
		if max_disp < d_m:
			max_disp = d_m
			print "##### change in disp ####"
			print max_disp  


print max_disp			