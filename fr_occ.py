from __future__ import division

import numpy as np 
import matplotlib.pyplot as plt

import sys
import math
import random
import os

sys.path.insert(0,'../pylibs')
sys.path.insert(0,'../src')

import cpputils 
import pfmutil as pfm

l_gt_p = "..../Freiburg/driving/disparity/15mm_focallength/scene_forwards/slow/left/"
r_gt_p =  "..../Freiburg/driving/disparity/15mm_focallength/scene_forwards/slow/right/"
save_p =  ".../Freiburg/driving/disparity/15mm_focallength/scene_forwards/slow/left_nonocc/"

ims =os.listdir(l_gt_p)

for im in ims:
	l_gt = pfm.load(l_gt_p+im)[0]
	r_gt = pfm.load(r_gt_p+im)[0]
	occ = cpputils.make_occ( l_gt,r_gt )
	pfm.save(save_p+im,occ)
