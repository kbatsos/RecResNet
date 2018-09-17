from __future__ import division
import numpy as np
import scipy
from sklearn.feature_extraction import image
from PIL import Image

import matplotlib.pyplot as plt

import os
import sys


sys.path.insert(0,'../../pylibs')

import pfmutil as pfm

freiburg_35mm_forward_fast = '/media/kbatsos/Data2/datasets/Freiburg/disparity/35mm_focallength/scene_forwards/fast/left/'
freiburg_35mm_forward_slow = '/media/kbatsos/Data2/datasets/Freiburg/disparity/35mm_focallength/scene_forwards/slow/left/'
freiburg_35mm_backward_fast = '/media/kbatsos/Data2/datasets/Freiburg/disparity/35mm_focallength/scene_backwards/fast/left/'
freiburg_35mm_backward_slow = '/media/kbatsos/Data2/datasets/Freiburg/disparity/35mm_focallength/scene_backwards/slow/left/'

freiburg_15mm_forward_fast = '/media/kbatsos/Data2/datasets/Freiburg/disparity/15mm_focallength/scene_forwards/fast/left/'
freiburg_15mm_forward_slow = '/media/kbatsos/Data2/datasets/Freiburg/disparity/15mm_focallength/scene_forwards/slow/left/'
freiburg_15mm_backward_fast = '/media/kbatsos/Data2/datasets/Freiburg/disparity/15mm_focallength/scene_backwards/fast/left/'
freiburg_15mm_backward_slow = '/media/kbatsos/Data2/datasets/Freiburg/disparity/15mm_focallength/scene_backwards/slow/left/'

set_search = freiburg_15mm_backward_slow

freiburg_sets = os.listdir(set_search)

max_d = 0;



for set_n in freiburg_sets:
	disp = pfm.load(set_search+set_n)
	dmax = np.max(disp[0])
	print "##################### set " + set_n + " ############################" 
	print dmax
	if max_d < dmax:
		max_d = dmax

print max_d
print len(freiburg_sets)
