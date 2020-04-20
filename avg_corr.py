#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:37:11 2020

@author: matteo
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 15:54:55 2019

@author: Matteo Sabato - msabato@g.harvard.edu

Program for calculating correlations for two different scattering angles
"""


import sys
import os
import numpy as np
import re
import math
import struct
from PIL import Image
import cv2
import warnings

import scipy as sp
import scipy.special as ssp
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import cmath
import bisect
from shutil import copyfile

# =============================================================================
# from os import walk
# import shutil
# from time import clock, localtime, strftime
# import numpy.ma as ma
# import matplotlib.animation as animation
# =============================================================================


ProgramRoot = os.path.dirname(sys.argv[0]) + '/'
#ProgramRoot = '/Users/matteo/Documents/Uni/TESI/_python/ProgrammaMatteo/'

print('Program root: ' + str(ProgramRoot))

#Initialize variables, functions and dictionaries
exec(open(str(ProgramRoot) + 'var_names.py').read())
exec(open(str(ProgramRoot) + 'functions.py').read())
exec(open(str(ProgramRoot) + 'config/config_calibration.py').read())

G = np.zeros(Analysis_info[K_AN_NUM_LAGS])

for i in range(Analysis_info[K_AN_NUM_LAGS]):
    
    filename = ('Correlation_map_d%s_h.dat') % Analysis_info[K_AN_LAG_LIST][i]
    
#    filename = ('non_aff_d%s_h.dat') % Analysis_info[K_AN_LAG_LIST][i]
    
    bytes_to_read = ((Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i]) * Analysis_info[K_AN_ROI_LINES_NUM] * Analysis_info[K_AN_ROI_per_line])

    struct_format = ('%sf') % bytes_to_read

    corr_array = (Load_from_file(CorrFolder, filename, struct_format)) 
    
    corr_reshaped = corr_array.reshape(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i], Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line])
    
    channel = np.zeros([Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i],Analysis_info[K_AOI_SIZE][1],Analysis_info[K_AOI_SIZE][0]])
    
    for j in range(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i]-1):
        
        channel[j]     = corr_reshaped[j][Analysis_info[K_AOI_POS][1]:Analysis_info[K_AOI_POS][1]+Analysis_info[K_AOI_SIZE][1], Analysis_info[K_AOI_POS][0]:Analysis_info[K_AOI_POS][0]+Analysis_info[K_AOI_SIZE][0]]

    G[i] = np.nanmean(channel)
    
print('Done!')