#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 16:46:08 2019

@author: matteo
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
from scipy import ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pyplot import figure
import cmath
import bisect

# =============================================================================
# from os import walk
# import shutil
# from time import clock, localtime, strftime
# import numpy.ma as ma
# =============================================================================


#ProgramRoot = os.path.dirname(sys.argv[0]) + '/'
ProgramRoot = '/Users/matteo/Documents/Uni/TESI/_python/ProgrammaMatteo/'

print('Program root: ' + str(ProgramRoot))


exec(open(str(ProgramRoot) + 'var_names.py').read())
exec(open(str(ProgramRoot) + 'functions.py').read())

#initialize variables and dictionaries
exec(open(str(ProgramRoot) + 'config/config_plastic_def.py').read())

print('\nLoading correlations from file...')

G_forward = []

for i in range(Analysis_info[K_AN_NUM_LAGS]):
    
    filename = ('Correlation_map_d%s_h.dat') % Analysis_info[K_AN_LAG_LIST][i]
    
    bytes_to_read = ((Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i]) * Analysis_info[K_AN_ROI_LINES_NUM] * Analysis_info[K_AN_ROI_per_line])

    struct_format = ('%sf') % bytes_to_read
    
    corr_array = Load_from_file(CorrFolder + 'forward/', filename, struct_format)
    
    corr_reshaped = corr_array.reshape(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i], Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line])

    G_forward.append(corr_reshaped)
    

G_back = []

G_back_tr = []



for i in range(Analysis_info[K_AN_NUM_LAGS]):
    
    filename = ('non_aff_d%s_h.dat') % Analysis_info[K_AN_LAG_LIST][i]
    
    bytes_to_read = ((Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i]) * Analysis_info[K_AN_ROI_LINES_NUM] * Analysis_info[K_AN_ROI_per_line])

    struct_format = ('%sf') % bytes_to_read
    
    corr_array = Load_from_file(CorrFolder + 'back/', filename, struct_format)
    
    corr_reshaped = corr_array.reshape(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i], Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line])

    G_back_tr.append(np.zeros([Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i], Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line]]))
    
    G_back.append(corr_reshaped)
    
print('\nApplying non-affine transformation...')


#non-affine transformation to superimpose forward and back scattering    
for i in range(Analysis_info[K_AN_NUM_LAGS]):
    for j in range(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i]):
        G_back_tr[i][j] = ImgAffineTransform(G_back[i][j])
        
#print('\nWriting files...')

overwrite_all = False

#for i in range(Analysis_info[K_AN_NUM_LAGS]):
#    
#    filename = ('non_aff_d%s_v.dat') % Analysis_info[K_AN_LAG_LIST][i]
#    
#    if overwrite_all == False:
#        overwrite_all = Check_overwrite(CorrFolder + filename)
#    
#    WriteFile(G_back_tr[i], 'f', CorrFolder, filename)
    


        
pl_def = []

print('\nCalculating plastic deformation...')

for i in range(Analysis_info[K_AN_NUM_LAGS]):
    pl_def.append(np.log(G_forward[i]/G_back_tr[i]))
    
    
print('\nWriting files...')

for i in range(Analysis_info[K_AN_NUM_LAGS]):
    
    filename = ('plastic_def_d%s_h.dat') % Analysis_info[K_AN_LAG_LIST][i]
    
    if overwrite_all == False:
        overwrite_all = Check_overwrite(CorrFolder + filename)
    
    WriteFile(pl_def[i], 'f', CorrFolder, filename)
    

print('\nAnalysis completed sucesfully!')
    


    
    
    
    
    
    