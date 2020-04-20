#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 16:27:16 2019

main program for calculating correlations
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
# import scipy as sp
# import scipy.special
# import numpy.ma as ma
# import bisect
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# =============================================================================

if sys.argv[0] is not None:
    ProgramRoot = os.path.dirname(sys.argv[0]) + '/'
else:
    ProgramRoot = '/Users/matteo/Documents/Uni/TESI/_python/ProgrammaMatteo/'

print('Program root: ' + str(ProgramRoot) + 'functions.py')


exec(open(str(ProgramRoot) + 'var_names.py').read())
exec(open(str(ProgramRoot) + 'functions.py').read())

#initialize variables and dictionaries
exec(open(str(ProgramRoot) + 'config/config_v_heatmap.py').read())

print('\nANGLE #1: Loading correlations...')

#G = CalculateCorrelation(MIfile_handle, MI_info, Analysis_info)

#instead of calculating correlations again is much better if you calcuate the heatmap from correlations
#form a file, it is much faster

G_h = []

for i in range(Analysis_info[K_AN_NUM_LAGS]):
    
    filename = ('Correlation_map_d%s_h.dat') % Analysis_info[K_AN_LAG_LIST][i]
    
    bytes_to_read = ((Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i]) * \
                     Analysis_info[K_AN_ROI_LINES_NUM] * Analysis_info[K_AN_ROI_per_line])

    struct_format = ('%sf') % bytes_to_read
    
    corr_array = Load_from_file(CorrFolder, filename, struct_format, pix_depth=4)
    
    corr_reshaped = corr_array.reshape(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i], \
                                       Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line])

    G_h.append(corr_reshaped)



print('ANGLE #1: Correlations loaded succesfully!')

dr_g = g_dr_relation()

print('ANGLE #1: Calculating heatmap...')

theta_f = 26.56*np.pi/180
theta_b = np.pi-theta_f
alpha_f = theta_b/2
alpha_b = theta_f/2
      
k = (2*np.pi)/0.661
q = 2 * k * np.sin(theta_f/2) * np.cos(alpha_f)

X, X_err    = Calculate_v_heatmap(G_h, Analysis_info, err = True)

if Analysis_info[K_AN_NL_FIT]:
    
    print('ANGLE #1: Calculating non-linear fit heatmap...')
    X_fit = Calculate_v_heatmap2(G_h, Analysis_info, q, seed = Analysis_info[K_AN_USE_SEED], v0 = X)



G_v = []

print('\nANGLE #2: Loading correlations...')


for i in range(Analysis_info[K_AN_NUM_LAGS]):
    
    filename = ('Correlation_map_d%s_v.dat') % Analysis_info[K_AN_LAG_LIST][i]
    
    bytes_to_read = ((Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i]) * Analysis_info[K_AN_ROI_LINES_NUM] * Analysis_info[K_AN_ROI_per_line])

    struct_format = ('%sf') % bytes_to_read
    
    corr_array = Load_from_file(CorrFolder, filename, struct_format, pix_depth=4)
    
    corr_reshaped = corr_array.reshape(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i], Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line])

    G_v.append(corr_reshaped)
    
print('ANGLE #2: Correlations loaded succesfully!')

dr_g = g_dr_relation()

print('ANGLE #2: Calculating heatmap...')

Y, Y_err     = Calculate_v_heatmap(G_v, Analysis_info, err = True)



if Analysis_info[K_AN_NL_FIT]:  
    print('ANGLE #1: Calculating non-linear fit heatmap...')
    Y_fit = Calculate_v_heatmap2(G_v, Analysis_info, q, seed = Analysis_info[K_AN_USE_SEED], v0 = Y)



v_heatmap     = np.sqrt(abs(np.square(X) + np.square(Y)))
ang_heatmap = angle_heatmap(X, Y, theta_0 = 0)



if Analysis_info[K_AN_NL_FIT]:
    v_heatmap_fit = np.sqrt(abs(np.square(X_fit) + np.square(Y_fit)))
    ang_heatmap_fit = angle_heatmap(X_fit, Y_fit, theta_0 = 0)



print('\nHeatmap calculated succesfully!')

print('\nWriting files...')


WriteFile(X, 'f', CorrFolder, 'v_x.dat')
WriteFile(Y, 'f', CorrFolder, 'v_y.dat')
WriteFile(X_err, 'f', CorrFolder, 'v_x_err.dat')
WriteFile(Y_err, 'f', CorrFolder, 'v_y_err.dat')
WriteFile(v_heatmap, 'f', CorrFolder, 'v_heatmap.dat')
WriteFile(ang_heatmap, 'f', CorrFolder, 'ang_heatmap.dat')


if Analysis_info[K_AN_NL_FIT]:
    WriteFile(X_fit, 'f', CorrFolder, 'FIT_v_x.dat')
    WriteFile(Y_fit, 'f', CorrFolder, 'FIT_v_y.dat')
    WriteFile(v_heatmap_fit, 'f', CorrFolder, 'FIT_v_heatmap.dat')
    WriteFile(ang_heatmap_fit, 'f', CorrFolder, 'FIT_ang_heatmap.dat')


print('\nFiles completed!')






