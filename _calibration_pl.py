#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 09:52:54 2019

@author: matteo

Routine for DSH calibration - plastic activity
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
from scipy import constants
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

pl_act = np.zeros(Analysis_info[K_AN_NUM_LAGS])

print('Loading data from: ' + str(CorrFolder))

print('Averaging plastic activity...')

for i in range(Analysis_info[K_AN_NUM_LAGS]):
    
    filename = ('plastic_def_d%s.dat') % Analysis_info[K_AN_LAG_LIST][i]
    
    bytes_to_read = ((Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i]) * Analysis_info[K_AN_ROI_LINES_NUM] * Analysis_info[K_AN_ROI_per_line])

    struct_format = ('%sf') % bytes_to_read
    
    pl_array = Load_from_file(CorrFolder, filename, struct_format)
    
    pl_reshaped = pl_array.reshape(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i], Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line])
    
    channel = np.zeros([Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i],Analysis_info[K_AOI_SIZE][1],Analysis_info[K_AOI_SIZE][0]])
    
    for j in range(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i]):
        channel[j] = pl_reshaped[j][Analysis_info[K_AOI_POS][1]:Analysis_info[K_AOI_POS][1]+Analysis_info[K_AOI_SIZE][1], Analysis_info[K_AOI_POS][0]:Analysis_info[K_AOI_POS][0]+Analysis_info[K_AOI_SIZE][0]]
        
    pl_act[i] = np.nanmean(channel, dtype = np.longdouble)
    print(filename + ': ' + str(pl_act[i]))


print('Creating plot...')

T   = 293
eta = 0.32
err_eta = 0.05
r   = 0.5e-6

theta_f = 26.56*np.pi/180
theta_b = np.pi-theta_f
alpha_f = theta_b/2
alpha_b = theta_f/2

D1   = sp.constants.k*T/(6*np.pi*(eta+err_eta)*r)
D2  = sp.constants.k*T/(6*np.pi*(eta-err_eta)*r)

k = (2*np.pi)/0.661

q_f = 2 * k * np.sin(theta_f/2) * np.cos(alpha_f)
q_b = 2 * k * np.sin(theta_b/2) * np.cos(alpha_b)
Q   = np.square(q_b)-np.square(q_f)
fps = MI_info[K_MI_ACQUISITION_FPS]

figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
            
        
plt.loglog(Analysis_info[K_AN_LAG_LIST]/fps, 3*pl_act/Q, '*', label = 'pl_act(t)', color = 'b')           
plt.loglog(Analysis_info[K_AN_LAG_LIST]/fps, (2e12*D1*Analysis_info[K_AN_LAG_LIST]/fps)/3, '--', label = 'expected', color = 'r')
plt.loglog(Analysis_info[K_AN_LAG_LIST]/fps, (2e12*D2*Analysis_info[K_AN_LAG_LIST]/fps)/3, '--', label = 'expected', color = 'r')
        
plt.legend(loc= 'best')
plt.xlabel('t [s]')
plt.ylabel('sigma^2 [um^2].')
    
plt.savefig(CorrFolder + 'PL_act_plot.png')

print('Plot saved!')
    