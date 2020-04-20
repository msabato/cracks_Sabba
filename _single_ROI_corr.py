"""
Created on Thu Jul 25 11:54:35 2019

@author: Matteo Sabato - msabato@g.harvard.edu

main program for calculating Strain Map
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

print('Program root: ' + str(ProgramRoot) + 'functions.py')

#Initialize variables, functions and dictionaries
exec(open(str(ProgramRoot) + 'var_names.py').read())
exec(open(str(ProgramRoot) + 'functions.py').read())
exec(open(str(ProgramRoot) + 'config/config_single_ROI_corr.py').read())


print('Loading correlations...')

open(CorrFolder + 'calibr_parameters.txt', 'a').close()
copyfile(ProgramRoot + 'config/ConfigFile_single_ROI_corr.txt', OutFolder + 'config.txt')      

g_t_h = np.zeros([Analysis_info[K_AN_t_0_NUM], Analysis_info[K_AN_NUM_LAGS]])

for i in range(Analysis_info[K_AN_NUM_LAGS]):
    
    filename = ('Correlation_map_d%s_h.dat') % Analysis_info[K_AN_LAG_LIST][i]
    
    print(filename)
    
    bytes_to_read = ((Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i]) * Analysis_info[K_AN_ROI_LINES_NUM] * Analysis_info[K_AN_ROI_per_line])

    struct_format = ('%sf') % bytes_to_read
    
    corr_array = Load_from_file(CorrFolder, filename, struct_format)
    
    corr_reshaped = corr_array.reshape(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i], Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line])
    
    for j in range(Analysis_info[K_AN_t_0_NUM]):
        
        g_t_h[j][i] = corr_reshaped[Analysis_info[K_AN_t_0][j]][Analysis_info[K_AN_ROI_POS][1]][Analysis_info[K_AN_ROI_POS][0]]
    
g_t_v = np.zeros([Analysis_info[K_AN_t_0_NUM], Analysis_info[K_AN_NUM_LAGS]])

for i in range(Analysis_info[K_AN_NUM_LAGS]):
    
    filename = ('Correlation_map_d%s_v.dat') % Analysis_info[K_AN_LAG_LIST][i]
    
    print(filename)
    
    bytes_to_read = ((Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i]) * Analysis_info[K_AN_ROI_LINES_NUM] * Analysis_info[K_AN_ROI_per_line])

    struct_format = ('%sf') % bytes_to_read
    
    corr_array = Load_from_file(CorrFolder, filename, struct_format)
    
    corr_reshaped = corr_array.reshape(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i], Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line])
    
    for j in range(Analysis_info[K_AN_t_0_NUM]):
        
        g_t_v[j][i] = corr_reshaped[Analysis_info[K_AN_t_0][j]][Analysis_info[K_AN_ROI_POS][1]][Analysis_info[K_AN_ROI_POS][0]]
    


print('\nSaving plot...')

multi_plot(g_t_h, Analysis_info, save = True, filename = 'correlation_h', location = 'best')

multi_plot(g_t_v, Analysis_info, save = True, filename = 'correlation_v', location = 'best')

print('\nAnalysis completed!')



