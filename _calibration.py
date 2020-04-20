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

open(CorrFolder + 'calibr_parameters.txt', 'a').close()
copyfile(ProgramRoot + 'config/ConfigFile_calibration.txt', CorrFolder + 'calibr_parameters.txt')      


#print('Loading data from: ' + CorrFolder)
#
#print('\nANGLE #1: Calculating averages...')
#
#G_h = []
#
#g_t_h = np.zeros(Analysis_info[K_AN_NUM_LAGS])
#
#
#for i in range(Analysis_info[K_AN_NUM_LAGS]):
#    
#    filename = ('Correlation_map_d%s_h.dat') % Analysis_info[K_AN_LAG_LIST][i]
#    
#    bytes_to_read = ((Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i]) * Analysis_info[K_AN_ROI_LINES_NUM] * Analysis_info[K_AN_ROI_per_line])
#
#    struct_format = ('%sf') % bytes_to_read
#    
#    corr_array = Load_from_file(CorrFolder, filename, struct_format)
#    
#    corr_reshaped = corr_array.reshape(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i], Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line])
#    
#    channel = np.zeros([Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i],Analysis_info[K_AOI_SIZE][1],Analysis_info[K_AOI_SIZE][0]])
#    
#    for j in range(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i]):
#        channel[j] = corr_reshaped[j][Analysis_info[K_AOI_POS][1]:Analysis_info[K_AOI_POS][1]+Analysis_info[K_AOI_SIZE][1], Analysis_info[K_AOI_POS][0]:Analysis_info[K_AOI_POS][0]+Analysis_info[K_AOI_SIZE][0]]
#        
#    g_t_h[i] = np.nanmean(channel)
#    
#    G_h.append(corr_reshaped)
#    
#print('ANGLE #2: Calculating averages...')   
#    
#
#G_v = []
#
#g_t_v = np.zeros(Analysis_info[K_AN_NUM_LAGS])
#
#
#for i in range(Analysis_info[K_AN_NUM_LAGS]):
#    
#    filename = ('Correlation_map_d%s_v.dat') % Analysis_info[K_AN_LAG_LIST][i]
#    
#    bytes_to_read = ((Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i]) * Analysis_info[K_AN_ROI_LINES_NUM] * Analysis_info[K_AN_ROI_per_line])
#
#    struct_format = ('%sf') % bytes_to_read
#    
#    corr_array = Load_from_file(CorrFolder, filename, struct_format)
#    
#    corr_reshaped = corr_array.reshape(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i], Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line])
#    
#    channel = np.zeros([Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i],Analysis_info[K_AOI_SIZE][1],Analysis_info[K_AOI_SIZE][0]])
#    
#    for j in range(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i]):
#        channel[j] = corr_reshaped[j][Analysis_info[K_AOI_POS][1]:Analysis_info[K_AOI_POS][1]+Analysis_info[K_AOI_SIZE][1], Analysis_info[K_AOI_POS][0]:Analysis_info[K_AOI_POS][0]+Analysis_info[K_AOI_SIZE][0]]
#         
#    g_t_v[i] = np.nanmean(channel)
#    
#    G_v.append(corr_reshaped)
#    
#print('Creating plot...')
#
#figure(num=None, figsize=(12, 9), dpi=80, facecolor='w', edgecolor='k')
#            
#        
#plt.plot(Analysis_info[K_AN_LAG_LIST], g_t_h, '*', label = 'G_h(t)', color = 'b')
#plt.plot(Analysis_info[K_AN_LAG_LIST], g_t_v, '*', label = 'G_v(t)', color = 'r')
#            
#        
#plt.legend(loc= 'best')
#plt.xlabel('tau')
#plt.ylabel('a.u.')
#    
#plt.savefig(CorrFolder + 'G(t)_plot.png')
#
#print('Plot saved!')

theta_f = 26.56*np.pi/180
theta_b = np.pi-theta_f
alpha_f = theta_b/2
alpha_b = theta_f/2

k = (2*np.pi)/0.661
q = 2 * k * np.sin(theta_f/2) * np.sin(alpha_f)

speed_conversion = MI_info[K_MI_ACQUISITION_FPS]/q

t = Analysis_info[K_AN_LAG_LIST]/MI_info[K_MI_ACQUISITION_FPS]

print('Averaging velocities...')



filename = ('v_x.dat') 
filename_err = ('v_x_err.dat') 
    
bytes_to_read = ((Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1]) * Analysis_info[K_AN_ROI_LINES_NUM] * Analysis_info[K_AN_ROI_per_line])

struct_format = ('%sf') % bytes_to_read

one_matr = np.ones([Analysis_info[K_AOI_SIZE][1],Analysis_info[K_AOI_SIZE][0]])
    
v_array = (Load_from_file(CorrFolder, filename, struct_format)) * speed_conversion
v_err_array = (Load_from_file(CorrFolder, filename_err, struct_format)) * speed_conversion
    
v_reshaped = v_array.reshape(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1], Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line])
v_err_reshaped = v_err_array.reshape(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1], Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line])
    
channel_x     = np.zeros([Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1],Analysis_info[K_AOI_SIZE][1],Analysis_info[K_AOI_SIZE][0]])
channel_x_err = np.zeros([Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1],Analysis_info[K_AOI_SIZE][1],Analysis_info[K_AOI_SIZE][0]])
weights_x = np.zeros([Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1],Analysis_info[K_AOI_SIZE][1],Analysis_info[K_AOI_SIZE][0]])
v_t_x = np.zeros(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1]-1)
std_t_x = np.zeros(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1]-1)    
v_xt_x   = np.zeros([Analysis_info[K_AOI_SIZE][1], Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1]-1])    

for j in range(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1]-1):
    channel_x[j]     = v_reshaped[j][Analysis_info[K_AOI_POS][1]:Analysis_info[K_AOI_POS][1]+Analysis_info[K_AOI_SIZE][1], Analysis_info[K_AOI_POS][0]:Analysis_info[K_AOI_POS][0]+Analysis_info[K_AOI_SIZE][0]]
    channel_x_err[j] = v_err_reshaped[j][Analysis_info[K_AOI_POS][1]:Analysis_info[K_AOI_POS][1]+Analysis_info[K_AOI_SIZE][1], Analysis_info[K_AOI_POS][0]:Analysis_info[K_AOI_POS][0]+Analysis_info[K_AOI_SIZE][0]]
#    weights_x[j] = np.divide(one_matr, np.square(channel_x_err[j]))
    v_t_x[j]    = np.nanmean(channel_x[j])
    std_t_x[j]  = np.nanstd(channel_x[j])

    for i in range(Analysis_info[K_AOI_SIZE][1]):
        
        v_xt_x[i][j] = np.nanmean(channel_x[j][i])
    
    
    
         
#avg_v_x, err_v_x = nan_weighted_avg(channel_x, weights_x) 
#
#avg_v_x = avg_v_x 
#
#err_v_x = err_v_x 



filename = ('v_y.dat') 
filename_err = ('v_y_err.dat') 
    
bytes_to_read = ((Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1]) * Analysis_info[K_AN_ROI_LINES_NUM] * Analysis_info[K_AN_ROI_per_line])

struct_format = ('%sf') % bytes_to_read
    
v_array = (Load_from_file(CorrFolder, filename, struct_format)) * speed_conversion
v_err_array = (Load_from_file(CorrFolder, filename_err, struct_format)) * speed_conversion
    
v_reshaped = v_array.reshape(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1], Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line])
v_err_reshaped = v_err_array.reshape(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1], Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line])
    
channel_y     = np.zeros([Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1],Analysis_info[K_AOI_SIZE][1],Analysis_info[K_AOI_SIZE][0]])
channel_y_err = np.zeros([Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1],Analysis_info[K_AOI_SIZE][1],Analysis_info[K_AOI_SIZE][0]])
weights_y = np.zeros([Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1],Analysis_info[K_AOI_SIZE][1],Analysis_info[K_AOI_SIZE][0]])
v_t_y = np.zeros(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1]-1)
std_t_y = np.zeros(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1]-1)
v_xt_y   = np.zeros([Analysis_info[K_AOI_SIZE][1], Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1]-1])    

for j in range(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1]-1):
    channel_y[j]     = v_reshaped[j][Analysis_info[K_AOI_POS][1]:Analysis_info[K_AOI_POS][1]+Analysis_info[K_AOI_SIZE][1], Analysis_info[K_AOI_POS][0]:Analysis_info[K_AOI_POS][0]+Analysis_info[K_AOI_SIZE][0]]
    channel_y_err[j] = v_err_reshaped[j][Analysis_info[K_AOI_POS][1]:Analysis_info[K_AOI_POS][1]+Analysis_info[K_AOI_SIZE][1], Analysis_info[K_AOI_POS][0]:Analysis_info[K_AOI_POS][0]+Analysis_info[K_AOI_SIZE][0]]
#    weights_y[j] = np.divide(one_matr, np.square(channel_y_err[j]))
    v_t_y[j]    = np.nanmean(channel_y[j])
    std_t_y[j]  = np.nanstd(channel_y[j])
    
    for i in range(Analysis_info[K_AOI_SIZE][1]):
        
        v_xt_y[i][j] = np.nanmean(channel_y[j][i])

            
#avg_v_y, err_v_y = nan_weighted_avg(channel_y, weights_y)         
#
#avg_v_y = avg_v_y 
#
#err_v_y = err_v_y 

#filename = ('v_heatmap.dat') 
#    
#bytes_to_read = ((Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1]) * Analysis_info[K_AN_ROI_LINES_NUM] * Analysis_info[K_AN_ROI_per_line])
#
#struct_format = ('%sf') % bytes_to_read
#    
#v_array = Load_from_file(CorrFolder, filename, struct_format)
#    
#v_reshaped = v_array.reshape(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][i], Analysis_info[K_AN_ROI_LINES_NUM], Analysis_info[K_AN_ROI_per_line])
#    
#channel = np.zeros([Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1],Analysis_info[K_AOI_SIZE][1],Analysis_info[K_AOI_SIZE][0]])
#    
#for j in range(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1]):
#    channel[j] = v_reshaped[j][Analysis_info[K_AOI_POS][1]:Analysis_info[K_AOI_POS][1]+Analysis_info[K_AOI_SIZE][1], Analysis_info[K_AOI_POS][0]:Analysis_info[K_AOI_POS][0]+Analysis_info[K_AOI_SIZE][0]]
  

#avg_v_x = np.float(sp.optimize.curve_fit(g2m1,q*t,g_t_h)[0])
#avg_v_y = np.float(sp.optimize.curve_fit(g2m1,q*t,g_t_v)[0])

v_x     = np.zeros(Analysis_info[K_AOI_SIZE][1])
std_x   = np.zeros(Analysis_info[K_AOI_SIZE][1])
     
v_t     = np.zeros(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1]-1)     
std_t   = np.zeros(Analysis_info[K_AN_IMGS_NUM] - Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1]-1)     

print('Writing files...')


##########################################################
# TO WRITE AVERAGE V IN EVERY FRAME AS A FUNCTIONOF TIME #
##########################################################

filename = 'v(t).txt'

write_handle = open(CorrFolder + filename, 'w')

write_handle.write('t \t V(t) \t std_t/2 \n')


for i in range(len(v_t)):
    
    v_t[i] = np.sqrt(np.square(v_t_x[i])+np.square(v_t_y[i]))
    std_t[i] = np.sqrt(np.square(std_t_x[i])+np.square(std_t_y[i]))
    write_handle.write(str(i)+ '\t' + '{:.5}'.format(v_t[i]) + '\t' + '{:.5}'.format(std_t[i]/2) + '\n')

write_handle.close()


##########################################################    
# TO WRITE AVERAGE ERROR OVER TIME AS A FUNCTIONOF SPACE #
##########################################################

filename = 'v_err(x).txt'

write_handle = open(CorrFolder + filename, 'w')

write_handle.write('t \t x err \t y err \t err \n')


for i in range(len(v_t)):
    
    err = np.sqrt(np.square(np.nanmean(channel_x_err[i])) + np.square(np.nanmean(channel_y_err[i])))
    
    write_handle.write(str(i)+ '\t' + '{:.5}'.format(np.nanmean(channel_x_err[i])) + '\t' + '{:.5}'.format(np.nanmean(channel_y_err[i])) + '\t' + '{:.5}'.format(err) + '\n')

write_handle.close()


####################################################
# TO WRITE V AS A FUNCTION OF X AVERAGED OVER TIME #
####################################################

filename = 'v(x).txt'

write_handle = open(CorrFolder + filename, 'w')

write_handle.write('t \t V(x) \t std/2 \t v_x(x) \t std_x/2 \t v_y(x) \t std_y/2 \n')

for i in range(len(v_x)):
    
    v_x[i]   = np.sqrt(np.square(np.nanmean(v_xt_x[i])) + np.square(np.nanmean(v_xt_y[i])))
    std_x[i] = np.sqrt(np.square( np.nanstd(v_xt_x[i])) +  np.square(np.nanstd(v_xt_y[i])))
    write_handle.write(str(i)+ '\t' + '{:.5}'.format(v_x[i]) + '\t' + '{:.5}'.format(std_x[i]/2) + '\t' + '{:.5}'.format(np.nanmean(v_xt_x[i])) + '\t' + '{:.5}'.format(np.nanstd(v_xt_x[i])/2) + '\t' + '{:.5}'.format(np.nanmean(v_xt_y[i])) + '\t' + '{:.5}'.format(np.nanstd(v_xt_y[i])/2) + '\n')

write_handle.close()
    

#avg_v = np.sqrt(np.square(avg_v_x) + np.square(avg_v_y))
#
#v_err = np.sqrt(np.square(err_v_x) + np.square(err_v_y))
#
#print('\nAverage v_x = ' + '{:.3}'.format(avg_v_x) + 'um/s')
#print('\nError v_x = ' + '{:.3}'.format(err_v_x) + 'um/s')
#print('\nAverage v_y = ' + '{:.3}'.format(avg_v_y) + 'um/s')
#print('\nError v_y = ' + '{:.3}'.format(err_v_y) + 'um/s')
#print('\nAverage v = ' + '{:.3}'.format(avg_v) + 'um/s')
#print('\nError v= ' + '{:.3}'.format(v_err) + 'um/s')
#    
#

print('File saved!')    
