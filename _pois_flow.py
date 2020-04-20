#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 11:59:29 2019

Program to check Poiseuille flow from confocal microscope data

@author: matteo
"""
import sys
import os
import numpy as np
import cv2
from cv2 import matchTemplate
import read_lif

#ProgramRoot = os.path.dirname(sys.argv[0]) + '/'
ProgramRoot = '/Users/matteo/Documents/Uni/TESI/_python/ProgrammaMatteo/'

print('Program root: ' + str(ProgramRoot))


exec(open(str(ProgramRoot) + 'var_names.py').read())
exec(open(str(ProgramRoot) + 'functions.py').read())

#initialize variables and dictionaries
exec(open(str(ProgramRoot) + 'config/config_Pois.py').read())

#loading .lif file

lif_reader = read_lif.Reader(InputFolder + LIF_file_name)


series = lif_reader.getSeries()

flow = []

profile = []

for i in range(LIF_info[K_LIF_SERIES_NUM]):
    
    print('Calculating profile for data sets number %s' % (i+1))
    
    hyperstack = series[LIF_info[K_LIF_SERIES_LIST][i]]
    
    flow_matrix = np.zeros([LIF_info[K_LIF_FRAMES_NUMBER][i]-1, LIF_info[K_LIF_STACK_HEIGHT]-2])
    
    for t in range (LIF_info[K_LIF_FRAMES_NUMBER][i]-1):
        
        t2 = hyperstack.getFrame(T=t+1)
        t2 = np.swapaxes(t2, 1,2)
        t2 = np.swapaxes(t2, 0,1)
        
        t1 = hyperstack.getFrame(T=t)
        t1 = np.swapaxes(t1, 1,2)
        t1 = np.swapaxes(t1, 0,1)
        
        for z in range(LIF_info[K_LIF_STACK_HEIGHT]-2):
            
            x1 = LIF_info[K_LIF_TEMP_POS][0]
            x2 = LIF_info[K_LIF_TEMP_POS][0]+LIF_info[K_LIF_TEMP_SIZE][0]-1
            
            y1 = LIF_info[K_LIF_TEMP_POS][1]
            y2 = LIF_info[K_LIF_TEMP_POS][1]+LIF_info[K_LIF_TEMP_SIZE][1]-1
            
            #t1[z] = remove_noise(t1[z], LIF_info[K_LIF_CUTOFF])
            #t2[z] = remove_noise(t2[z], LIF_info[K_LIF_CUTOFF])
            
            template = t1[z+1][x1:x2, y1:y2]
            
            match_map = cv2.matchTemplate(t2[z+1], template, method=3)
            
            match_pos = cv2.minMaxLoc(match_map)[3]
            
            flow_matrix[t][z] = np.sqrt((match_pos[0]-x1)^2 + (match_pos[1]-y1)^2)*LIF_info[K_LIF_PXL_SIZE]*LIF_info[K_LIF_FPS]
            
            print('{:.1%}'.format(((t*LIF_info[K_LIF_STACK_HEIGHT])+(z+1))/(LIF_info[K_LIF_STACK_HEIGHT]*(LIF_info[K_LIF_FRAMES_NUMBER][i]-1))))
            
        
        
        flow_matrix[t] = flow_matrix[t]/np.max(flow_matrix[t])
    
    flow.append(flow_matrix)
    
    flow_matrix = np.swapaxes(flow_matrix, 0, 1)
    
    temp = np.zeros(LIF_info[K_LIF_STACK_HEIGHT]-2)
    
    for i in range(LIF_info[K_LIF_STACK_HEIGHT]-2):
        
        temp[i] = np.nanmean(flow_matrix[i])
        
    profile.append(temp)
            

    
    
    
    

    