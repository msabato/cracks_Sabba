#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:41:40 2020

Correlation map with moving average and average calcualted with gaussian weight
I use indexes to do that

@author: matteo
"""

raw_filename = '/Users/matteo/Documents/Uni/TESI/_data/light_scattering/191011_Sample54/Part1/MI0001_h.dat'

ROI = [640,0,1279,511] #[top_left, bottom_right]

ROI_size = [640, 512]

img_size = [1280,1024]

pix_per_img = img_size[0]*img_size[1]

start_idx = 288

img_num = 11

ROIs = np.empty([img_num, ROI_size[1], ROI_size[0]])

for i in range(img_num):

    temp = Load_single_img(raw_filename, img_size, data_format='B', pix_depth = 1, header_size = 0, image_pos = start_idx+i+1, gap = 0)
    ROIs[i] = temp[0:512, 640:1280]
    
lag_list = [1,2,4,6,8,10]

x = np.asarray(range(0, ROI_size[0]))
y = np.asarray(range(0, ROI_size[1]))

grid    = np.meshgrid(x,y)
weights = np.empty([ROI_size[1], ROI_size[0]])

sigma = 2.5

G = []

OutFolder = '/Users/matteo/Documents/Uni/TESI/_analysis/Sample_54/Fwd/mv_avg/'
for i in range(len(lag_list)):
    
    G.append(np.empty([img_num-lag_list[i], ROI_size[1], ROI_size[0]]))
    
    for t in range(img_num-lag_list[i]):
        for j in range(ROI_size[1]):
            for k in range(ROI_size[0]):
                weights = np.exp(np.divide(np.square(grid[0]-k)+np.square(grid[1]-j),-np.square(sigma)))                
                idx = np.where(weights>0.1)
                
                temp_I_mean = nan_weighted_avg(ROIs[i][idx], weights[idx], err_bool = False)
                temp_I_sq   = nan_weighted_avg(np.square(ROIs[i][idx]), weights[idx], err_bool = False)
                temp_d0     = temp_I_sq/np.square(temp_I_mean) - 1
            
                G[i][t][j][k]      = nan_weighted_avg(np.multiply(ROIs[i][idx], ROIs[i+lag_list[i]][idx]), weights[idx], err_bool = False)/np.square(temp_I_mean)-1
                
        print('i='+str(i)+',\t'+'t='+str(t))
    
    filename = ('mv_corr_map_d%s.dat') % lag_list[i]   
    
    WriteFile(G[i], 'f', OutFolder, filename, overwrite=True)
    
#prova di GitHub
            
    

    

