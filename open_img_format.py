#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:27:06 2019

python routine for reading image files with custom header and gap between images

@author: matteo
"""
import sys
import os
import numpy as np
import re
import struct
import warnings


ProgramRoot = os.path.dirname(sys.argv[0]) + '/'
#ProgramRoot = '/Users/matteo/Documents/Uni/TESI/_python/ProgrammaMatteo/'

print('Program root: ' + str(ProgramRoot))


exec(open(str(ProgramRoot) + 'var_names.py').read())
exec(open(str(ProgramRoot) + 'functions.py').read())

exec(open(str(ProgramRoot) + 'config/config_img_format.py').read())


output_handle = open(OutFolder + out_filename, 'ab')

bytes_to_read = img_info[IMG_WIDTH]*img_info[IMG_HEIGHT]*img_info[IMG_PIXEL_DEPTH]   

#If there is more than one file I need to iterate over number of files
if img_info[IMG_BOOL_MULTI]: 
    
    for i in range(img_info[IMG_FILES_NUM]):
    
        filename = (img_info[IMG_FILENAME] + '%s' + img_info[IMG_EXT]) % str(img_info[IMG_START_FILE]+i)
        Check_Path(InputFolder + filename)
        input_handle = open(InputFolder + filename, 'rb')    
    
        for j in range(img_info[IMG_TO_READ]):
        
            start_pos = int(img_info[IMG_HEADER_SIZE] + (img_info[IMG_PIXELS_PER_IMAGE]*img_info[IMG_PIXEL_DEPTH]+img_info[IMG_GAP])*(img_info[IMG_START_IDX]-1+j*(img_info[IMG_SKIP]+1)))      
            input_handle.seek(start_pos)      
            bytes_read = input_handle.read(bytes_to_read)      
            output_handle.write(bytes_read)
            
        print('{:.1%}'.format((i+1)/img_info[IMG_FILES_NUM]))

        
 
else:
     
     filename = img_info[IMG_FILENAME] + img_info[IMG_EXT]  
     Check_Path(InputFolder + filename)      
     input_handle = open(InputFolder + filename, 'rb')  
     
     for j in range(img_info[IMG_TO_READ]):
        
            start_pos = int(img_info[IMG_HEADER_SIZE] + (img_info[IMG_PIXELS_PER_IMAGE]*img_info[IMG_PIXEL_DEPTH]+img_info[IMG_GAP])*(img_info[IMG_START_IDX]-1+j*(img_info[IMG_SKIP]+1)))      
            input_handle.seek(start_pos)      
            bytes_read = input_handle.read(bytes_to_read)      
            output_handle.write(bytes_read)
     
     

                

