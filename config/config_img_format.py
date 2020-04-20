#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 14:33:01 2019

@author: matteo
"""

f_conf = open(ProgramRoot + 'config/ConfigFile_img_format.txt', 'r')

InputFolder = ReadConfigFile(f_conf)
Check_Path(InputFolder)

OutFolder   = ReadConfigFile(f_conf)
Check_Path(OutFolder)

out_filename = ReadConfigFile(f_conf)

#avoid overwriting old files by accident
file_number = 0

while os.path.isfile(OutFolder+out_filename):
    file_number += 1
    out_filename = str(file_number) + out_filename 
    

#storing information about MI file


img_info = {
                IMG_FILENAME:ReadConfigFile(f_conf),
                IMG_EXT:ReadConfigFile(f_conf),
                IMG_BOOL_MULTI:ReadConfigFile(f_conf),
                IMG_FILES_NUM:ReadConfigFile(f_conf),
                IMG_START_FILE:ReadConfigFile(f_conf),
                IMG_HEADER_SIZE:ReadConfigFile(f_conf),
                IMG_GAP:ReadConfigFile(f_conf),
                IMG_NUM_per_FILE:ReadConfigFile(f_conf),
                IMG_WIDTH:ReadConfigFile(f_conf),
                IMG_HEIGHT:ReadConfigFile(f_conf)
                
            }

                
img_info.update( 
                         {
            
                            IMG_PIXELS_PER_IMAGE:int(img_info[IMG_WIDTH]*img_info[IMG_HEIGHT]),
                            IMG_PIXEL_DEPTH:ReadConfigFile(f_conf),
                            IMG_VAR_FORMAT:ReadConfigFile(f_conf),
                            IMG_START_IDX:ReadConfigFile(f_conf),
                            IMG_LAST_IDX:ReadConfigFile(f_conf),
                            IMG_SKIP:ReadConfigFile(f_conf)
                        
                          }
                    )

img_info.update( 
                         {
            
                            IMG_TO_READ:int((img_info[IMG_LAST_IDX]-img_info[IMG_START_IDX]+1)//(img_info[IMG_SKIP]+1))
                            
                         }
                         
                )