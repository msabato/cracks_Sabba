#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 11:20:35 2019

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
from shutil import copyfile

import scipy as sp
import scipy.special as ssp
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.pyplot import figure
from matplotlib.pyplot import quiver
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


ProgramRoot = os.path.dirname(sys.argv[0]) + '/'

print('Program root: ' + str(ProgramRoot))


exec(open(str(ProgramRoot) + 'var_names.py').read())
exec(open(str(ProgramRoot) + 'functions.py').read())

#initialize variables and dictionaries
exec(open(str(ProgramRoot) + 'config/config_animations.py').read())

open(OutFolder + 'Config_animation.txt', 'a').close()
copyfile(ProgramRoot + 'config/ConfigFile_animations.txt', OutFolder + 'config_ani.txt')

print('Loading frames...')    
bytes_to_read = Ani_config[K_ANI_NUM_FRAMES] * MI_info[K_MI_PIXELS_PER_IMAGE]
struct_format = ('%sB') % bytes_to_read               
Frames_array = Load_from_file(InputFolder, MIfile_name, struct_format, start_idx=MI_info[K_MI_PIXELS_PER_IMAGE]*Ani_config[K_ANI_START_IDX])
Frames = Frames_array.reshape(Ani_config[K_ANI_NUM_FRAMES], MI_info[K_MI_IMAGE_HEIGHT], MI_info[K_MI_IMAGE_WIDTH])

if Ani_config[K_ANI_OVRL_BOOL] is True:
    print('Loading overlays...')
    ovrl_pix_per_image = Ani_config[K_ANI_OVRL_FRAMESIZE][0] * Ani_config[K_ANI_OVRL_FRAMESIZE][1] 
    bytes_to_read = Ani_config[K_ANI_NUM_FRAMES] * ovrl_pix_per_image
    struct_format = ('%sf') % bytes_to_read
    start_byte = ovrl_pix_per_image*(Ani_config[K_ANI_START_IDX]-Analysis_info[K_AN_FIRST_IMG])
    ovrl_array = Load_from_file(OvrlFolder, Ani_config[K_ANI_OVRL_FILENAME], struct_format, \
                                start_idx=start_byte, pix_depth=4)
    ovrl_Frames = ovrl_array.reshape(Ani_config[K_ANI_NUM_FRAMES], Ani_config[K_ANI_OVRL_FRAMESIZE][1], Ani_config[K_ANI_OVRL_FRAMESIZE][0])


AnimateCorrMap(Frames, FileName=Ani_config[K_ANI_FILENAME], Times=None, fLog=None, Title=None, Comment=None, ColorMap=None, OverlayFrames=ovrl_Frames,\
                   Quiver=None, VelocityMaps=None, OverlayCmap=None, BadValue=None, MapShape=None)
                    
    
#    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(Frames), interval=1.0/g_config[K_CORRMAP_VIDEO_FPS], repeat_delay=1000)
#ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(Frames), interval=1.0/Ani_config[K_ANI_FPS], repeat_delay=1000)

#    ExportAnimation(ani, g_config[K_CORRMAP_FOLDER_PATH], FileName, FPS=g_config[K_CORRMAP_VIDEO_FPS], ForceOverwrite=False, Title=Title, Artist=g_config[K_USER_NAME], Comment=Comment)
#ExportAnimation(ani, CorrFolder, Ani_config[K_ANI_FILENAME], FPS=Ani_config[K_ANI_FPS], ForceOverwrite=False, Title=Ani_config[K_ANI_TITLE], Artist=Analysis_info[K_USER_NAME])

