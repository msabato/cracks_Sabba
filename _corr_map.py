"""
Created on Tue Jul 16 11:25:58 2019

@author: Matteo Sabato - msabato@g.harvard.edu

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
from shutil import copyfile

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
exec(open(str(ProgramRoot) + 'config/config_corr_map.py').read())

open(OutFolder + 'Config.txt', 'a').close()
copyfile(ProgramRoot + 'config/ConfigFile_corr_map.txt', OutFolder + 'config.txt')

print('Calculating correlations...')

G = CalculateCorrelation(MIfile_handle, MI_info, Analysis_info)

print('\nCorrelations calculated succesfully!')



print('\nWriting .dat files...')

for i in range(Analysis_info[K_AN_NUM_LAGS]):
    
    filename = ('Correlation_map_d%s.dat') % Analysis_info[K_AN_LAG_LIST][i]
    
    WriteFile(G[i], 'f', OutFolder, filename)


print('\nFiles completed!')

#print('\nCreating videos...')
    
#for i in range(Analysis_info[K_AN_NUM_LAGS]):
    
#    filename = ('Correlation_map_d%s.avi') % Analysis_info[K_AN_LAG_LIST][i]
    
#    CreateVideo(G[i], Analysis_info[K_AN_VIDEO_FPS], filename, Analysis_info[K_AN_MASK])

