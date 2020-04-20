#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 14:49:35 2020

@author: matteo
"""

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




def AnimateCorrMap(Frames, FileName=None, Times=None, fLog=None, Title=None, Comment=None, ColorMap=None, OverlayFrames=None,\
                   Quiver=None, VelocityMaps=None, OverlayCmap=None, BadValue=None, MapShape=None):

   
    if Times is None:
        
        if Analysis_info[K_AN_IMGS_NUM] is not None:
            Times = np.asarray(range(0, len(Frames)))/int(MI_info[K_MI_ACQUISITION_FPS])
            
        elif Ani_config[K_ANI_TIME_BOOL] == True:
            raise IOError("Timestamps or acquisition framerate must be provided!")    
            

            
   
    if len(Times)<len(Frames):
        raise IOError("Time list has fewer elements (" + str(len(Times)) + ") than the number of frames (" + str(len(Frames)) + ")!")

    if OverlayFrames is not None:

        if len(OverlayFrames)<len(Frames):
            raise IOError('\nWARNING: Number of overlay frames (' + str(len(OverlayFrames)) + ') is smaller than number of frames (' +\
                        str(len(Frames)) + '). Overlay disabled')

            OverlayFrames = None

        elif len(OverlayFrames)<len(Frames):
            raise IOError('\nWARNING: Number of overlay frames (' + str(len(OverlayFrames)) + ') is larger than number of frames (' +\
                        str(len(Frames)) + '). Only the first ' + str(len(Frames)) + ' overlay frames will be displayed')

        OverlayFrames_arr = np.asarray(OverlayFrames)

        if OverlayCmap is None:
            OverlayCmap = Ani_config[K_ANI_OVRL_CMAP]

   

    if MapShape is None:
        MapShape = np.shape(Frames[0])
        
    FigExtent = [0, MapShape[1], 0, MapShape[0]]

    if (Title==None):
        Title = Ani_config[K_ANI_TITLE]
        
    if (ColorMap==None):
        ColorMap = 'gray'

    if (BadValue==None):
        global_min = np.min(Frames)
        global_max = np.max(Frames)

        if OverlayFrames is not None:
            g_min_ovr = np.min(OverlayFrames_arr)
            g_max_ovr = np.max(OverlayFrames_arr)

    else:

        Frames_arr = np.asarray(Frames)
        global_min = np.min(Frames_arr[np.nonzero(np.subtract(Frames_arr, BadValue))])
        global_max = np.max(Frames_arr[np.nonzero(np.subtract(Frames_arr, BadValue))])

        if OverlayFrames is not None:
            g_min_ovr = np.min(OverlayFrames_arr[np.nonzero(np.subtract(OverlayFrames_arr, BadValue))])
            g_max_ovr = np.max(OverlayFrames_arr[np.nonzero(np.subtract(OverlayFrames_arr, BadValue))])

       
    

    im_vmin = global_min
    im_vmax = global_max

            
  

   
    if (OverlayFrames is not None):
#        if g_config[K_CORRMAP_OVERLAY_ALPHA_CLIP] is None:
        g_ovr_bounds = np.asarray((g_min_ovr, g_max_ovr))

#        else:
#            g_ovr_bounds = np.asarray(g_config[K_CORRMAP_OVERLAY_ALPHA_CLIP])
        if (Ani_config[K_ANI_OVRL_ALPHA_EXP]==0):
            AlphaValues = np.full(OverlayFrames_arr.shape, Ani_config[K_CORRMAP_OVERLAY_ALPHA_NORM], dtype=np.uint8)
            alpha_weight = None

        else:

            if (Ani_config[K_ANI_OVRL_WEIGHT] == 0):
                alpha_weight = OverlayFrames_arr
                alpha_ovr_bounds = g_ovr_bounds

            elif (Ani_config[K_ANI_OVRL_WEIGHT] == 1):
                alpha_weight = 1-OverlayFrames_arr
                alpha_ovr_bounds = 1-g_ovr_bounds

            elif (Ani_config[K_ANI_OVRL_WEIGHT] == 2):
                alpha_weight = - np.log(OverlayFrames_arr)
                alpha_ovr_bounds = - np.log(g_ovr_bounds)

            else:
                raise ValueError('Unsupported value for configuration parameter ' + str(K_ANI_OVRL_WEIGHT) + ': ' + str(Ani_config[K_ANI_OVRL_WEIGHT]))

            if (np.isnan(alpha_ovr_bounds).any()):
                print('WARNING: non-numeric boundaries for normalization ')
                
            print('max1: '+ str(np.nanmax(alpha_ovr_bounds)))
            print('min: '+ str(np.nanmin(alpha_ovr_bounds)))
            
            alpha_weight_bounds = np.nan_to_num(np.power(alpha_ovr_bounds, Ani_config[K_ANI_OVRL_ALPHA_EXP]))            
            alpha_nan_mask = np.isnan(alpha_weight)
            alpha_weight[alpha_nan_mask] = np.max(alpha_weight_bounds)
            alpha_weight = np.nan_to_num(np.power(alpha_weight, Ani_config[K_ANI_OVRL_ALPHA_EXP]))
            alpha_weight = mpl.colors.Normalize(vmin=np.min(alpha_weight), vmax=np.max(alpha_weight), clip=True)(alpha_weight)
            AlphaValues = np.multiply(alpha_weight, Ani_config[K_ANI_OVRL_ALPHA_NORM])
    

        ovr_cmap = plt.cm.get_cmap(OverlayCmap)

        OverlayFramesBlend = mpl.colors.Normalize(vmin=np.min(alpha_weight), vmax=np.max(alpha_weight), clip=True)(OverlayFrames_arr)
        OverlayFramesBlend = ovr_cmap(OverlayFramesBlend)
        OverlayFramesBlend[..., -1] = AlphaValues


    fig = plt.figure(Title, frameon=False)

    ax = fig.add_subplot(111, aspect='auto', autoscale_on=False, xlim=(0, MapShape[1]), ylim=(0, MapShape[0]))   

    im = ax.imshow(Frames[0], animated=True, vmin=im_vmin, vmax=im_vmax, cmap=ColorMap, extent=FigExtent)
    if (OverlayFrames is not None):
        ovr = ax.imshow(OverlayFramesBlend[0], animated=True, vmin=g_min_ovr, vmax=g_max_ovr, cmap=OverlayCmap, extent=FigExtent)
    
    text = ax.text(Ani_config[K_ANI_TIME_POS][0], Ani_config[K_ANI_TIME_POS][1],\
                     't=' + str('{:0.2f}'.format(Times[0])) + 's', fontsize=12, color= 'r')

#    if (g_config[K_CORRMAP_VIDEO_SCALE_POS]!=None):
#        ax.plot([g_config[K_CORRMAP_VIDEO_SCALE_POS][0], g_config[K_CORRMAP_VIDEO_SCALE_POS][1]],\
#                [g_config[K_CORRMAP_VIDEO_SCALE_POS][2], g_config[K_CORRMAP_VIDEO_SCALE_POS][3]],\
#                '-', linewidth=g_config[K_CORRMAP_VIDEO_SCALE_THICK], color=g_config[K_CORRMAP_VIDEO_LABEL_COLOR])
#        ax.text(g_config[K_CORRMAP_VIDEO_SCALE_L_POS][0], g_config[K_CORRMAP_VIDEO_SCALE_L_POS][1],\
#                g_config[K_CORRMAP_VIDEO_SCALE_LABEL], fontsize=g_config[K_CORRMAP_VIDEO_LABEL_FONT], color=g_config[K_CORRMAP_VIDEO_LABEL_COLOR])

    plt.axis('off')
    fig.tight_layout()
    #plt.show()
    
    def init():
        return 
    
    def animate(idx):
        if (Frames is not None):
            im.set_array(Frames[idx])
        if (OverlayFrames is not None):
            ovr.set_array(OverlayFramesBlend[idx])
        if (text!=None):
            text.set_text('t=' + str('{:0.2f}'.format(Times[idx])) + 's')
        return 
    
#    if len(Frames) > 1:
#        
#        ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(Frames), interval=1.0/Ani_config[K_ANI_FPS], repeat_delay=1000)
#        ExportAnimation(ani, CorrFolder, Ani_config[K_ANI_FILENAME], FPS=Ani_config[K_ANI_FPS], ForceOverwrite=False, Title=Title, Artist=Analysis_info[K_USER_NAME], Comment=Comment)
#
#    else:

    for i in range(len(Frames)):

        animate(i)
        metadata = {'Title':Title, 'Artist':Analysis_info[K_USER_NAME], 'Comment':Comment}
#        if g_config[K_PLT_SAVEFIG_DPI] is not None:
#            g_config[K_PLT_SAVEFIG_DPI] = int(g_config[K_PLT_SAVEFIG_DPI])
        figopts = dict(dpi=Ani_config[K_ANI_DPI], bbox_inches='tight', metadata=metadata)
        fname = OutFolder+Ani_config[K_ANI_FILENAME]+'_'+str(i).zfill(4)+Ani_config[K_ANI_FRAME_EXT]
#        if g_config[K_MAP_BKG_COLOR] is None:
        fig.savefig(fname, transparent=True, **figopts)
#        else:
#            fig.savefig(fname, facecolor=g_config[K_MAP_BKG_COLOR], edgecolor=g_config[K_MAP_BKG_COLOR], **figopts)
    
   
    
#def init():
#    return im, text, qvr
#    
#def animate(idx):
#    if (Frames is not None):
#        im.set_array(Frames[idx])
##       if (OverlayFrames is not None):
##           ovr.set_array(OverlayFramesBlend[idx])
#    if (Quiver is not None):
#        qvr.set_UVC(Quiver[qvr_idx_uv[0]][idx], Quiver[qvr_idx_uv[1]][idx])
#    if (text!=None):
#        text.set_text('t=' + str('{:0.2f}'.format(Times[idx])) + 's')
#    return im, text, qvr
  
    



#def init():
#
#    return im, text
#
#   
#
#def animate(idx):
#
#    im.set_array(frames[idx])
#
##        if (OverlayFrames is not None):
##            ovr.set_array(OverlayFramesBlend[idx])
#
#    if (text!=None):
#            text.set_text('t=' + str('{:0.2f}'.format(times[idx])) + 's')
#
#    return im, text

   

 

# warning: problem if fps<10!

def ExportAnimation(ani, MovieFolder, MovieFilename, FPS=10, ForceOverwrite=False, Title=None, Artist=None, Comment=None):


#    if (g_config[K_FFMPEG_PATH]==None):
#        raise IOError('path for ffmpeg.exe application not specified in global configuration file')

    if (Title==None):
        Title='My animation'


    if (Artist==None):
        Artist=g_config[K_USER_NAME]

   
    if (Comment==None):
        Comment=''

#    while True:
#        if (os.path.isfile(os.path.join(MovieFolder, MovieFilename)) and not ForceOverwrite):
#            if query_yes_no('File ' + MovieFilename + ' already present in folder ' + MovieFolder + '. Overwrite it?'):
#                ForceOverwrite = True
#
#            else:
#                MovieFilename = query_string('Specify new filename (*.mp4)', accept_if_substr='.mp4')
#
#        else:
#            break

#    if g_config[K_FFMPEG_PATH]!=None and g_config[K_FFMPEG_PATH]!='conda':
#        unicode_path = str(g_config[K_FFMPEG_PATH])
#
#        plt.rcParams['animation.ffmpeg_path'] = unicode_path

#    unicode_path = Ani_config[K_ANI_FFMPEG_PATH]
#    plt.rcParams['animation.ffmpeg_path'] = unicode_path   
    metadata = dict(title=Title, artist=Artist, comment=Comment)
    mywriter = animation.FFMpegWriter(fps=FPS, metadata=metadata) # warning: problem if fps<10!
    ani.save(MovieFolder + MovieFilename, writer=mywriter)

    print('Animation successfully exported to file {0}'.format(MovieFolder + MovieFilename))
    

bytes_to_read = Ani_config[K_ANI_NUM_FRAMES] * MI_info[K_MI_PIXELS_PER_IMAGE]
struct_format = ('%sB') % bytes_to_read
                
Frames_array = Load_from_file(InputFolder, MIfile_name, struct_format, start_idx=MI_info[K_MI_PIXELS_PER_IMAGE]*Analysis_info[K_AN_FIRST_IMG])
Frames = Frames_array.reshape(Ani_config[K_ANI_NUM_FRAMES], MI_info[K_MI_IMAGE_HEIGHT], MI_info[K_MI_IMAGE_WIDTH])

ovrl_pix_per_image = Ani_config[K_ANI_OVRL_FRAMESIZE][0] * Ani_config[K_ANI_OVRL_FRAMESIZE][1] 
bytes_to_read = Ani_config[K_ANI_NUM_FRAMES] * ovrl_pix_per_image
struct_format = ('%sf') % bytes_to_read

ovrl_array = Load_from_file(OvrlFolder, Ani_config[K_ANI_OVRL_FILENAME], struct_format, start_idx=ovrl_pix_per_image*(Ani_config[K_ANI_START_IDX]-Analysis_info[K_AN_FIRST_IMG]), pix_depth=4)
ovrl_Frames = ovrl_array.reshape(Ani_config[K_ANI_NUM_FRAMES], Ani_config[K_ANI_OVRL_FRAMESIZE][1], Ani_config[K_ANI_OVRL_FRAMESIZE][0])


AnimateCorrMap(Frames, FileName=Ani_config[K_ANI_FILENAME], Times=None, fLog=None, Title=None, Comment=None, ColorMap=None, OverlayFrames=ovrl_Frames,\
                   Quiver=None, VelocityMaps=None, OverlayCmap=None, BadValue=None, MapShape=None)
                    
    
#    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(Frames), interval=1.0/g_config[K_CORRMAP_VIDEO_FPS], repeat_delay=1000)
#ani = animation.FuncAnimation(fig, animate, init_func=init, frames=len(Frames), interval=1.0/Ani_config[K_ANI_FPS], repeat_delay=1000)

#    ExportAnimation(ani, g_config[K_CORRMAP_FOLDER_PATH], FileName, FPS=g_config[K_CORRMAP_VIDEO_FPS], ForceOverwrite=False, Title=Title, Artist=g_config[K_USER_NAME], Comment=Comment)
#ExportAnimation(ani, CorrFolder, Ani_config[K_ANI_FILENAME], FPS=Ani_config[K_ANI_FPS], ForceOverwrite=False, Title=Ani_config[K_ANI_TITLE], Artist=Analysis_info[K_USER_NAME])

