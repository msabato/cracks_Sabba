"""
Created on Tue Jul 16 11:35:13 2019

@author: Matteo Sabato - msabato@g.harvard.edu

Initializing dictionary and other configurations
"""

#initializing hande to read configuration file

ProgramRoot = '/Users/matteo/Documents/Uni/TESI/_python/ProgrammaMatteo/'

f_conf = open(ProgramRoot + 'config/ConfigFile_animations.txt', 'r')

InputFolder = ReadConfigFile(f_conf)
Check_Path(InputFolder)

OvrlFolder = ReadConfigFile(f_conf)
Check_Path(InputFolder)
 
AnalysisFolder = ReadConfigFile(f_conf)
Check_Path(AnalysisFolder)

CorrFolder = AnalysisFolder + ReadConfigFile(f_conf)
Check_Path(CorrFolder)

OutFolder = CorrFolder + 'animation/'
if not os.path.isdir(OutFolder):
    os.mkdir(OutFolder)
    
if not os.path.isdir(OutFolder + 'frames/'):
    os.mkdir(OutFolder + 'frames/')
    

#OutFolder   = AnalysisFolder + 'out/'
#PlotsFolder = AnalysisFolder + 'plots/'
#
##avoid overwriting old files by accident
#folder_number = 0
#
#while os.path.isdir(OutFolder):
#    folder_number += 1
#    OutFolder = AnalysisFolder + 'out' + str(folder_number) + '/'
#    
#os.mkdir(OutFolder)
    

#storing information about MI file

MIfile_name = ReadConfigFile(f_conf)

MI_info = {
                K_MI_TOT_HEADER_SIZE:ReadConfigFile(f_conf),
                K_MI_IMAGE_NUMBER:ReadConfigFile(f_conf),
                K_MI_PIXELS_PER_IMAGE:ReadConfigFile(f_conf),              
                K_MI_IMAGE_WIDTH:ReadConfigFile(f_conf),
                K_MI_IMAGE_HEIGHT:ReadConfigFile(f_conf),
                K_MI_PIXEL_DEPTH:ReadConfigFile(f_conf),
                K_MI_ACQUISITION_FPS:ReadConfigFile(f_conf),
                K_MI_VARIABLE_FORMAT:ReadConfigFile(f_conf)
            }

#initializing handle to read file
MIfile_handle = open(InputFolder + MIfile_name, 'rb')

#analysis parameters
Analysis_info = {
                   
                    K_USER_NAME:ReadConfigFile(f_conf),
                    K_AN_IMAGE_SIZE:ReadConfigFile(f_conf),
                    K_AN_ROI_SIZE:ReadConfigFile(f_conf)
                    
                }
                
Analysis_info.update( 
                         {
            
                            K_AN_ROI_per_line:int(MI_info[K_MI_IMAGE_WIDTH])//Analysis_info[K_AN_ROI_SIZE][0],
                            K_AN_ROI_LINES_NUM:int(MI_info[K_MI_IMAGE_HEIGHT])//Analysis_info[K_AN_ROI_SIZE][1],
                            K_AN_FIRST_IMG:ReadConfigFile(f_conf),
                            K_AN_LAST_IMG:ReadConfigFile(f_conf)
                        
                          }
                    )

Analysis_info.update( 
                         {
            
                            K_AN_IMGS_NUM:Analysis_info[K_AN_LAST_IMG] - Analysis_info[K_AN_FIRST_IMG] + 1,
                            K_AN_NUM_LAGS:ReadConfigFile(f_conf),
                            K_AN_LAG_LIST:ReadConfigFile(f_conf)
                            
                         }
                    )

Ani_config = {
                K_ANI_FILENAME:ReadConfigFile(f_conf),
                K_ANI_START_IDX:ReadConfigFile(f_conf),
                K_ANI_END_IDX:ReadConfigFile(f_conf)
            }

ani_numFrames = (Ani_config[K_ANI_END_IDX]-Ani_config[K_ANI_START_IDX]+1)-Analysis_info[K_AN_LAG_LIST][Analysis_info[K_AN_NUM_LAGS]-1]
Ani_config.update(
                    {
                            
                        K_ANI_NUM_FRAMES:ani_numFrames,    
                        K_ANI_FPS:ReadConfigFile(f_conf),
                        K_ANI_DPI:ReadConfigFile(f_conf),
                        K_ANI_FRAME_EXT:ReadConfigFile(f_conf),
                        K_ANI_FFMPEG_PATH:ReadConfigFile(f_conf),
                        K_ANI_TIME_BOOL:bool(ReadConfigFile(f_conf)),
                        K_ANI_TIME_POS:ReadConfigFile(f_conf),
                        K_ANI_QUIV_BOOL:bool(ReadConfigFile(f_conf))
                
                    }
                )


vel_names = []
vel_names.append(ReadConfigFile(f_conf))
vel_names.append(ReadConfigFile(f_conf))

Ani_config.update(
                    {
        
                        K_ANI_VEL_FILENAMES:vel_names,
                        K_ANI_COARSE:ReadConfigFile(f_conf),
                        K_ANI_TITLE:ReadConfigFile(f_conf),
                        K_ANI_ARROW_COLOR:ReadConfigFile(f_conf),
                        K_ANI_ARROW_SCALE:ReadConfigFile(f_conf),
                        K_ANI_ARROW_UNITS:ReadConfigFile(f_conf),
                        K_ANI_ARROW_HEADLENGTH:ReadConfigFile(f_conf),
                        K_ANI_ARROW_PIVOT:ReadConfigFile(f_conf),
                        K_ANI_ARROW_HEADWIDTH:ReadConfigFile(f_conf),
                        K_ANI_ARROW_LINEWIDTH:ReadConfigFile(f_conf),
                        K_ANI_ARROW_WIDTH:ReadConfigFile(f_conf),
                        K_ANI_OVRL_BOOL:bool(ReadConfigFile(f_conf)),
                        K_ANI_OVRL_FILENAME:ReadConfigFile(f_conf),
                        K_ANI_OVRL_FRAMESIZE:ReadConfigFile(f_conf),
                        K_ANI_OVRL_PIX_DEPTH:ReadConfigFile(f_conf),
                        K_ANI_OVRL_WEIGHT:ReadConfigFile(f_conf),
                        K_ANI_OVRL_ALPHA_EXP:ReadConfigFile(f_conf),
                        K_ANI_OVRL_ALPHA_NORM:ReadConfigFile(f_conf),
                        K_ANI_OVRL_CMAP:ReadConfigFile(f_conf)                
            
                    }
                 )             
                         
                         
                         
                            