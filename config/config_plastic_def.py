"""
Created on Tue Jul 16 11:35:13 2019

@author: Matteo Sabato - msabato@g.harvard.edu

Initializing dictionary and other configurations
"""

#initializing hande to read configuration file
f_conf = open(ProgramRoot + 'config/ConfigFile_plastic_def.txt', 'r')
 
AnalysisFolder = ReadConfigFile(f_conf)
Check_Path(AnalysisFolder)

CorrFolder = AnalysisFolder + ReadConfigFile(f_conf)
Check_Path(CorrFolder)
    

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
#MIfile_handle = open(InputFolder + MIfile_name, 'rb')

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
                            K_AN_LAG_LIST:ReadConfigFile(f_conf),
                            K_AN_CUTOFF:ReadConfigFile(f_conf),
                            K_AN_CORR_CUTOFF:ReadConfigFile(f_conf),
                            
                            K_CORRTODISPL_NONAFF_LINTR:ReadConfigFile(f_conf),
                            K_CORRTODISPL_NONAFF_CVAL:ReadConfigFile(f_conf),
                            K_CORRTODISPL_NONAFF_INTERP:ReadConfigFile(f_conf),
                            
                            K_USE_AOI:ReadConfigFile(f_conf),
                            K_AOI_POS:ReadConfigFile(f_conf),
                            K_AOI_SIZE:ReadConfigFile(f_conf)
                            
                         }
                     )

