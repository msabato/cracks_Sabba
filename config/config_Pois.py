"""
Created on Tue Jul 16 11:35:13 2019

@author: Matteo Sabato - msabato@g.harvard.edu

Initializing dictionary and other configurations
"""

#initializing hande to read configuration file
f_conf = open(ProgramRoot + 'config/ConfigFile_Pois.txt', 'r')

InputFolder = ReadConfigFile(f_conf)
Check_Path(InputFolder)
 
AnalysisFolder = ReadConfigFile(f_conf)
Check_Path(AnalysisFolder)

CorrFolder = AnalysisFolder + ReadConfigFile(f_conf)
Check_Path(CorrFolder)

OutFolder   = AnalysisFolder + 'out/'
PlotsFolder = AnalysisFolder + 'plots/'

#avoid overwriting old files by accident
folder_number = 0

while os.path.isdir(OutFolder):
    folder_number += 1
    OutFolder = AnalysisFolder + 'out' + str(folder_number) + '/'
    

#storing information about MI file

LIF_file_name = ReadConfigFile(f_conf)

LIF_info = {
                K_LIF_SERIES_NUM:ReadConfigFile(f_conf),
                K_LIF_SERIES_LIST:ReadConfigFile(f_conf),
                K_LIF_STACK_HEIGHT:ReadConfigFile(f_conf),
                K_LIF_FRAMES_NUMBER:ReadConfigFile(f_conf),
                K_LIF_TEMP_SIZE:ReadConfigFile(f_conf),
                K_LIF_TEMP_POS:ReadConfigFile(f_conf),
                K_LIF_CUTOFF:ReadConfigFile(f_conf),
                K_LIF_FPS:ReadConfigFile(f_conf),
                K_LIF_PXL_SIZE:ReadConfigFile(f_conf)

            }

