"""
Created on Tue Jul 16 11:43:20 2019

@author: Matteo Sabato - msabato@g.harvard.edu

Dictionary variable names
"""

#keys in MI_info

K_MI_TOT_HEADER_SIZE        = 'tot_hdr_size'
K_MI_IMAGE_NUMBER           = 'img_num'
K_MI_PIXELS_PER_IMAGE       = 'px_per_img'
K_MI_VARIABLE_FORMAT        = 'var_format'
K_MI_PIXEL_DEPTH            = 'px_depth'
K_MI_ACQUISITION_FPS        = 'acquisition_framerate'
K_MI_IMAGE_WIDTH            = 'img_width'
K_MI_IMAGE_HEIGHT           = 'img_height'

#keys in Analysis_info

K_USER_NAME                 = 'username'

K_AN_IMAGE_SIZE             = 'image_size'
K_AN_ROI_SIZE               = 'ROI_size'
K_AN_ROI_per_line           = 'ROIs_per_line'
K_AN_ROI_LINES_NUM          = 'lines_num'               # number of ROI lines in file
K_AN_FIRST_IMG              = 'first_image'
K_AN_LAST_IMG               = 'last_image'
K_AN_IMGS_NUM               = 'images_num'              #number of correlation functions to be calculated
K_AN_NUM_LAGS               = 'number_of_lags'          #list of correlation functions
K_AN_LAG_LIST               = 'lags_lis'
K_AN_NL_FIT                 = 'bool_nl_fit'
K_AN_USE_SEED               = 'bool_use_seed'
K_AN_VIDEO_FPS              = 'video_framerate'         #framerate of output video
K_AN_MASK                   = 'apply_mask'
K_AN_MASK_FILENAME          = 'mask_filename'

K_AN_LINE_N                 = 'line_n'                  #line number for deformation field analysis
K_AN_ROI_N                  = 'ROI_n'                   #ROI number for deformation field analysis
K_AN_ROI_POS                = 'ROI_pos'
K_AN_t_0_NUM                = 't_0_num'                 #number of different t_0
K_AN_t_0                    = 't_0'                     #frame number for deformation field analysis
K_AN_CUTOFF                 = 'cutoff_value'
K_AN_CORR_CUTOFF            = 'corr_cutoff'

K_CORRMAP_ROWSTEP           = 'correlation_map_rowstep'     # Load 1 image every N. Specify N
K_CORRMAP_AVGSTEP           = 'correlation_map_avgblock'    # Average on blocks of N images. Specify N
K_CORRMAP_VIDEO_COLORMAP    = 'map_video_colormap'
K_CORRMAP_VIDEO_T_LABEL_POS = 'map_video_t_label_pos'
K_CORRMAP_VIDEO_LABEL_FONT  = 'label_font'
K_CORRMAP_VIDEO_LABEL_COLOR = 'label_color'

K_CRACK_START               = 'crack_start_frame'
K_DATA_FPS                  = 'data_fps'
K_FFMPEG_PATH               = 'ffmpeg_path'

K_CORRTODISPL_NONAFF_LINTR  = 'nonaff_trans_coefficients'
K_CORRTODISPL_NONAFF_CVAL   = 'nonaff_trans_fillvalue'
K_CORRTODISPL_NONAFF_INTERP = 'interp_order'

#lif file infos

K_LIF_SERIES_NUM            = 'series_number'
K_LIF_SERIES_LIST           = 'series_list'
K_LIF_STACK_HEIGHT          = 'stack_height'
K_LIF_FRAMES_NUMBER         = 'frames_number'
K_LIF_TEMP_SIZE             = 'template_size'
K_LIF_TEMP_POS              = 'template_pos'
K_LIF_CUTOFF                = 'cutoff_value'
K_LIF_FPS                   = 'lif_fps'
K_LIF_PXL_SIZE              = 'pixel_size'

#calibration

K_USE_AOI                   = 'bool_use_aoi'
K_AOI_POS                   = 'aoi_pos'
K_AOI_SIZE                  = 'aoi_size'

#open custom img format

IMG_FILENAME                = 'filename'
IMG_EXT                     = 'extension'
IMG_BOOL_MULTI              = 'bool_multi_file'
IMG_FILES_NUM               = 'files_num'
IMG_START_FILE              = 'starting_file_num'

IMG_HEADER_SIZE             = 'header_size'
IMG_GAP                     = 'gap_between_images'
IMG_NUM_per_FILE            = 'img_num_per_file'
IMG_WIDTH                   = 'img_width'
IMG_HEIGHT                  = 'img_height'

IMG_PIXELS_PER_IMAGE        = 'pixels_per_image'
IMG_PIXEL_DEPTH             = 'pixel_depth'
IMG_VAR_FORMAT              = 'var_format'
IMG_START_IDX               = 'start_idx'
IMG_LAST_IDX                = 'last_idx'
IMG_SKIP                    = 'imgs_to_skip'
IMG_TO_READ                 = 'imgs_to_read_per_file'

#Animations

K_ANI_FILENAME              = 'animation_filename'
K_ANI_START_IDX             = 'animation_start_idx'
K_ANI_END_IDX               = 'animation_last_idx'
K_ANI_NUM_FRAMES            = 'animation_num_frames'
K_ANI_FPS                   = 'animation_framerate'
K_ANI_DPI                   = 'animation_dpi'
K_ANI_FRAME_EXT             = 'ani_frame_ext'
K_ANI_FFMPEG_PATH           = 'ffmpeg_path'
K_ANI_TIME_BOOL             = 'show_times'
K_ANI_TIME_POS              = 'time_label_pos'
K_ANI_QUIV_BOOL             = 'use_quiver'
K_ANI_VEL_FILENAMES         = 'vel_filenames'
K_ANI_COARSE                = 'coarsening_factor'
K_ANI_TITLE                 = 'title'
K_ANI_ARROW_COLOR           = 'arr_color'
K_ANI_ARROW_SCALE           = 'arr_scale'
K_ANI_ARROW_UNITS           = 'arr_units'
K_ANI_ARROW_HEADLENGTH      = 'arr_headlength'
K_ANI_ARROW_PIVOT           = 'arr_pivot'
K_ANI_ARROW_HEADWIDTH       = 'arr_headwidth'
K_ANI_ARROW_LINEWIDTH       = 'arr_linewidth'
K_ANI_ARROW_WIDTH           = 'arr_width'
K_ANI_OVRL_BOOL             = 'ovrl_bool'
K_ANI_OVRL_FILENAME         = 'ovrl_filename'
K_ANI_OVRL_FRAMESIZE        = 'ovrl_framesize'
K_ANI_OVRL_PIX_DEPTH        = 'ovrl_pix_depth'
K_ANI_OVRL_WEIGHT           = 'ovrl_weight'
K_ANI_OVRL_ALPHA_EXP        = 'ovrl_exp'
K_ANI_OVRL_ALPHA_NORM       = 'ovrl_alpha_norm'
K_ANI_OVRL_CMAP             = 'ovrl_colormap'

