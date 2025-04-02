# %% Imports
##############################################################################
#%matplotlib widget

import os
import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality
import cedalion.xrutils as xrutils
from cedalion.sigdecomp.ERBM import ERBM

import xarray as xr
import matplotlib.pyplot as p
import cedalion.plots as plots
from cedalion import units
import numpy as np
import pandas as pd
from math import ceil

import gzip
import pickle
import json


# import my own functions from a different directory
import sys
#sys.path.append('/Users/dboas/Documents/GitHub/cedalion-dab-funcs')
sys.path.append('/projectnb/nphfnirs/ns/Shannon/Code/cedalion-dab-funcs2/modules')
import module_load_and_preprocess as pfDAB
import module_plot_DQR as pfDAB_dqr
import module_group_avg as pfDAB_grp_avg    
import module_ERBM_ICA as pfDAB_ERBM
import module_image_recon as pfDAB_img
import module_spatial_basis_funs_ced as sbf 


# Turn off all warnings
import warnings
warnings.filterwarnings('ignore')

import pyvista as pv
# %% 
##############################################################################
import importlib
importlib.reload(pfDAB_dqr)
importlib.reload(pfDAB)
importlib.reload(pfDAB_grp_avg)


# %% Initial root directory and analysis parameters
##############################################################################

cfg_hrf = {
    'stim_lst' : ['ST', 'DT'], 
    't_pre' : 5 *units.s, 
    't_post' : 33 *units.s
    #'t_post' : [ 33, 33 ] *units.s   # !!! GLM does not let you have different time ranges for diff stims right now
    }

cfg_dataset = {
    'root_dir' : "/projectnb/nphfnirs/ns/Shannon/Data/Interactive_Walking_HD/",
    'subj_ids' : ['01','02','03','04','05','06','07','08','09','10', '11', '12', '13', '14', '15', '16', '17', '18', '19'],
    'file_ids' : ['IWHD_run-01'],
    'subj_id_exclude' : ['10', '15', '16', '17'], #['05','07'] # if you want to exclude a subject from the group average
    'cfg_hrf' : cfg_hrf
}

# Add 'filenm_lst' separately after cfg_dataset is initialized
cfg_dataset['filenm_lst'] = [
    [f"sub-{subj_id}_task-{file_id}_nirs"] 
    for subj_id in cfg_dataset['subj_ids'] 
    for file_id in cfg_dataset['file_ids']
    ]



cfg_prune = {
    'snr_thresh' : 5, # the SNR (std/mean) of a channel. 
    'sd_threshs' : [1, 60]*units.mm, # defines the lower and upper bounds for the source-detector separation that we would like to keep
    'amp_threshs' : [1e-5, 0.84], # define whether a channel's amplitude is within a certain range
    'perc_time_clean_thresh' : 0.6,
    'sci_threshold' : 0.6,
    'psp_threshold' : 0.1,
    'window_length' : 5 * units.s,
    'flag_use_sci' : True,
    'flag_use_psp' : False
}

cfg_imu_glm = {'statesPerDataFrame' : 89,   # FOR WALKING DATA
		'hWin' : np.arange(-3,5,1), # window for impulse response function 
		'statesPerDataFrame' : 89,
		'n_components' : [3, 2],  # [gyro, accel]       # !!! note: changing this will change fig sizes - add that in?
        'butter_order' : 4,   # butterworth filter order
        'Fc' : 0.1,   # cutoff freq (Hz)
        'plot_flag_imu' : True  
}

cfg_motion_correct = {
    #'flag_do_splineSG' : False, # !!! This is not doing anything. left out for now. if True, will do splineSG motion correction
    #'splineSG_p' : 0.99, 
    #'splineSG_frame_size' : 10 * units.s,
    'flag_do_tddr' : True,  
    'flag_do_imu_glm' : True,
    'cfg_imu_glm' : cfg_imu_glm,
}

cfg_bandpass = { 
    'fmin' : 0.01 * units.Hz, #0.02 * units.Hz,
    'fmax' : 0.5 * units.Hz  #3 * units.Hz
}


cfg_GLM = {
    'drift_order' : 1,
    'distance_threshold' : 20 *units.mm, # for ssr
    'short_channel_method' : 'mean',
    'noise_model' : "ols",    # !!! add choice of basis func 
    't_delta' : 1 *units.s ,   # for seq of Gauss basis func - the temporal spacing between consecutive gaussians
    't_std' : 1 *units.s ,     #  the temporal spacing between consecutive gaussians
    'cfg_hrf' : cfg_hrf
    }           


cfg_preprocess = {
    'flag_prune_channels' : False,  # FALSE = does not prune chans and does weighted averaging, TRUE = prunes channels and no weighted averaging
    'median_filt' : 1, # set to 1 if you don't want to do median filtering
    'cfg_prune' : cfg_prune,
    'cfg_motion_correct' : cfg_motion_correct,
    'cfg_bandpass' : cfg_bandpass,
    'flag_do_GLM_filter' : True,
    'cfg_GLM' : cfg_GLM 
}


cfg_mse_conc = {                
    'mse_val_for_bad_data' : 1e7 * units.micromolar**2, 
    'mse_amp_thresh' : 1.1e-6,
    'mse_min_thresh' : 1e0 * units.micromolar**2, 
    'blockaverage_val' : 0 * units.micromolar
    }

# if block averaging on OD:
cfg_mse_od = {
    'mse_val_for_bad_data' : 1e1, 
    'mse_amp_thresh' : 1.1e-6,
    'mse_min_thresh' : 1e-6,
    'blockaverage_val' : 0      # blockaverage val for bad data?
    }

cfg_blockavg = {
    'rec_str' : 'od_corrected',   # what you want to block average (will be either 'od_corrected' or 'conc')
    'flag_prune_channels' : cfg_preprocess['flag_prune_channels'],
    'cfg_hrf' : cfg_hrf,
    'trange_hrf_stat' : [10, 20],  
    'flag_save_group_avg_hrf': False,
    'flag_save_each_subj' : False,  # if True, will save the block average data for each subject
    'cfg_mse_conc' : cfg_mse_conc,
    'cfg_mse_od' : cfg_mse_od
    }               



cfg_erbmICA = {}

save_path = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data')

flag_load_preprocessed_data = True  
flag_save_preprocessed_data = False   # SAVE or no save

flag_load_blockaveraged_data = False


# %% Load and preprocess the data
##############################################################################

# determine the number of subjects and files. Often used in loops.
n_subjects = len(cfg_dataset['subj_ids'])
n_files_per_subject = len(cfg_dataset['file_ids'])

# files to load
for subj_id in cfg_dataset['subj_ids']:
    subj_idx = cfg_dataset['subj_ids'].index(subj_id)
    for file_id in cfg_dataset['file_ids']:
        file_idx = cfg_dataset['file_ids'].index(file_id)
        filenm = f'sub-{subj_id}_task-{file_id}_nirs'
        if subj_idx == 0 and file_idx == 0:
            cfg_dataset['filenm_lst'] = []
            cfg_dataset['filenm_lst'].append( [filenm] )
        elif file_idx == 0:
            cfg_dataset['filenm_lst'].append( [filenm] )
        else:
            cfg_dataset['filenm_lst'][subj_idx].append( filenm )

import importlib
importlib.reload(pfDAB)


# File naming stuff
p_save_str = ''
if cfg_motion_correct['flag_do_imu_glm']:  # to identify if data is pruned or unpruned
    p_save_str =  p_save_str + '_imuGLM' 
else:
    p_save_str =  p_save_str
if cfg_motion_correct['flag_do_tddr']:  # to identify if data is pruned or unpruned
    p_save_str =  p_save_str + '_tddr' 
else:
    p_save_str =  p_save_str 
if cfg_preprocess['flag_do_GLM_filter']:  # to identify if data is pruned or unpruned
    p_save_str =  p_save_str + '_GLMfilt' 
else:
    p_save_str =  p_save_str   
if cfg_preprocess['flag_prune_channels']:  # to identify if data is pruned or unpruned
    p_save_str =  p_save_str + '_pruned' 
else:
    p_save_str =  p_save_str + '_unpruned' 
    
    
# RUN PREPROCESSING
if not flag_load_preprocessed_data:
    print("Running load and process function")
    
    # RUN preprocessing
    rec, chs_pruned_subjs = pfDAB.load_and_preprocess( cfg_dataset, cfg_preprocess ) 

    
    # SAVE preprocessed data 
    if flag_save_preprocessed_data:
        print(f"Saving preprocessed data for {cfg_dataset['file_ids']}")
        with gzip.open( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data', 
                                     'chs_pruned_subjs_ts_' + cfg_dataset["file_ids"][0].split('_')[0] + p_save_str + '.pkl'), 'wb') as f: # !!! FIX ME: naming convention assumes file_ids only includes ONE task
            pickle.dump(chs_pruned_subjs, f, protocol=pickle.HIGHEST_PROTOCOL )
            
        with gzip.open( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data', 
                                     'rec_list_ts_' + cfg_dataset["file_ids"][0].split('_')[0] + p_save_str + '.pkl'), 'wb') as f:
            pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL )
            
            
        # SAVE cfg params to json file
        # !!! ADD image recon cfg  ?? - or make it its own .json since i am planning to separate into 2 scripts
        dict_cfg_save = {"cfg_hrf": cfg_hrf, "cfg_dataset" : cfg_dataset, "cfg_preprocess" : cfg_preprocess, "cfg_GLM" : cfg_GLM, "cfg_blockavg" : cfg_blockavg}
        
        cfg_save_str = 'cfg_params_' + cfg_dataset["file_ids"][0].split('_')[0] + p_save_str + '.json'
            
        with open(os.path.join(save_path, cfg_save_str), "w", encoding="utf-8") as f:
            json.dump(dict_cfg_save, f, indent=4, default = str)  # Save as JSON with indentation
        print("Preprocessed data successfully saved.")
        
        
# LOAD IN SAVED DATA
else:
    print("Loading saved data")   # !!! update with new naming for pruned or unpruned above
    with gzip.open( os.path.join(save_path, 'rec_list_ts_' + cfg_dataset["file_ids"][0].split('_')[0] + p_save_str + '.pkl'), 'rb') as f: # !!! FIX ME: this assumes file_ids only includes ONE task
         rec = pickle.load(f)
    with gzip.open( os.path.join(save_path, 'chs_pruned_subjs_ts_' + cfg_dataset["file_ids"][0].split('_')[0] + p_save_str + '.pkl'), 'rb') as f:
         chs_pruned_subjs = pickle.load(f)
    print(f'Data loaded successfully for {cfg_dataset["file_ids"][0].split("_")[0]}')





# %% ERBM ICA Filtering 
##############################################################################
'''
import importlib
importlib.reload(pfDAB_ERBM)

# used for helping determine which ICA components to keep and remove
trange_hrf = [5, 35] * units.s # time range for block averaging
trange_hrf_stat = [5, 20] # time range for t-stat
stim_lst_hrf_ica = ['STS'] # which trial_types to consider for which ICA components to keep

ica_spatial_mask_thresh = 1.0 # for selecting "etCO2" components to remove
ica_tstat_thresh = 1.0 # for selecting significant components to keep

pca_var_thresh = 0.99 # keep enough PCs to explain this fraction of the variance
p_ica = 27 # not sure what this does

ica_lpf = 1.0 * units.Hz # low pass filter the data before ICA
ica_downsample = 1  # downsample the data by this factor before running ICA. ICA cost is linear with number of samples.
                    # and since we low pass filtered the data before ICA, we can downsample it to save time.
                    # Note that the NN22 sample rate is often ~9 Hz, and will be reduced by this factor.

cov_amp_thresh = 1.1e-6 # threshold for the amplitude of the channels below which we assign a high variance
                        # for ninjaNIRS, negative amp's are set to 1e-6. Sometimes spikes bring the mean slightly above 1e-6


flag_do_pca_filter = True
flag_calculate_ICA_matrix = False
flag_do_ica_filter = True

flag_ICA_use_pruned_data = False # if True, use the pruned data for ICA, otherwise use the original data
                                 # if False, then we need to correct the variances of the pruned channels for the ts_zscore
flag_ERBM_vs_EBM = False # if True, use ERBM, otherwise use EBM


# FIXME: I want to verify that this properly scales back the NOT pruned data to channel space
rec = pfDAB_ERBM.ERBM_run_ica( rec, filenm_lst, flag_ICA_use_pruned_data, ica_lpf, ica_downsample, cov_amp_thresh, chs_pruned_subjs, pca_var_thresh, flag_do_pca_filter, flag_calculate_ICA_matrix, flag_ERBM_vs_EBM, p_ica, rootDir_data, flag_do_ica_filter, ica_spatial_mask_thresh, ica_tstat_thresh, trange_hrf, trange_hrf_stat, stim_lst_hrf_ica )


# FIXME: should not be needed here... shouldbe handled in ICA step above
ica_lpf = 1.0 * units.Hz # MUST be the same as used when creating W_ica

'''

# %% Block Average - unweighted and weighted
##############################################################################

import importlib
importlib.reload(pfDAB_grp_avg)

#flag_load_blockaveraged_data = False


# for saving file name 
    
if 'conc' in cfg_blockavg['rec_str']:  
    save_str = p_save_str + '_CONC' 
else:
    save_str = p_save_str + '_OD' 
    

# Compute block average
if not flag_load_blockaveraged_data:  
    
    if cfg_preprocess['flag_prune_channels']:   # if using pruned data, don't save weighted
        blockaverage_mean, _, blockaverage_stderr, blockaverage_subj, blockaverage_mse_subj = pfDAB_grp_avg.run_group_block_average( rec, cfg_blockavg['rec_str'], chs_pruned_subjs, cfg_dataset, cfg_blockavg )
    
    else:    # if not pruning, save weighted blockaverage data
        _, blockaverage_mean, blockaverage_stderr, blockaverage_subj, blockaverage_mse_subj = pfDAB_grp_avg.run_group_block_average( rec, cfg_blockavg['rec_str'], chs_pruned_subjs, cfg_dataset, cfg_blockavg )
    
    groupavg_results = {'blockaverage': blockaverage_mean,
               'blockaverage_stderr': blockaverage_stderr,
               'blockaverage_subj': blockaverage_subj,
               'blockaverage_mse_subj': blockaverage_mse_subj,
               'geo2d' : rec[0][0].geo2d,
               'geo3d' : rec[0][0].geo3d
               }
    
    if cfg_blockavg['flag_save_group_avg_hrf']:
        file_path_pkl = os.path.join(save_path, 'blockaverage_' + cfg_dataset["file_ids"][0].split('_')[0] + save_str + '.pkl.gz')
        file = gzip.GzipFile(file_path_pkl, 'wb')
        file.write(pickle.dumps(groupavg_results))
        file.close()
        print('Saved group average HRF to ' + file_path_pkl)

else: # LOAD data
    filname =  'blockaverage_' + cfg_dataset["file_ids"][0].split('_')[0]  + save_str + '.pkl.gz'
    filepath_bl = os.path.join(save_path , filname)
    
    if os.path.exists(filepath_bl):
        with gzip.open(filepath_bl, 'rb') as f:
            groupavg_results = pickle.load(f)
        blockaverage_mean = groupavg_results['blockaverage']
        blockaverage_stderr = groupavg_results['blockaverage_stderr']
        blockaverage_subj = groupavg_results['blockaverage_subj']
        blockaverage_mse_subj = groupavg_results['blockaverage_mse_subj']
        geo2d = groupavg_results['geo2d']
        geo2d = groupavg_results['geo3d']
        print("Blockaverage file loaded successfully!")
    
    else:
        print(f"Error: File '{filepath_bl}' not found!")
        
blockaverage_all = blockaverage_mean.copy()


