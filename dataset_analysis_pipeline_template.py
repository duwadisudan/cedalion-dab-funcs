# %% Imports
##############################################################################
#%matplotlib widget

import os
import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality

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


# import my own functions from a different directory
import sys
sys.path.append('/Users/dboas/Documents/GitHub/cedalion-dab-funcs')
#sys.path.append('C:\\Users\\shank\\Documents\\GitHub\\cedalion-dab-funcs2')
#sys.path.append('/projectnb/nphfnirs/ns/Shannon/Code/cedalion-dab-funcs2')
import DABfuncs_load_and_preprocess as pfDAB
import DABfuncs_plot_DQR as pfDAB_dqr
import DABfuncs_group_avg as pfDAB_grp_avg
import DABfuncs_ERBM_ICA as pfDAB_ERBM
import DABfuncs_image_recon as pfDAB_img
import spatial_basis_funs_ced as sbf 


# Turn off all warnings
import warnings
warnings.filterwarnings('ignore')


# %% 
##############################################################################
import importlib
importlib.reload(pfDAB_dqr)
importlib.reload(pfDAB)
importlib.reload(pfDAB_grp_avg)


# %% Initial root directory and analysis parameters
##############################################################################

flag_load_preprocessed_data = False  # if 1, will skip load_and_preprocess function and use saved data
rootDir_saveData = '/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data/Datasets/Laura_WMandRest/derivatives/processed_data/'
flag_save_preprocessed_data = True


cfg_dataset = {
    'root_dir' : '/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data/Datasets/Laura_WMandRest/',
    'subj_ids' : ['01','02','03','04','05','06','07','08'],
    'file_ids' : ['WM_run-01','WM_run-02','WM_run-03','WM_run-04'],
    'subj_id_exclude' : [], #['05','07'] # if you want to exclude a subject from the group average
    #'stim_lst' : ['ST', 'DT']  # FIXME: use this instead of having separate stim lists for hrf, dqr, ica, etc ???? - SK
}

# Add 'filenm_lst' separately after cfg_dataset is initialized
cfg_dataset['filenm_lst'] = [
    [f"sub-{subj_id}_task-{file_id}_nirs"] 
    for subj_id in cfg_dataset['subj_ids'] 
    for file_id in cfg_dataset['file_ids']
    ]

cfg_dqr = {
    'stim_lst_dqr' : ['passive', 'active'] # FIXME: why have multiple of this
}

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

cfg_imu_glm = {'statesPerDataFrame' : 89,
		'hWin' : np.arange(-3,5,1), # window for impulse response function 
		'statesPerDataFrame' : 89,
		'n_components' : [3, 2],  # [gyro, accel]       # --- note: this will change fig sizes
        'butter_order' : 4,   # butterworth filter order
        'Fc' : 0.1,   # cutoff freq (Hz)
        'plot_flag_imu' : True
}

cfg_motion_correct = {
    'flag_do_splineSG' : False, # if True, will do splineSG motion correction
    'splineSG_p' : 0.99, # FIXME: pass the splineSG params as cfg_splineSG
    'splineSG_frame_size' : 10 * units.s,
    'flag_do_tddr' : True,
    'flag_do_imu_glm' : False,
    'cfg_imu_glm' : cfg_imu_glm,
}

cfg_bandpass = { 
    'fmin' : 0.02 * units.Hz,
    'fmax' : 3 * units.Hz
}

cfg_preprocess = {
    'median_filt' : 3, # set to 1 if you don't want to do median filtering
    'cfg_prune' : cfg_prune,
    'cfg_motion_correct' : cfg_motion_correct,
    'cfg_bandpass' : cfg_bandpass
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
    'blockaverage_val' : 0 
    }


cfg_GLM = {
    'drift_order' : 1,
    'distance_threshold' : 20*units.mm, # for ssr
    'short_channel_method' : 'mean',
    'noise_model' : "ols",
    't_delta' : 1*units.s ,   # for seq of Gauss basis func - the temporal spacing between consecutive gaussians
    't_std' : 1*units.s ,     #  the temporal spacing between consecutive gaussians
    }


cfg_blockavg = {
    'trange_hrf' : [5, 35] * units.s,
    'trange_hrf_stat' : [10, 20],
    'stim_lst_hrf' : ['passive', 'active'], # FIXME: why have multiple of this
    'flag_do_GLM_filter' : True,
    'cfg_GLM' : cfg_GLM,
    'flag_save_group_avg_hrf': True,
    'flag_save_each_subj' : False,  # if True, will save the block average data for each subject
    'cfg_mse_conc' : cfg_mse_conc,
    'cfg_mse_od' : cfg_mse_od
    }               # !!! provide list of rec str and whether or not to save weighted for each one


cfg_erbmICA = {}



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

# RUN PREPROCESSING
if not flag_load_preprocessed_data:
    print("Running load and process function")
    rec, chs_pruned_subjs = pfDAB.load_and_preprocess( cfg_dataset, cfg_preprocess, cfg_dqr )
    
    
    # SAVE preprocessed data 
    if flag_save_preprocessed_data:
    
        with gzip.open( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data', 
                                     'chs_pruned_subjs_ts_' + cfg_dataset["file_ids"][0].split('_')[0]+ '.pkl'), 'wb') as f: # !!! FIX ME: naming convention assumes file_ids only includes ONE task
            pickle.dump(chs_pruned_subjs, f, protocol=pickle.HIGHEST_PROTOCOL )
            
        with gzip.open( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data', 
                                     'rec_list_ts_' + cfg_dataset["file_ids"][0].split('_')[0] + '.pkl'), 'wb') as f:
            pickle.dump(rec, f, protocol=pickle.HIGHEST_PROTOCOL )
            
            
        # SAVE cfg params to txt file
        dict_cfg_save = {"cfg_dataset" : cfg_dataset, "cfg_preprocess" : cfg_preprocess, "cfg_GLM" : cfg_GLM, "cfg_blockavg" : cfg_blockavg}
        cfg_save_str = 'cfg_params_' + cfg_dataset["file_ids"][0].split('_')[0] + '.txt'

        with open(rootDir_saveData + cfg_save_str, "w", encoding="utf-8") as f:
                def write_dict(d, level=0):
                    for key, value in d.items():
                        f.write("  " * level + f"{key}: ")
                        if isinstance(value, dict):  # If the value is a sub-dictionary, recurse
                            f.write("\n")
                            write_dict(value, level + 1)
                        else:
                            f.write(f"{value}\n")
                            
                for var_name, dict_data in dict_cfg_save.items():
                        f.write(f"=== {var_name} ===\n")  # Writing the dictionary name
                        if isinstance(dict_data, dict):
                            write_dict(dict_data)
                        else:
                            f.write(str(dict_data) + "\n")  # If it's not a dict, just write its value
                        f.write("\n")  # Separate different dictionaries
        
        
# LOAD in saved data
else:
    print("Loading saved data")
    with gzip.open( os.path.join(rootDir_saveData, 'rec_list_ts_' + cfg_dataset["file_ids"][0].split('_')[0] + '.pkl'), 'rb') as f: # !!! FIX ME: this assumes file_ids only includes ONE task
         rec = pickle.load(f)
    with gzip.open( os.path.join(rootDir_saveData, 'chs_pruned_subjs_ts_' + cfg_dataset["file_ids"][0].split('_')[0] + '.pkl'), 'rb') as f:
         chs_pruned_subjs = pickle.load(f)



# %% ERBM ICA Filtering 
##############################################################################
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






# %% Block Average - unweighted and weighted
##############################################################################

import importlib
importlib.reload(pfDAB_grp_avg)


# FIXME: should not be needed here... shouldbe handled in ICA step above
ica_lpf = 1.0 * units.Hz # MUST be the same as used when creating W_ica
# import pdb
# pdb.set_trace()
if 1:
    rec_str = 'conc_tddr'
    y_mean, y_mean_weighted, y_stderr_weighted, _, _ = pfDAB_grp_avg.run_group_block_average( rec, rec_str, chs_pruned_subjs, cfg_dataset, cfg_blockavg)
    blockaverage_mean = y_mean

    # rec_str = 'conc_tddr_pca'
    # y_mean, y_mean_weighted, y_stderr_weighted, _, _ = pfDAB_grp_avg.run_group_block_average( rec, rec_str, chs_pruned_subjs, cfg_dataset, cfg_blockavg )
    # blockaverage_mean_tmp = y_mean.assign_coords(trial_type=[x + '-pca' for x in y_mean_weighted.trial_type.values])
    # blockaverage_mean = xr.concat([blockaverage_mean, blockaverage_mean_tmp],dim='trial_type')

    # rec_str = 'conc_tddr_ica'
    # y_mean, y_mean_weighted, y_stderr_weighted, _, _ = pfDAB_grp_avg.run_group_block_average( rec, rec_str, chs_pruned_subjs, cfg_dataset, cfg_blockavg )
    # blockaverage_mean_tmp = y_mean.assign_coords(trial_type=[x + '-ica' for x in y_mean_weighted.trial_type.values])
    # blockaverage_mean = xr.concat([blockaverage_mean, blockaverage_mean_tmp],dim='trial_type')

    # rec_str = 'conc_tddr_glm'
    # y_mean, y_mean_weighted, y_stderr_weighted, _, _ = pfDAB_grp_avg.run_group_block_average( rec, rec_str, chs_pruned_subjs, cfg_dataset, cfg_blockavg )
    # blockaverage_mean_tmp = y_mean.assign_coords(trial_type=[x + '-glm' for x in y_mean_weighted.trial_type.values])
    # blockaverage_mean = xr.concat([blockaverage_mean, blockaverage_mean_tmp],dim='trial_type')

if 0:  # !!! sub 3 has no od_imu --- go back and check preprocess for this sub?????????
    # rec_str = 'conc_o_tddr' # just doing this because I want the DQR scalp plot for this
    # y_mean, y_mean_weighted, y_stderr_weighted, _, _ = pfDAB_grp_avg.run_group_block_average( rec, filenm_lst, rec_str, ica_lpf, trange_hrf, stim_lst_hrf, flag_save_each_subj, subj_ids, subj_id_exclude, chs_pruned_subjs, rootDir_data, trange_hrf_stat )

    rec_str = 'od_tddr'
    y_mean, y_mean_weighted, y_stderr_weighted, _, _ = pfDAB_grp_avg.run_group_block_average( rec, rec_str, chs_pruned_subjs, cfg_dataset, cfg_blockavg)
    blockaverage_mean = y_mean
    
    rec_str = 'od_o_tddr'
    y_mean, y_mean_weighted, y_stderr_weighted, y_subj, y_mse_subj = pfDAB_grp_avg.run_group_block_average( rec, rec_str, chs_pruned_subjs, cfg_dataset, cfg_blockavg )
    blockaverage_mean_tmp = y_mean_weighted.assign_coords(trial_type=[x + '-o' for x in y_mean_weighted.trial_type.values])
    blockaverage_mean = xr.concat([blockaverage_mean, blockaverage_mean_tmp],dim='trial_type')
    
    # rec_str = 'od_imu_tddr'
    # y_mean, y_mean_weighted, y_stderr_weighted, y_subj, y_mse_subj = pfDAB_grp_avg.run_group_block_average( rec, rec_str, chs_pruned_subjs, cfg_dataset, cfg_blockavg )
    # blockaverage_mean_tmp = y_mean.assign_coords(trial_type=[x + '-imu' for x in y_mean_weighted.trial_type.values])
    # blockaverage_mean = xr.concat([blockaverage_mean, blockaverage_mean_tmp],dim='trial_type')
    
    # rec_str = 'od_o_imu_tddr'
    # y_mean, y_mean_weighted, y_stderr_weighted, y_subj, y_mse_subj = pfDAB_grp_avg.run_group_block_average( rec, rec_str, chs_pruned_subjs, cfg_dataset, cfg_blockavg )
    # blockaverage_mean_tmp = y_mean_weighted.assign_coords(trial_type=[x + '-o-imu' for x in y_mean_weighted.trial_type.values])
    # blockaverage_mean = xr.concat([blockaverage_mean, blockaverage_mean_tmp],dim='trial_type')

    # # rec_str = 'od_o_tddr_pca'
    # # y_mean, y_mean_weighted_pca, y_stderr_weighted_pca, _, y_mse_subj_pca = pfDAB_grp_avg.run_group_block_average( rec, rec_str, chs_pruned_subjs, cfg_dataset, cfg_blockavg )
    # # blockaverage_mean_tmp = y_mean_weighted_pca.assign_coords(trial_type=[x + '-o-pca' for x in y_mean_weighted.trial_type.values])
    # # blockaverage_mean = xr.concat([blockaverage_mean, blockaverage_mean_tmp],dim='trial_type')

    # # rec_str = 'od_o_tddr_ica'
    # # y_mean, y_mean_weighted_ica, y_stderr_weighted_ica, _, y_mse_subj_ica = pfDAB_grp_avg.run_group_block_average( rec, rec_str, chs_pruned_subjs, cfg_dataset, cfg_blockavg )
    # # blockaverage_mean_tmp = y_mean_weighted_ica.assign_coords(trial_type=[x + '-o-ica' for x in y_mean_weighted.trial_type.values])
    # # blockaverage_mean = xr.concat([blockaverage_mean, blockaverage_mean_tmp],dim='trial_type')

# save the results to a pickle file
blockaverage = blockaverage_mean

if cfg_blockavg['flag_save_each_subj']:
    # FIXME: this assumes the number of subjects and trial_type. Generalize this in the future.
    # blockaverage = blockaverage.sel(trial_type=['ST', 'ST-ica', 'ST-01', 'ST-ica-01', 'ST-02', 'ST-ica-02', 'ST-03', 'ST-ica-03', 'ST-04', 'ST-ica-04', 'ST-05', 'ST-ica-05', 'ST-06', 'ST-ica-06', 'ST-07', 'ST-ica-07', 'ST-08', 'ST-ica-08', 'ST-09', 'ST-ica-09'])
    blockaverage = blockaverage.sel(trial_type=['ST', 'ST-ica', 'ST-01', 'ST-ica-01', 'ST-02', 'ST-ica-02', 'ST-03', 'ST-ica-03', 'ST-04', 'ST-ica-04', 'ST-06', 'ST-ica-06', 'ST-08', 'ST-ica-08', 'ST-09', 'ST-ica-09'])

if cfg_blockavg['flag_save_group_avg_hrf']:
    file_path_pkl = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data', 'blockaverage_IWHD_OD.pkl.gz')
    file = gzip.GzipFile(file_path_pkl, 'wb')
    
    if 'chromo' in blockaverage.dims:
        file.write(pickle.dumps([blockaverage, rec[0][0].geo2d, rec[0][0].geo3d]))
    else:
        # convert to concentration
        dpf = xr.DataArray(
            [1, 1],
            dims="wavelength",
            coords={"wavelength": rec[0][0]['amp'].wavelength},
        )
        foo = blockaverage.copy()
        foo = foo.rename({'reltime':'time'})
        foo.time.attrs['units'] = 'second'
        foo = cedalion.nirs.od2conc(foo, rec[0][0].geo3d, dpf, spectrum="prahl")
        foo = foo.rename({'time':'reltime'})
        foo = foo.transpose('trial_type','chromo','channel','reltime')
    
        file.write(pickle.dumps([foo, rec[0][0].geo2d, rec[0][0].geo3d]))
    
    file.close()
    
    blockaverage_all = blockaverage.copy()
    blockaverage_all_o = blockaverage_all.copy()
    
    print('Saved group average HRF to ' + file_path_pkl)



# %% Load the Sensitivity Matrix and Head Model
##############################################################################
import importlib
importlib.reload(pfDAB_img)

#path_to_dataset = '/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data/Datasets/BallSqueezing_WHHD/'
probe_dir = "/projectnb/nphfnirs/ns/lcarlton/DATA/probes/NN22_WHHD/12NN/fw/"
head_model = 'ICBM152'

Adot, head = pfDAB_img.load_Adot( probe_dir, head_model )



# %% Do the image reconstruction
##############################################################################

import importlib
importlib.reload(pfDAB_img)


trial_type_img = 'DT' #'STS-o' # 'DT', 'DT-ica', 'ST', 'ST-ica'
t_win = (10, 20)

file_save = True
flag_Cmeas = True # if True make sure you are using the correct y_stderr_weighted below -- covariance

BRAIN_ONLY = False
SB = False  # spatial basis

sb_cfg = {
    'mask_threshold': -2,
    'threshold_brain': 5*units.mm,      # threshold_brain / threshold_scalp: Defines spatial limits for brain vs. scalp contributions.
    'threshold_scalp': 20*units.mm,
    'sigma_brain': 5*units.mm,      # sigma_brain / sigma_scalp: Controls smoothing or spatial regularization strength.
    'sigma_scalp': 20*units.mm,
    'lambda1': 0.01,        # regularization params
    'lambda2': 0.1
}

alpha_meas_list = [1e0] #[1e-2, 1e-3, 1e-5] #[1e-3]
alpha_spatial_list = [1e-1]#[1e-2, 1e-4, 1e-5, 1e-3, 1e-1] #[1e-3]


file_path0 = cfg_dataset['root_dir'] + 'derivatives/processed_data/'
wavelength = rec[0][0]['amp'].wavelength.values
spectrum = 'prahl'


#
# Get the group average image
#

if 'chromo' in blockaverage_all.dims:
    # get the group average HRF over a time window
    hrf_conc_mag = blockaverage_all.sel(trial_type=trial_type_img).sel(reltime=slice(t_win[0], t_win[1])).mean('reltime')
    hrf_conc_ts = blockaverage_all.sel(trial_type=trial_type_img)

    # convert back to OD
    E = cedalion.nirs.get_extinction_coefficients(spectrum, wavelength)
    hrf_od_mag = xr.dot(E, hrf_conc_mag * 1*units.mm * 1e-6*units.molar / units.micromolar, dim=["chromo"]) # assumes DPF = 1
    hrf_od_ts = xr.dot(E, hrf_conc_ts * 1*units.mm * 1e-6*units.molar / units.micromolar, dim=["chromo"]) # assumes DPF = 1
else:
    hrf_od_mag = blockaverage_all.sel(trial_type=trial_type_img).sel(reltime=slice(t_win[0], t_win[1])).mean('reltime')
    hrf_od_ts = blockaverage_all.sel(trial_type=trial_type_img)


if not flag_Cmeas:    
    X_grp, W, C, D = pfDAB_img.do_image_recon( hrf_od_mag, head, Adot, None, wavelength, BRAIN_ONLY, SB, sb_cfg, alpha_spatial_list, alpha_meas_list, file_save, file_path0, trial_type_img)
else:
    trial_type_img_split = trial_type_img.split('-')
    C_meas = y_stderr_weighted.sel(trial_type=trial_type_img_split[0]).sel(reltime=slice(t_win[0], t_win[1])).mean('reltime') # FIXME: what is the correct error estimate?
    C_meas = C_meas.pint.dequantify()     # remove units
    C_meas = C_meas**2  # get variance
    C_meas = C_meas.stack(measurement=('channel', 'wavelength')).sortby('wavelength')  # !!! assumes y_stderr_weighted is in  OD - FIX
    X_grp, W, C, D = pfDAB_img.do_image_recon( hrf_od_mag, head, Adot, C_meas, wavelength, BRAIN_ONLY, SB, sb_cfg, alpha_spatial_list, alpha_meas_list, file_save, file_path0, trial_type_img)

print('Done with Image Reconstruction')




# %% Calculate the image noise and image CNR
##############################################################################

# scale columns of W by y_stderr_weighted**2
cov_img_tmp = W * np.sqrt(C_meas.values) # W is pseudo inverse  --- diagonal (faster than W C W.T)
cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)

nV = X_grp.shape[0]
cov_img_diag = np.reshape( cov_img_diag, (2,nV) ).T

# image noise
X_noise = X_grp.copy()
X_noise.values = np.sqrt(cov_img_diag)

filepath = os.path.join(file_path0, f'X_noise_{trial_type_img}_cov_alpha_spatial_{alpha_spatial_list[-1]:.0e}_alpha_meas_{alpha_meas_list[-1]:.0e}.pkl.gz')
print(f'   Saving to X_noise_{trial_type_img}_cov_alpha_spatial_{alpha_spatial_list[-1]:.0e}_alpha_meas_{alpha_meas_list[-1]:.0e}.pkl.gz')
file = gzip.GzipFile(filepath, 'wb')
file.write(pickle.dumps([X_noise, alpha_meas_list[-1], alpha_spatial_list[-1]]))
file.close()  

# image t-stat (i.e. CNR)
X_tstat = X_grp / np.sqrt(cov_img_diag)

X_tstat[ np.where(cov_img_diag[:,0]==0)[0], 0 ] = 0
X_tstat[ np.where(cov_img_diag[:,1]==0)[0], 1 ] = 0

filepath = os.path.join(file_path0, f'X_tstat_{trial_type_img}_cov_alpha_spatial_{alpha_spatial_list[-1]:.0e}_alpha_meas_{alpha_meas_list[-1]:.0e}.pkl.gz')
print(f'   Saving to X_tstat_{trial_type_img}_cov_alpha_spatial_{alpha_spatial_list[-1]:.0e}_alpha_meas_{alpha_meas_list[-1]:.0e}.pkl.gz')
file = gzip.GzipFile(filepath, 'wb')
file.write(pickle.dumps([X_tstat, alpha_meas_list[-1], alpha_spatial_list[-1]]))
file.close()     


# %% Get image for each subject and do weighted average
##############################################################################
import importlib
importlib.reload(pfDAB_img)


file_save = False
trial_type_img = 'DT' # 'STS', 'DT-ica', 'ST', 'ST-ica'
t_win = (10, 20)

BRAIN_ONLY = False
SB = False

sb_cfg = {
    'mask_threshold': -2,
    'threshold_brain': 5*units.mm,
    'threshold_scalp': 20*units.mm,
    'sigma_brain': 5*units.mm,
    'sigma_scalp': 20*units.mm,
    'lambda1': 0.01,
    'lambda2': 0.1
}

alpha_meas_list = [1e0] #[1e-2, 1e-3, 1e-5] #[1e-3]
alpha_spatial_list = [1e-1]#[1e-2, 1e-4, 1e-5, 1e-3, 1e-1] #[1e-3]


file_path0 = cfg_dataset['root_dir'] + 'derivatives/processed_data/'
wavelength = rec[0][0]['amp'].wavelength.values
spectrum = 'prahl'


X_hrf_mag_subj = None
C = None # spatial regularization 
D = None

# !!! go thru each trial type (outside function)
    # 
for idx_subj in range(n_subjects):

# !!!vstart trial type loop
    hrf_od_mag = y_subj.sel(subj=cfg_dataset['subj_ids'][idx_subj]).sel(trial_type=trial_type_img).sel(reltime=slice(t_win[0], t_win[1])).mean('reltime')
    # hrf_od_ts = blockaverage_all.sel(trial_type=trial_type_img)

    # get the image
    trial_type_img_split = trial_type_img.split('-')
    C_meas = y_mse_subj.sel(subj=cfg_dataset['subj_ids'][idx_subj]).sel(reltime=slice(t_win[0], t_win[1])).mean('reltime').mean('trial_type') # FIXME: handle more than one trial_type -- he was getting rid of trial type dim

#    trial_type_img_split = trial_type_img.split('-')
    C_meas = y_mse_subj.sel(subj=cfg_dataset['subj_ids'][idx_subj]).sel(trial_type=trial_type_img).sel(reltime=slice(t_win[0], t_win[1])).mean('reltime') # FIXME: handle more than one trial_type

    C_meas = C_meas.pint.dequantify()
    C_meas = C_meas.stack(measurement=('channel', 'wavelength')).sortby('wavelength')
    if C is None or D is None:
        X_hrf_mag_tmp, W, C, D = pfDAB_img.do_image_recon( hrf_od_mag, head, Adot, C_meas, wavelength, BRAIN_ONLY, SB, sb_cfg, alpha_spatial_list, alpha_meas_list, file_save, file_path0, trial_type_img) 
    else:
        X_hrf_mag_tmp, W, _, _ = pfDAB_img.do_image_recon( hrf_od_mag, head, Adot, C_meas, wavelength, BRAIN_ONLY, SB, sb_cfg, alpha_spatial_list, alpha_meas_list, file_save, file_path0, trial_type_img, None, C, D)

    # get image noise
    cov_img_tmp = W * np.sqrt(C_meas.values) # get diag of image covariance
    cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)

    nV = X_hrf_mag_tmp.vertex.size
    cov_img_diag = np.reshape( cov_img_diag, (2,nV) ).T

    X_mse = X_hrf_mag_tmp.copy() 
    X_mse.values = cov_img_diag
    
    # !!! end trial type loop

    # weighted average -- same as chan space - but now is vertex space
    if X_hrf_mag_subj is None:
        X_hrf_mag_subj = X_hrf_mag_tmp
        X_hrf_mag_subj = X_hrf_mag_subj.expand_dims('subj')
        X_hrf_mag_subj = X_hrf_mag_subj.assign_coords(subj=[cfg_dataset['subj_ids'][idx_subj]])

        X_mse_subj = X_mse.copy()
        X_mse_subj = X_mse_subj.expand_dims('subj')
        X_mse_subj = X_mse_subj.assign_coords(subj=[cfg_dataset['subj_ids'][idx_subj]])

        X_hrf_mag_weighted = X_hrf_mag_tmp / X_mse
        X_mse_inv_weighted = 1 / X_mse
        X_mse_inv_weighted_max = 1 / X_mse
    elif cfg_dataset['subj_ids'][idx_subj] not in subj_id_exclude:
        X_hrf_mag_subj_tmp = X_hrf_mag_tmp.expand_dims('subj') # !!! will need to expand dims to get back trial type -- can do in function 
        X_hrf_mag_subj_tmp = X_hrf_mag_subj_tmp.assign_coords(subj=[cfg_dataset['subj_ids'][idx_subj]])

        X_mse_subj_tmp = X_mse.copy().expand_dims('subj')
        X_mse_subj_tmp = X_mse_subj_tmp.assign_coords(subj=[cfg_dataset['subj_ids'][idx_subj]])

        X_hrf_mag_subj = xr.concat([X_hrf_mag_subj, X_hrf_mag_subj_tmp], dim='subj')
        X_mse_subj = xr.concat([X_mse_subj, X_mse_subj_tmp], dim='subj')

        X_hrf_mag_weighted = X_hrf_mag_weighted + X_hrf_mag_tmp / X_mse
        X_mse_inv_weighted = X_mse_inv_weighted + 1 / X_mse
        X_mse_inv_weighted_max = np.maximum(X_mse_inv_weighted_max, 1 / X_mse)
    else:
        print(f"   Subject {cfg_dataset['subj_ids'][idx_subj]} excluded from group average")


# %%

# get the average
X_hrf_mag_mean = X_hrf_mag_subj.mean('subj')
X_hrf_mag_mean_weighted = X_hrf_mag_weighted / X_mse_inv_weighted

X_mse_mean_within_subject = 1 / X_mse_inv_weighted

X_mse_subj_tmp = X_mse_subj.copy()
X_mse_subj_tmp = xr.where(X_mse_subj_tmp < 1e-6, 1e-6, X_mse_subj_tmp)
X_mse_weighted_between_subjects_tmp = (X_hrf_mag_subj - X_hrf_mag_mean)**2 / X_mse_subj_tmp # X_mse_subj_tmp is weights for each sub
X_mse_weighted_between_subjects = X_mse_weighted_between_subjects_tmp.mean('subj')
X_mse_weighted_between_subjects = X_mse_weighted_between_subjects / (X_mse_subj**-1).mean('subj')

X_stderr_weighted = np.sqrt( X_mse_mean_within_subject + X_mse_weighted_between_subjects )

X_tstat = X_hrf_mag_mean_weighted / X_stderr_weighted

X_weight_sum = X_mse_inv_weighted / X_mse_inv_weighted_max 
# FIXME: I am trying to get something like number of subjects per vertex...
# maybe I need to change X_mse_inv_weighted_max to be some typical value 
# because when all subjects have a really low value, then it won't scale the way I want

#    blockaverage_stderr_weighted = blockaverage_stderr_weighted.assign_coords(trial_type=blockaverage_mean_weighted.trial_type)

filepath = os.path.join(file_path0, f'Xs_{trial_type_img}_cov_alpha_spatial_{alpha_spatial_list[-1]:.0e}_alpha_meas_{alpha_meas_list[-1]:.0e}.pkl.gz')
print(f'   Saving to Xs_{trial_type_img}_cov_alpha_spatial_{alpha_spatial_list[-1]:.0e}_alpha_meas_{alpha_meas_list[-1]:.0e}.pkl.gz')
file = gzip.GzipFile(filepath, 'wb')
file.write(pickle.dumps([X_weight_sum, alpha_meas_list[-1], alpha_spatial_list[-1]]))
file.close()     


# %%
threshold = -2 # log10 absolute
wl_idx = 1
M = sbf.get_sensitivity_mask(Adot, threshold, wl_idx)


# %% Plot the images
##############################################################################
import importlib
importlib.reload(pfDAB_img)

flag_hbo = True
flag_brain = True
flag_img = 'noise' # 'tstat', 'mag', 'noise'
flag_condition = 'STS' # 'ST', 'DT', 'STS'


if flag_hbo:
    title_str = 'HbO'
    hbx_brain_scalp = 'hbo'
else:
    title_str = 'HbR'
    hbx_brain_scalp = 'hbr'

if flag_brain:
    title_str = title_str + ' brain'
    hbx_brain_scalp = hbx_brain_scalp + '_brain'
else:
    title_str = title_str + ' scalp'
    hbx_brain_scalp = hbx_brain_scalp + '_scalp'

if flag_img == 'tstat':
    foo_img = X_tstat.copy()
    title_str = title_str + ' t-stat'
elif flag_img == 'mag':
    foo_img = X_hrf_mag_mean_weighted.copy()
    title_str = title_str + ' magnitude'
elif flag_img == 'noise':
    foo_img = X_stderr_weighted.copy()
    title_str = title_str + ' noise'
#    foo_img.values = np.log10(foo_img.values)+7


# title_str = 'HbR'
# hbx_brain_scalp = 'hbr_brain'
# foo_img = X_hrf_mag_mean_weighted

# title_str = 'HbR t-stat'
# hbx_brain_scalp = 'hbr_brain'
# foo_img = X_tstat




foo_img[~M] = np.nan
# foo_img = xr.where(np.abs(foo_img) < 1.86, np.nan, foo_img) # one-tail is 1.86 and two tail is 2.3



p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (1,1), hbx_brain_scalp, 'scale_bar', None, title_str)
p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (0,0), hbx_brain_scalp, 'left', p0)
p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (0,1), hbx_brain_scalp, 'superior', p0)
p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (0,2), hbx_brain_scalp, 'right', p0)
p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (1,0), hbx_brain_scalp, 'anterior', p0)
p0 = pfDAB_img.plot_image_recon(foo_img, head, (2,3), (1,2), hbx_brain_scalp, 'posterior', p0)

p0.screenshot( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'plots', f'IMG.png') )
p0.close()


# %%
X_foo = X_tstat.copy()
X_foo[:,0] = 0

# select parcels
# parcels with '_LH' at the end
parcels = np.unique(X_grp['parcel'].values)
parcels_LH = [x for x in parcels if x.endswith('_LH')]

parcels_sel = [x for x in parcels_LH if 'DefaultB_PFCv' in x]

X_foo[np.isin(X_foo['parcel'].values, parcels_sel), 0] = 1


p0 = pfDAB_img.plot_image_recon(X_foo, head, 'hbo_brain', 'left')



# %% MNI coordinates
head_ras = head.apply_transform(head.t_ijk2ras)

# brain indices
idx_brain = np.where(Adot.is_brain)[0]

# make an xarray associating parcels with MNI coordinates
parcels_mni_xr = xr.DataArray(
    head_ras.brain.mesh.vertices[idx_brain,:],
    dims = ('vertex', 'coord'),
    coords = {'parcel': ('vertex', Adot.coords['parcel'].values[idx_brain])},
)

# get MNI coordinates of a specific parcel 'VisCent_ExStr_11_LH'
parcel_specific = parcels_mni_xr.where(parcels_mni_xr['parcel'] == 'VisCent_ExStr_11_LH', drop=True)

# find the parcel closest to a specific MNI coordinate
mni_coord = np.array([[ -27.1, -100.1 ,    9.4]])
dist = np.linalg.norm(parcels_mni_xr.values - mni_coord, axis=1)
parcel_closest = parcels_mni_xr[np.argmin(dist)]
print(f'Parcel closest to {mni_coord} is {parcel_closest["parcel"].values} with MNI coordinates {parcel_closest.values}')
print(f'Distance is {np.min(dist):0.2f} mm')

# %% Parcels
##############################################################################
# list unique parcels in X
parcels = np.unique(X_grp['parcel'].values)

# parcels with '_LH' at the end
parcels_LH = [x for x in parcels if x.endswith('_LH')]

# select parcels with a specific name
parcels_sel = [x for x in parcels_LH if 'DefaultB_PFCv' in x]



Xo = X_tstat.sel(chromo='HbO')

# Create a mapping from vertex to parcel
vertex_to_parcel = Xo['parcel'].values

# Add the parcel information as a coordinate to the DataArray/Dataset
Xo = Xo.assign_coords(parcel=('vertex', vertex_to_parcel))

# Group by the parcel coordinate and calculate the mean over the vertex dimension
Xo_parcel = Xo.groupby('parcel').mean(dim='vertex')


if 0: # find Xo_parcel values > 2 and from parcels_LH
    Xo_parcel_2 = Xo_parcel.where(np.abs(Xo_parcel) > 1).dropna('parcel').where(Xo_parcel['parcel'].isin(parcels_LH)).dropna('parcel')
else: # find Xo_parcel values > 2 and from parcels_sel
    Xo_parcel_2 = Xo_parcel.where(np.abs(Xo_parcel) > 1).dropna('parcel').where(Xo_parcel['parcel'].isin(parcels_sel)).dropna('parcel')

X_foo = X_tstat.copy()
X_foo[:,0] = 0
X_foo[np.isin(X_foo['parcel'].values, np.unique(Xo_parcel_2['parcel'].values) ), 0] = 1



od_ts = hrf_od_ts.stack(measurement=('channel', 'wavelength')).sortby('wavelength').T
X_grp_ts = W @ od_ts.values

split = len(X_grp_ts)//2
X_grp_ts = X_grp_ts.reshape([2, split, X_grp_ts.shape[1]])
X_grp_ts = X_grp_ts.transpose(1,0,2)

X_grp_ts = xr.DataArray(X_grp_ts,
                    dims = ('vertex', 'chromo', 'reltime'),
                    coords = {'chromo': ['HbO', 'HbR'],
                            'parcel': ('vertex',Adot.coords['parcel'].values),
                            'is_brain':('vertex', Adot.coords['is_brain'].values),
                            'reltime': od_ts.reltime.values},
                    )
X_grp_ts = X_grp_ts.set_xindex("parcel")



# get the time series for the parcels
Xo_ts = X_grp_ts #.sel(chromo='HbO')
vertex_to_parcel = Xo_ts['parcel'].values
Xo_ts = Xo_ts.assign_coords(parcel=('vertex', vertex_to_parcel))
Xo_ts_parcel = Xo_ts.groupby('parcel').mean(dim='vertex')

# plot the significant parcels
foo = Xo_ts_parcel.sel(parcel=Xo_parcel_2.parcel.values)

f, ax = p.subplots(1, 1, figsize=(7, 5))
for i in range(foo.sizes['parcel']):
    line, = ax.plot(foo['reltime'], foo.sel(parcel=foo['parcel'][i], chromo='HbO'), label=foo['parcel'][i].values)
    ax.plot(foo['reltime'], foo.sel(parcel=foo['parcel'][i], chromo='HbR'), linestyle='--', color=line.get_color())
ax.set_title('Significant parcels')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Concentration (M)')
ax.legend()
p.show()

p0 = pfDAB_img.plot_image_recon(X_foo, head, 'hbo_brain', 'left')









# %% Old Code
##############################################################################
##############################################################################

