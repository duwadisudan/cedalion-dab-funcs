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
import module_image_recon as img_recon 
import module_spatial_basis_funs_ced as sbf 


# Turn off all warnings
import warnings
warnings.filterwarnings('ignore')

import pyvista as pv

#%%
# Have summed wt for each vertices across subjects


# %% Initial root directory and analysis parameters
##############################################################################

cfg_dataset = {  # !!! NOTE: this needs to be the same as what was used for preproc and blockavg
    'root_dir' : "/projectnb/nphfnirs/ns/Shannon/Data/Interactive_Walking_HD/",
    'subj_ids' : ['01','02','03','04','05','06','07','08','09','10', '11', '12', '13', '14', '15', '16', '17', '18', '19'], 
    'file_ids' : ['IWHD_run-01'],
    'subj_id_exclude' : ['10', '15', '16', '17'] # if you want to exclude a subject from the group average
}
subj_ids_new = [s for s in cfg_dataset['subj_ids'] if s not in cfg_dataset['subj_id_exclude']]


cfg_sb = {
    'mask_threshold': -2,
    'threshold_brain': 5*units.mm,      # threshold_brain / threshold_scalp: Defines spatial limits for brain vs. scalp contributions.
    'threshold_scalp': 20*units.mm,
    'sigma_brain': 5*units.mm,      # sigma_brain / sigma_scalp: Controls smoothing or spatial regularization strength.
    'sigma_scalp': 20*units.mm,
    'lambda1': 0.01,        # regularization params
    'lambda2': 0.1
}


cfg_img_recon = {
    'probe_dir' : "/projectnb/nphfnirs/s/users/lcarlton/DATA/probes/NN22_WHHD/12NN/fw/",
    'head_model' : 'ICBM152',
    'img_recon_on_group' : False,
    't_win' : (10, 20), 
    'DIRECT' : True,  # If true, does direct method, False = does indirect
    'flag_Cmeas' : False,   # if True make sure you are using the correct y_stderr_weighted below (or blockaverage_stderr now)-- covariance
    'BRAIN_ONLY' : False,
    'SB' : False,    # spatial basis
    'alpha_meas' : 1e-2,  #[1e0]    measurement regularization (w/ Cmeas, 1 is good)  (w/out Cmeas do 1e-2?)
    'alpha_spatial' : 1e-1,    #  spatial reg , small pushes deeper into the brain   -- # use smaller alpha spatial od 10^-2 or -3 w/out cmeas
    'spectrum' : 'prahl',
    'cfg_sb' : cfg_sb,
    'flag_save_img_results' : False
    }

mse_min_thresh = 1e-3 

save_path = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'processed_data')


#%% Load Saved data

#save_str = '_imuGLM_tddr_GLMfilt_unpruned_OD'
#filname =  'blockaverage_' + cfg_dataset["file_ids"][0].split('_')[0] + '_' + save_str + '.pkl.gz'

print("Loading saved data")
filname = 	'blockaverage_IWHD_imuGLM_tddr_GLMfilt_unpruned_OD.pkl.gz'
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


# !!! add flag for if doing image recon on group avg or direct or indirect
# !!! ADD flag for if doing image recon on ts or hrf mag ?


# %% Get the group average image
#

# Load the Sensitivity Matrix and Head Model
#
wavelength = blockaverage_all.wavelength.values   
Adot, head = img_recon.load_Adot( cfg_img_recon['probe_dir'], cfg_img_recon['head_model'])

if cfg_img_recon['img_recon_on_group']:
    
    all_trial_X_grp = None
    
    for idx, trial_type in enumerate(blockaverage_all.trial_type):  #enumerate([blockaverage_all.trial_type.values[2]]):
        
        print(f'Getting images for trial type = {trial_type.values}')
        
        if 'chromo' in blockaverage_all.dims:
            # get the group average HRF over a time window
            hrf_conc_mag = blockaverage_all.sel(trial_type=trial_type).sel(reltime=slice(cfg_img_recon['t_win'][0],cfg_img_recon['t_win'][1])).mean('reltime')
            hrf_conc_ts = blockaverage_all.sel(trial_type=trial_type)
            
            blockaverage_stderr_conc = blockaverage_stderr.sel(trial_type=trial_type) # need to convert blockaverage_stderr to od if its in conc
        
            # convert back to OD
            E = cedalion.nirs.get_extinction_coefficients(cfg_img_recon['spectrum'], wavelength)
            hrf_od_mag = xr.dot(E, hrf_conc_mag * 1*units.mm * 1e-6*units.molar / units.micromolar, dim=["chromo"]) # assumes DPF = 1
            hrf_od_ts = xr.dot(E, hrf_conc_ts * 1*units.mm * 1e-6*units.molar / units.micromolar, dim=["chromo"]) # assumes DPF = 1
            
            blockaverage_stderr = xr.dot(E, blockaverage_stderr_conc * 1*units.mm * 1e-6*units.molar / units.micromolar, dim=["chromo"]) # assumes DPF = 1
                
        else:
            hrf_od_mag = blockaverage_all.sel(trial_type=trial_type).sel(reltime=slice(cfg_img_recon['t_win'][0], cfg_img_recon['t_win'][1])).mean('reltime')
            hrf_od_ts = blockaverage_all.sel(trial_type=trial_type)
        
        if not cfg_img_recon['flag_Cmeas']:  
            cov_str = '' # for name
            X_grp, W, C, D = img_recon.do_image_recon_DB( hrf_od_mag, head, Adot, None, wavelength, cfg_img_recon, trial_type, save_path)
        
        else:
            cov_str = 'cov'
           
            C_meas = blockaverage_stderr.sel(trial_type=trial_type).sel(reltime=slice(cfg_img_recon['t_win'][0], cfg_img_recon['t_win'][1])).mean('reltime') 
            C_meas = C_meas.pint.dequantify()     # remove units
            C_meas = C_meas**2  # get variance
            C_meas = C_meas.stack(measurement=('channel', 'wavelength')).sortby('wavelength')  
            X_grp, W, C, D = img_recon.do_image_recon_DB( hrf_od_mag, head, Adot, C_meas, wavelength, cfg_img_recon, trial_type, save_path)
        
        print(f'Done with Image Reconstruction for trial type = {trial_type.values}')
      
        X_grp = X_grp.assign_coords(trial_type = trial_type)
        
        #
        #  Calculate the image noise and image CNR
        #
        if cfg_img_recon['flag_Cmeas']:
            X_noise, X_tstat = img_recon.img_noise_tstat(X_grp, W, C_meas)
            
            if cfg_img_recon['flag_save_img_results']:
                img_recon.save_image_results(X_noise, 'X_noise', save_path, trial_type, cfg_img_recon)
                img_recon.save_image_results(X_tstat, 'X_tstat', save_path, trial_type, cfg_img_recon)
            
            X_noise = X_noise.assign_coords(trial_type = trial_type)
            X_tstat = X_tstat.assign_coords(trial_type = trial_type)
            
            # save results for all trial types
            if all_trial_X_grp is None:
                all_trial_X_grp = X_grp
                all_trial_X_noise = X_noise  # comes from diag of covariance matrix
                all_trial_X_tstat = X_tstat 
            else:
                all_trial_X_grp = xr.concat([all_trial_X_grp, X_grp], dim='trial_type')
                all_trial_X_noise = xr.concat([all_trial_X_noise, X_noise], dim='trial_type')
                all_trial_X_tstat = xr.concat([all_trial_X_tstat, X_tstat], dim='trial_type')
                
            results_img_grp = {'X_grp_all_trial': all_trial_X_grp,
                       'X_noise_grp_all_trial': all_trial_X_noise,
                       'X_tstat_grp_all_trial': all_trial_X_tstat
                       }
        
        # if flag_Cmeas is false, can't calc tstat and noise
        else:
            if all_trial_X_grp is None:
                all_trial_X_grp = X_grp
            else:
                all_trial_X_grp = xr.concat([all_trial_X_grp, X_grp], dim='trial_type')
        
    tasknm = cfg_dataset["file_ids"][0].split('_')[0] # get task name
    
    filepath = os.path.join(cfg_dataset['root_dir'], f'X_{tasknm}_alltrials_{cov_str}_alpha_spatial_{cfg_img_recon["alpha_spatial_list"][-1]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas_list"][-1]:.0e}.pkl.gz')
    print(f'   Saving to X_{tasknm}_alltrials_{cov_str}_alpha_spatial_{cfg_img_recon["alpha_spatial_list"][-1]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas_list"][-1]:.0e}.pkl.gz')
    file = gzip.GzipFile(filepath, 'wb')
    file.write(pickle.dumps(results_img_grp))
    file.close()    


# %% LC:
#%% load head model 
probe_dir = "/projectnb/nphfnirs/s/users/lcarlton/DATA/probes/NN22_WHHD/12NN/" # just for this code t work
head, PARCEL_DIR = img_recon.load_head_model(cfg_img_recon['head_model'], with_parcels=False)
Adot, meas_list, geo3d, amp = img_recon.load_probe(probe_dir, snirf_name='fullhead_56x144_System2.snirf')


ec = cedalion.nirs.get_extinction_coefficients('prahl', Adot.wavelength)
einv = cedalion.xrutils.pinv(ec)

#%% run image recon

"""
do the image reconstruction of each subject independently 
- this is the unweighted subject block average magnitude 
- then reconstruct their individual MSE
- then get the weighted average in image space 
- get the total standard error using between + within subject MSE 
"""
threshold = -2 # log10 absolute
wl_idx = 1
M = sbf.get_sensitivity_mask(Adot, threshold, wl_idx)


#ind_subj_blockavg = all_results['ind_subj_blockavg_unweighted']
ind_subj_blockavg = groupavg_results['blockaverage_subj']  # !!! assuming these vars map to same data ?
#ind_subj_mse = all_results['ind_subj_mse']
ind_subj_mse = groupavg_results['blockaverage_mse_subj']

F = None
D = None
G = None
# import pdb
# pdb.set_trace()
all_trial_X_hrf_mag = None

for trial_type in ind_subj_blockavg.trial_type:
    
    print(f'Getting images for trial type = {trial_type.values}')
    all_subj_X_hrf_mag = None
    
    for subj in ind_subj_blockavg.subj:
        print(f'Calculating subject = {subj.values}')

        od_hrf = ind_subj_blockavg.sel(subj=subj, trial_type=trial_type) 
        # od_hrf = od_hrf.stack(measurement=('channel', 'wavelength')).sortby('wavelength')

        od_mse = ind_subj_mse.sel(subj=subj, trial_type=trial_type).drop_vars(['subj', 'trial_type'])
        
        od_hrf_mag = od_hrf.sel(reltime=slice(cfg_img_recon['t_win'][0], cfg_img_recon['t_win'][1])).mean('reltime')
        od_mse_mag = od_mse.sel(reltime=slice(cfg_img_recon['t_win'][0], cfg_img_recon['t_win'][1])).mean('reltime')
        
        C_meas = od_mse_mag.pint.dequantify()
        C_meas = C_meas.stack(measurement=('channel', 'wavelength')).sortby('wavelength')
        C_meas = xr.where(C_meas < mse_min_thresh, mse_min_thresh, C_meas)

            
        # pdb.set_trace()

        X_hrf_mag, W, D, F, G = img_recon.do_image_recon(od_hrf_mag, head = head, Adot = Adot, C_meas_flag = cfg_img_recon['flag_Cmeas'], C_meas = C_meas, 
                                                    wavelength = [760,850], BRAIN_ONLY = cfg_img_recon['BRAIN_ONLY'], DIRECT = cfg_img_recon['DIRECT'], SB = cfg_img_recon['SB'], 
                                                    cfg_sbf = cfg_img_recon['cfg_sb'], alpha_spatial = cfg_img_recon['alpha_spatial'], alpha_meas = cfg_img_recon['alpha_meas'],
                                                    F = F, D = D, G = G)

        
        
        # pdb.set_trace()
        X_mse = img_recon.get_image_noise(C_meas, X_hrf_mag, W, DIRECT = cfg_img_recon['DIRECT'], SB= cfg_img_recon['SB'], G=G)
        
        # X_mse_o = X_mse.copy()

        # weighted average -- same as chan space - but now is vertex space
        if all_subj_X_hrf_mag is None:
            
            all_subj_X_hrf_mag = X_hrf_mag
            all_subj_X_hrf_mag = all_subj_X_hrf_mag.assign_coords(subj=subj)
            all_subj_X_hrf_mag = all_subj_X_hrf_mag.assign_coords(trial_type=trial_type)

            all_subj_X_mse = X_mse
            all_subj_X_mse = all_subj_X_mse.assign_coords(subj=subj)
            all_subj_X_mse = all_subj_X_mse.assign_coords(trial_type=trial_type)

            X_hrf_mag_weighted = X_hrf_mag / X_mse
            X_mse_inv_weighted = 1 / X_mse   # X_mse = mse for 1 subject across all vertices , inverse is wt
            
        else:

            X_hrf_mag_tmp = X_hrf_mag.assign_coords(subj=subj)
            X_hrf_mag_tmp = X_hrf_mag_tmp.assign_coords(trial_type=trial_type)

            X_mse_tmp = X_mse.assign_coords(subj=subj)
            X_mse_tmp = X_mse_tmp.assign_coords(trial_type=trial_type)

            all_subj_X_hrf_mag = xr.concat([all_subj_X_hrf_mag, X_hrf_mag_tmp], dim='subj')
            all_subj_X_mse = xr.concat([all_subj_X_mse, X_mse_tmp], dim='subj')

            X_hrf_mag_weighted = X_hrf_mag_weighted + X_hrf_mag_tmp / X_mse
            X_mse_inv_weighted = X_mse_inv_weighted + 1 / X_mse       # summing weight over all subjects -- viz X_mse_inv_weighted will tell us which regions of brain we are most conf in
        # END OF SUBJECT LOOP

    # get the average
    X_hrf_mag_mean = all_subj_X_hrf_mag.mean('subj')
    X_hrf_mag_mean_weighted = X_hrf_mag_weighted / X_mse_inv_weighted
    
    X_mse_mean_within_subject = 1 / X_mse_inv_weighted
    X_mse_mean_within_subject = X_mse_mean_within_subject.assign_coords({'trial_type': trial_type})
    
    X_mse_subj_tmp = all_subj_X_mse # PLOT THIS
    
    # temp = all_subj_X_mse.copy()
    # temp[: ~M] = np.nan
    # temp = np.log10(temp.sel(vertex=all_subj_X_mse.is_brain.values).stack(val=('vertex', 'chromo', 'subj')))
    # temp[np.isneginf(temp)] = np.nan
    
    # plt.hist(temp, bins=100)
    # plt.axvline(np.log10(mse_min_thresh), color='k')
    
    # X_mse_subj_tmp = xr.where(X_mse_subj_tmp < mse_min_thresh, mse_min_thresh, X_mse_subj_tmp)
    X_mse_weighted_between_subjects_tmp = (all_subj_X_hrf_mag - X_hrf_mag_mean)**2 / X_mse_subj_tmp # X_mse_subj_tmp is weights for each sub
    X_mse_weighted_between_subjects = X_mse_weighted_between_subjects_tmp.mean('subj')
    X_mse_weighted_between_subjects = X_mse_weighted_between_subjects * X_mse_mean_within_subject # / (all_subj_X_mse**-1).mean('subj')
    X_mse_weighted_between_subjects = X_mse_weighted_between_subjects.pint.dequantify()
    
    X_stderr_weighted = np.sqrt( X_mse_mean_within_subject + X_mse_weighted_between_subjects )
    
    X_tstat = X_hrf_mag_mean_weighted / X_stderr_weighted
    
    if all_trial_X_hrf_mag is None:
        
        all_trial_X_hrf_mag = X_hrf_mag_mean
        all_trial_X_hrf_mag_weighted = X_hrf_mag_mean_weighted
        all_trial_X_stderr = X_stderr_weighted
        all_trial_X_tstat = X_tstat
        all_trial_X_mse_between = X_mse_weighted_between_subjects
        all_trial_X_mse_within = X_mse_mean_within_subject
    else:

        all_trial_X_hrf_mag = xr.concat([all_trial_X_hrf_mag, X_hrf_mag_mean], dim='trial_type')
        all_trial_X_hrf_mag_weighted = xr.concat([all_trial_X_hrf_mag_weighted, X_hrf_mag_mean_weighted], dim='trial_type')
        all_trial_X_stderr = xr.concat([all_trial_X_stderr, X_stderr_weighted], dim='trial_type')
        all_trial_X_tstat = xr.concat([all_trial_X_tstat, X_tstat], dim='trial_type')
        all_trial_X_mse_between = xr.concat([all_trial_X_mse_between, X_mse_weighted_between_subjects], dim='trial_type')
        all_trial_X_mse_within = xr.concat([all_trial_X_mse_within, X_mse_mean_within_subject], dim='trial_type')

# END OF TRIAL TYPE LOOP
results = {'X_hrf_mag': all_trial_X_hrf_mag,
           'X_hrf_mag_weighted': all_trial_X_hrf_mag_weighted,
           'X_std_err': all_trial_X_stderr,
           'X_tstat': all_trial_X_tstat,
           'X_mse_between': all_trial_X_mse_between,
           'X_mse_within': all_trial_X_mse_within
           }
if cfg_img_recon['DIRECT']:
    direct_name = 'direct'
else:
    direct_name = 'indirect'
    
if cfg_img_recon['SB']:
    SB_name = 'SB'
else:
    SB_name = 'noSB'

if cfg_img_recon['flag_Cmeas']:
    Cmeas_name = 'Cmeas'
else:
    Cmeas_name = 'noCmeas'
    
filepath = os.path.join(cfg_dataset['root_dir'], f'Xs_cov_alpha_spatial_{cfg_img_recon["alpha_spatial"]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas"]:.0e}_{direct_name}_{Cmeas_name}_{SB_name}.pkl.gz')
print(f'   Saving to Xs_cov_alpha_spatial_{cfg_img_recon["alpha_spatial"]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas"]:.0e}_{direct_name}_{Cmeas_name}_{SB_name}.pkl.gz')
file = gzip.GzipFile(filepath, 'wb')
file.write(pickle.dumps(results))
file.close()     


#%% build plots 
import importlib
importlib.reload(img_recon)

threshold = -2 # log10 absolute
wl_idx = 1
M = sbf.get_sensitivity_mask(Adot, threshold, wl_idx)
SAVE = True
flag_hbo_list = [True, False]
flag_brain_list = [True, False]
flag_img_list = ['mag', 'tstat', 'noise'] #, 'noise'
flag_condition_list = ['ST', 'DT'] # 'ST', 'DT', 'STS'

# with gzip.open( filepath, 'rb') as f:
#      results = pickle.load(f)

# all_trial_X_hrf_mag = results['X_hrf_mag']
for flag_hbo in flag_hbo_list:
    
    for flag_brain in flag_brain_list: 
        
        for flag_condition in flag_condition_list:
            
            for flag_img in flag_img_list:
                
                if flag_hbo:
                    title_str = flag_condition + ' ' + 'HbO'
                    hbx_brain_scalp = 'hbo'
                else:
                    title_str = flag_condition + ' ' + 'HbR'
                    hbx_brain_scalp = 'hbr'
                
                if flag_brain:
                    title_str = title_str + ' brain'
                    hbx_brain_scalp = hbx_brain_scalp + '_brain'
                else:
                    title_str = title_str + ' scalp'
                    hbx_brain_scalp = hbx_brain_scalp + '_scalp'
                
                if flag_img == 'tstat':
                    foo_img = all_trial_X_tstat.sel(trial_type=flag_condition).copy()
                    title_str = title_str + ' t-stat'
                elif flag_img == 'mag':
                    foo_img = all_trial_X_hrf_mag_weighted.sel(trial_type=flag_condition).copy()
                    title_str = title_str + ' magnitude'
                elif flag_img == 'noise':
                    foo_img = all_trial_X_stderr.sel(trial_type=flag_condition).copy()
                    title_str = title_str + ' noise'
        
                foo_img = foo_img.pint.dequantify()
                foo_img = foo_img.transpose('vertex', 'chromo')
                foo_img[~M] = np.nan
                
             # 
                clim = (-foo_img.sel(chromo='HbO').max(), foo_img.sel(chromo='HbO').max())
                # if flag_img == 'mag':
                #     clim = [-7.6e-4, 7.6e-4]
                p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (1,1), clim, hbx_brain_scalp, 'scale_bar',
                                          None, title_str, off_screen=SAVE )
                p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (0,0), clim, hbx_brain_scalp, 'left', p0)
                p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (0,1), clim, hbx_brain_scalp, 'superior', p0)
                p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (0,2), clim, hbx_brain_scalp, 'right', p0)
                p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (1,0), clim, hbx_brain_scalp, 'anterior', p0)
                p0 = img_recon.plot_image_recon(foo_img, head, (2,3), (1,2), clim, hbx_brain_scalp, 'posterior', p0)
                
                if SAVE:
                    img_folder = f'{direct_name}_aspatial-{cfg_img_recon["alpha_spatial"]}_ameas-{cfg_img_recon["alpha_meas"]}_{Cmeas_name}_{SB_name}'
                    save_dir_tmp= os.path.join(cfg_dataset["root_dir"], 'derivatives', 'plots', 'image_recon', img_folder)
                    if not os.path.exists(save_dir_tmp):
                        os.makedirs(save_dir_tmp)
                    file_name = f'IMG_{flag_condition}_{flag_img}_{hbx_brain_scalp}.png'
                    p0.screenshot( os.path.join(save_dir_tmp, file_name) )
                    p0.close()
                else:
                    p0.show()
                    
                    
                    
                    
                    
                    
                    




#%%

# #################################################################################
# DB:
# ###############################################################################
# %% 
#
# Get image for each subject and do weighted average
#
##############################################################################
import importlib
importlib.reload(img_recon)


# add if chromo in blockaverage_subj.dims -> convert to OD --- 
    # !!! ^^ I think if in conc it will give error bc of blockaverage_subj_mse - check
# !!! ADD flag for if doing image recon on ts or hrf mag  ??

X_hrf_mag_subj = None
C = None # spatial regularization 
D = None


all_trial_X_hrf_mag = None

for idx_trial, trial_type in enumerate(blockaverage_subj.trial_type):
    
    print(f'Getting images for trial type = {trial_type.values}')
    all_subj_X_hrf_mag = None
    
    for idx_subj, curr_subj in enumerate(subj_ids_new):

        print(f'Starting image recon on subject {curr_subj}')
        
        if 'chromo' in blockaverage_subj.dims:
            # get the group average HRF over a time window
            hrf_conc_mag = blockaverage_subj.sel(subj= curr_subj).sel(trial_type=trial_type).sel(reltime=slice(cfg_img_recon['t_win'][0],cfg_img_recon['t_win'][1])).mean('reltime')
            hrf_conc_ts = blockaverage_subj.sel(subj= curr_subj).sel(trial_type=trial_type)
            
            blockaverage_mse_subj_conc = blockaverage_mse_subj.sel(subj= curr_subj).sel(trial_type=trial_type)
            
            # convert back to OD
            E = cedalion.nirs.get_extinction_coefficients(cfg_img_recon['spectrum'], wavelength)
            hrf_od_mag = xr.dot(E, hrf_conc_mag * 1*units.mm * 1e-6*units.molar / units.micromolar, dim=["chromo"]) # !!! assumes DPF = 1
            hrf_od_ts = xr.dot(E, hrf_conc_ts * 1*units.mm * 1e-6*units.molar / units.micromolar, dim=["chromo"]) # assumes DPF = 1
                
            blockaverage_mse_subj= xr.dot(E, blockaverage_mse_subj_conc * 1*units.mm * 1e-6*units.molar / units.micromolar, dim=["chromo"]) # assumes DPF = 1

        else:
            hrf_od_mag = blockaverage_subj.sel(subj= curr_subj).sel(trial_type=trial_type).sel(reltime=slice(cfg_img_recon['t_win'][0], cfg_img_recon['t_win'][1])).mean('reltime')
            hrf_od_ts = blockaverage_subj.sel(subj= curr_subj).sel(trial_type=trial_type)

        #
        #hrf_od_mag = blockaverage_subj.sel(subj=cfg_dataset['subj_ids'][idx_subj]).sel(trial_type=trial_type).sel(reltime=slice(cfg_img_recon['t_win'][0], cfg_img_recon['t_win'][1])).mean('reltime') 
        # hrf_od_ts = blockaverage_all.sel(trial_type=trial_type)
    
        # get the image
        
        C_meas = blockaverage_mse_subj.sel(subj=subj_ids_new[idx_subj]).sel(trial_type=trial_type).sel(reltime=slice(cfg_img_recon['t_win'][0], cfg_img_recon['t_win'][1])).mean('reltime') 
    
        C_meas = C_meas.pint.dequantify()
        C_meas = C_meas.stack(measurement=('channel', 'wavelength')).sortby('wavelength')
        
        if cfg_img_recon['flag_Cmeas']:
            cov_str = 'cov'
            if C is None or D is None:
                #X_hrf_mag_tmp, W, C, D = img_recon.do_image_recon( hrf_od_mag, head, Adot, C_meas, wavelength, BRAIN_ONLY, SB, sb_cfg, alpha_spatial_list, alpha_meas_list, file_save, file_path0, trial_type) 
                X_hrf_mag_tmp, W, C, D = img_recon.do_image_recon_DB( hrf_od = hrf_od_mag, head = head, Adot = Adot, C_meas = C_meas,
                                                                  wavelength = wavelength, cfg_img_recon = cfg_img_recon, 
                                                                  trial_type_img = trial_type, save_path = save_path,
                                                                  W = None, C = None, D = None) 
        
            else:
                X_hrf_mag_tmp, W, _, _ = img_recon.do_image_recon_DB( hrf_od = hrf_od_mag, head = head, Adot = Adot, C_meas = C_meas, 
                                                                  wavelength = wavelength, cfg_img_recon = cfg_img_recon, 
                                                                  trial_type_img = trial_type, save_path = save_path, 
                                                                  W = None, C = C, D = D)
        else:
            cov_str = ''
            if C is None or D is None:
                X_hrf_mag_tmp, W, C, D = img_recon.do_image_recon_DB( hrf_od = hrf_od_mag, head = head, Adot = Adot, C_meas = None,
                                                                  wavelength = wavelength, cfg_img_recon = cfg_img_recon, 
                                                                  trial_type_img = trial_type, save_path = save_path,
                                                                  W = None, C = None, D = None) 
        
            else:
                X_hrf_mag_tmp, W, _, _ = img_recon.do_image_recon_DB( hrf_od = hrf_od_mag, head = head, Adot = Adot, C_meas = None, 
                                                                  wavelength = wavelength, cfg_img_recon = cfg_img_recon, 
                                                                  trial_type_img = trial_type, save_path = save_path, 
                                                                  W = None, C = C, D = D)
        

        # get image noise
        cov_img_tmp = W * np.sqrt(C_meas.values) # get diag of image covariance
        cov_img_diag = np.nansum(cov_img_tmp**2, axis=1)
    
        nV = X_hrf_mag_tmp.vertex.size
        cov_img_diag = np.reshape( cov_img_diag, (2,nV) ).T
    
        X_mse = X_hrf_mag_tmp.copy() 
        X_mse.values = cov_img_diag # !!! SAVE nult trial types
        
        
        # weighted average -- same as chan space - but now is vertex space
        if all_subj_X_hrf_mag is None:
            all_subj_X_hrf_mag = X_hrf_mag_tmp
            all_subj_X_hrf_mag = all_subj_X_hrf_mag.expand_dims('subj')
            all_subj_X_hrf_mag = all_subj_X_hrf_mag.assign_coords(subj=subj_ids_new[idx_subj])
    
            X_mse_subj = X_mse.copy()
            X_mse_subj = X_mse_subj.expand_dims('subj')
            X_mse_subj = X_mse_subj.assign_coords(subj=subj_ids_new[idx_subj])
            
            X_hrf_mag_weighted = X_hrf_mag_tmp / X_mse
            X_mse_inv_weighted = 1 / X_mse
            X_mse_inv_weighted_max = 1 / X_mse
        else:
            X_hrf_mag_subj_tmp = X_hrf_mag_tmp.expand_dims('subj') # !!! will need to expand dims to get back trial type -- can do in function 
            X_hrf_mag_subj_tmp = X_hrf_mag_subj_tmp.assign_coords(subj=subj_ids_new[idx_subj])
    
            X_mse_subj_tmp = X_mse.copy().expand_dims('subj')
            X_mse_subj_tmp = X_mse_subj_tmp.assign_coords(subj=[subj_ids_new[idx_subj]])
    
            all_subj_X_hrf_mag = xr.concat([all_subj_X_hrf_mag, X_hrf_mag_subj_tmp], dim='subj')
            X_mse_subj = xr.concat([X_mse_subj, X_mse_subj_tmp], dim='subj')
    
            X_hrf_mag_weighted = X_hrf_mag_weighted + X_hrf_mag_tmp / X_mse
            X_mse_inv_weighted = X_mse_inv_weighted + 1 / X_mse
            X_mse_inv_weighted_max = np.maximum(X_mse_inv_weighted_max, 1 / X_mse)
        
    
    # END OF SUBJECT LOOP
    
    # get the average
    X_hrf_mag_mean = all_subj_X_hrf_mag.mean('subj')
    X_hrf_mag_mean_weighted = X_hrf_mag_weighted / X_mse_inv_weighted
    
    X_mse_mean_within_subject = 1 / X_mse_inv_weighted
    
    X_mse_subj_tmp = X_mse_subj.copy()
    X_mse_subj_tmp = xr.where(X_mse_subj_tmp < 1e-6, 1e-6, X_mse_subj_tmp)
    X_mse_weighted_between_subjects_tmp = (all_subj_X_hrf_mag - X_hrf_mag_mean)**2 / X_mse_subj_tmp # X_mse_subj_tmp is weights for each sub
    X_mse_weighted_between_subjects = X_mse_weighted_between_subjects_tmp.mean('subj')
    X_mse_weighted_between_subjects = X_mse_weighted_between_subjects / (X_mse_subj**-1).mean('subj')
    
    X_stderr_weighted = np.sqrt( X_mse_mean_within_subject + X_mse_weighted_between_subjects )
    
    X_tstat = X_hrf_mag_mean_weighted / X_stderr_weighted
    
    X_weight_sum = X_mse_inv_weighted / X_mse_inv_weighted_max  # tstat = weighted group avg / noise # !!! not saving?
    
    # Assign trial type coord
    X_hrf_mag_mean = X_hrf_mag_mean.assign_coords(trial_type = trial_type)
    X_hrf_mag_mean_weighted = X_hrf_mag_mean_weighted.assign_coords(trial_type = trial_type)
    X_stderr_weighted = X_stderr_weighted.assign_coords(trial_type = trial_type)
    X_tstat = X_tstat.assign_coords(trial_type = trial_type)

    if all_trial_X_hrf_mag is None:
        
        all_trial_X_hrf_mag = X_hrf_mag_mean
        all_trial_X_hrf_mag_weighted = X_hrf_mag_mean_weighted
        all_trial_X_stderr = X_stderr_weighted # noise
        all_trial_X_tstat = X_tstat # tstat
    else:
    
        all_trial_X_hrf_mag = xr.concat([all_trial_X_hrf_mag, X_hrf_mag_mean], dim='trial_type')
        all_trial_X_hrf_mag_weighted = xr.concat([all_trial_X_hrf_mag_weighted, X_hrf_mag_mean_weighted], dim='trial_type')
        all_trial_X_stderr = xr.concat([all_trial_X_stderr, X_stderr_weighted], dim='trial_type')
        all_trial_X_tstat = xr.concat([all_trial_X_tstat, X_tstat], dim='trial_type')

# END OF TRIAL TYPE LOOP

# FIXME: I am trying to get something like number of subjects per vertex...
# maybe I need to change X_mse_inv_weighted_max to be some typical value 
# because when all subjects have a really low value, then it won't scale the way I want

results_img_s = {'X_hrf_mag_all_trial': all_trial_X_hrf_mag,
           'X_hrf_mag_weighted_all_trial': all_trial_X_hrf_mag_weighted,
           'X_std_err_all_trial': all_trial_X_stderr,  # noise
           'X_tstat_all_trial': all_trial_X_tstat
           }

tasknm = cfg_dataset["file_ids"][0].split('_')[0]

# !!! chang name when indirect is implemented
if not cfg_img_recon['SB']:
    filepath = os.path.join(save_path, f'Xs_{tasknm}_direct_alltrial_{cov_str}_alpha_spatial_{cfg_img_recon["alpha_spatial_list"][-1]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas_list"][-1]:.0e}.pkl.gz')
    print(f'   Saving to Xs_{tasknm}_direct_alltrials_{cov_str}_alpha_spatial_{cfg_img_recon["alpha_spatial_list"][-1]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas_list"][-1]:.0e}.pkl.gz')
    file = gzip.GzipFile(filepath, 'wb')
    file.write(pickle.dumps(results_img_s))
    file.close()     
else:
    filepath = os.path.join(save_path, f'Xs_{tasknm}_direct_alltrial_{cov_str}_alpha_spatial_{cfg_img_recon["alpha_spatial_list"][-1]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas_list"][-1]:.0e}_SB_sigma_brain_{cfg_img_recon["sigma_brain"]}_sigma_scalp_{cfg_img_recon["sigma_scalp"]}.pkl.gz')
    print(f'   Saving to Xs_{tasknm}_direct_alltrials_{cov_str}_alpha_spatial_{cfg_img_recon["alpha_spatial_list"][-1]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas_list"][-1]:.0e}.pkl.gz')
    file = gzip.GzipFile(filepath, 'wb')
    file.write(pickle.dumps(results_img_s))
    file.close()     


# if DIRECT:
#     direct_name = 'direct'
# else:
#     direct_name = 'indirect'
# filepath = os.path.join(root_data_dir, f'Xs_{trial_type}_cov_alpha_spatial_{alpha_spatial:.0e}_alpha_meas_{alpha_meas:.0e}_{direct_name}_noCmeas.pkl.gz')
# print(f'   Saving to Xs_{trial_type}_cov_alpha_spatial_{alpha_spatial:.0e}_alpha_meas_{alpha_meas:.0e}_{direct_name}_noCmeas.pkl.gz')
# file = gzip.GzipFile(filepath, 'wb')
# file.write(pickle.dumps(results))
# file.close()   

#%%
# tasknm = cfg_dataset["file_ids"][0].split('_')[0]

# filepath = os.path.join(save_path, f'Xs_{tasknm}_direct_alltrial_{cov_str}_alpha_spatial_{cfg_img_recon["alpha_spatial_list"][-1]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas_list"][-1]:.0e}.pkl.gz')
# print(f'   Saving to Xs_{tasknm}_direct_alltrials_{cov_str}_alpha_spatial_{cfg_img_recon["alpha_spatial_list"][-1]:.0e}_alpha_meas_{cfg_img_recon["alpha_meas_list"][-1]:.0e}.pkl.gz')
# file = gzip.GzipFile(filepath, 'wb')
# file.write(pickle.dumps(results_img_s))
# file.close()   


#%% Load image recon results

# filname =  'Xs_STS_direct_alltrial_cov_alpha_spatial_1e-01_alpha_meas_1e+00.pkl.gz'
# filepath_bl = os.path.join(save_path , filname)

# if os.path.exists(filepath_bl):
#     with gzip.open(filepath_bl, 'rb') as f:
#         results_img_s = pickle.load(f)
#     all_trial_X_hrf_mag = results_img_s['X_hrf_mag_all_trial']
#     all_trial_X_hrf_mag_weighted = results_img_s['X_hrf_mag_weighted_all_trial']
#     all_trial_X_tstat = results_img_s['X_tstat_all_trial']
#     all_trial_X_stderr = results_img_s['X_std_err_all_trial']
    
#     print("Image results file loaded successfully!")


# %% Plot the images
##############################################################################

# !!! CHANGE FILE NAME IF GROUP INSTEAD OF XS


if all_trial_X_hrf_mag.trial_type.values.ndim > 0:
    
    threshold = -2 # log10 absolute
    wl_idx = 1
    M = sbf.get_sensitivity_mask(Adot, threshold, wl_idx)
    SAVE = True
    flag_hbo = True
    flag_brain = True
    flag_img_list = ['mag','tstat', 'noise']    # ['mag', 'tstat', 'noise'] #, 'noise'
    #flag_condition_list =['ST_o_tddr', 'ST_o_imu_tddr', 'DT_o_tddr', 'DT_o_imu_tddr'] #
    flag_condition_list = all_trial_X_hrf_mag.trial_type.values
    
    
    der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'plots', 'img_recon')
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)
    
    direct_name = 'Direct'  # !!! Change when implementing indirect method
    
    for flag_condition in flag_condition_list:
        
        for flag_img in flag_img_list:
            
            if flag_hbo:
                title_str = flag_condition + ' HbO'
                hbx_brain_scalp = 'hbo'
            else:
                title_str = flag_condition + ' HbR'
                hbx_brain_scalp = 'hbr'
            
            if flag_brain:
                title_str = title_str + ' brain'
                hbx_brain_scalp = hbx_brain_scalp + '_brain'
            else:
                title_str = title_str + ' scalp'
                hbx_brain_scalp = hbx_brain_scalp + '_scalp'
            
            if flag_img == 'tstat':
                foo_img = all_trial_X_tstat.sel(trial_type=flag_condition).copy()
                title_str = title_str + ' t-stat'
            elif flag_img == 'mag':
                foo_img = all_trial_X_hrf_mag_weighted.sel(trial_type=flag_condition).copy()  # plotting weighted
                title_str = title_str + ' magnitude'
            elif flag_img == 'noise':
                foo_img = all_trial_X_stderr.sel(trial_type=flag_condition).copy()
                title_str = title_str + ' noise'
    
            foo_img = foo_img.pint.dequantify()
            foo_img = foo_img.transpose('vertex', 'chromo') # why r we transposing these?
            foo_img[~M] = np.nan
            
            clim = (-foo_img.sel(chromo='HbO').max(), foo_img.sel(chromo='HbO').max())
            # if flag_img == 'tstat':
            #     clim = [-5, 5]
            p0 = img_recon.plot_img_recon(foo_img, head, (2,3), (1,1), clim, hbx_brain_scalp, 'scale_bar',
                                      None, title_str, off_screen=SAVE )
            p0 = img_recon.plot_img_recon(foo_img, head, (2,3), (0,0), clim, hbx_brain_scalp, 'left', p0)
            p0 = img_recon.plot_img_recon(foo_img, head, (2,3), (0,1), clim, hbx_brain_scalp, 'superior', p0)
            p0 = img_recon.plot_img_recon(foo_img, head, (2,3), (0,2), clim, hbx_brain_scalp, 'right', p0)
            p0 = img_recon.plot_img_recon(foo_img, head, (2,3), (1,0), clim, hbx_brain_scalp, 'anterior', p0)
            p0 = img_recon.plot_img_recon(foo_img, head, (2,3), (1,2), clim, hbx_brain_scalp, 'posterior', p0)
            
            if SAVE:
                if not cfg_img_recon['SB']:
                    filname = f'IMG_{flag_condition}_{direct_name}_{cov_str}_{flag_img}_{hbx_brain_scalp}.png'
                else:
                    filname = f'IMG_{flag_condition}_{direct_name}_{cov_str}_{flag_img}_{hbx_brain_scalp}_SB.png'
                p0.screenshot( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'plots', 'img_recon', filname) )
                p0.close()
            else:
                p0.show()
                
else:        
    #%%        
    # IF ONLY 1 TRIAL TYPE - plot images this way
    
    all_trial_X_hrf_mag_weighted_new = all_trial_X_hrf_mag_weighted.expand_dims("trial_type")
    all_trial_X_tstat_new = all_trial_X_tstat.expand_dims("trial_type")
    all_trial_X_stderr_new = all_trial_X_stderr.expand_dims("trial_type")
    
    
    threshold = -2 # log10 absolute
    wl_idx = 1
    M = sbf.get_sensitivity_mask(Adot, threshold, wl_idx)
    SAVE = True
    flag_hbo = True
    flag_brain = True
    flag_img_list = ['mag','tstat', 'noise']    # ['mag', 'tstat', 'noise'] #, 'noise'
    flag_condition_list = [[all_trial_X_hrf_mag.trial_type.values.item()]]
    #flag_condition_list = [['STS_o_tddr']]
    
    
    der_dir = os.path.join(cfg_dataset['root_dir'], 'derivatives', 'plots', 'img_recon')
    if not os.path.exists(der_dir):
        os.makedirs(der_dir)
    
    direct_name = 'Direct'  # !!! Change when implementing indirect method
    
    for flag_condition in flag_condition_list:
        
        for flag_img in flag_img_list:
            
            if flag_hbo:
                title_str = flag_condition[0] + ' HbO'
                hbx_brain_scalp = 'hbo'
            else:
                title_str = flag_condition[0] + ' HbR'
                hbx_brain_scalp = 'hbr'
            
            if flag_brain:
                title_str = title_str + ' brain'
                hbx_brain_scalp = hbx_brain_scalp + '_brain'
            else:
                title_str = title_str + ' scalp'
                hbx_brain_scalp = hbx_brain_scalp + '_scalp'
            
            if flag_img == 'tstat':
                #foo_img = all_trial_X_tstat.copy()
                foo_img = all_trial_X_tstat_new.sel(trial_type=flag_condition).sel(trial_type=flag_condition[0]).copy()
                title_str = title_str + ' t-stat'
            elif flag_img == 'mag':
                foo_img = all_trial_X_hrf_mag_weighted_new.sel(trial_type=flag_condition).sel(trial_type=flag_condition[0]).copy()  # plotting weighted
                title_str = title_str + ' magnitude'
            elif flag_img == 'noise':
                #foo_img = all_trial_X_stderr.copy()
                foo_img = all_trial_X_stderr_new.sel(trial_type=flag_condition).sel(trial_type=flag_condition[0]).copy()
                title_str = title_str + ' noise'
    
            foo_img = foo_img.pint.dequantify()
            foo_img = foo_img.transpose('vertex', 'chromo')    # why r we transposing these?
            foo_img[~M] = np.nan
            
            clim = (-foo_img.sel(chromo='HbO').max(), foo_img.sel(chromo='HbO').max())
            # if flag_img == 'tstat':
            #     clim = [-5, 5]
            p0 = img_recon.plot_img_recon(foo_img, head, (2,3), (1,1), clim, hbx_brain_scalp, 'scale_bar',
                                      None, title_str, off_screen=SAVE )
            p0 = img_recon.plot_img_recon(foo_img, head, (2,3), (0,0), clim, hbx_brain_scalp, 'left', p0)
            p0 = img_recon.plot_img_recon(foo_img, head, (2,3), (0,1), clim, hbx_brain_scalp, 'superior', p0)
            p0 = img_recon.plot_img_recon(foo_img, head, (2,3), (0,2), clim, hbx_brain_scalp, 'right', p0)
            p0 = img_recon.plot_img_recon(foo_img, head, (2,3), (1,0), clim, hbx_brain_scalp, 'anterior', p0)
            p0 = img_recon.plot_img_recon(foo_img, head, (2,3), (1,2), clim, hbx_brain_scalp, 'posterior', p0)
            
            if SAVE:
                if not cfg_img_recon['SB']:
                    filname = f'IMG_{flag_condition[0]}_{direct_name}_{cov_str}_{flag_img}_{hbx_brain_scalp}.png'
                else:
                    filname = f'IMG_{flag_condition[0]}_{direct_name}_{cov_str}_{flag_img}_{hbx_brain_scalp}_SB.png'
                p0.screenshot( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'plots', 'img_recon', filname) )
                p0.close()
            else:
                p0.show()
            
            
#%%


# # import importlib
# # importlib.reload(img_recon)


# flag_hbo = True
# flag_brain = True
# flag_recon = 'group'  # if image recon done on group avg or subjects
# flag_img = 'tstat' # 'tstat', 'mag', 'noise'
# flag_condition = trial_type #'DT_o_tddr' # 'ST', 'DT', 'STS'


# if flag_hbo:
#     title_str = flag_condition + '_HbO'
#     hbx_brain_scalp = 'hbo'
# else:
#     title_str = flag_condition + '_HbR'
#     hbx_brain_scalp = 'hbr'

# if flag_brain:
#     title_str = title_str + '_brain'
#     hbx_brain_scalp = hbx_brain_scalp + '_brain'
# else:
#     title_str = title_str + '_scalp'
#     hbx_brain_scalp = hbx_brain_scalp + '_scalp'


# if flag_recon == 'group':
#     if flag_img == 'tstat':
#         foo_img = X_tstat.copy()
#         title_str = title_str + '_' + flag_recon + '_t-stat'
        
#     elif flag_img == 'mag':
#         foo_img = X_grp.copy()
#         title_str = title_str + '_' + flag_recon + '_magnitude'
        
#     elif flag_img == 'noise':
#         foo_img = X_noise.copy()
#         title_str = title_str + '_' + flag_recon + '_noise'

# else:  # image recon done on indiv subjs and weighted block avg done in image space
#     foo_img[~M] = np.nan # !!! somethign to do with something that's not group 
    
#     if flag_img == 'tstat':
#         foo_img = X_tstat.copy()
#         title_str = title_str + '_' + flag_recon + '_t-stat' 
#     elif flag_img == 'mag':
#         foo_img = X_hrf_mag_mean_weighted.copy()
#         title_str = title_str + '_' + flag_recon + '_magnitude'
#     elif flag_img == 'noise':
#         foo_img = X_stderr_weighted.copy()
#         title_str = title_str + '_' + flag_recon + '_noise'

# #    foo_img.values = np.log10(foo_img.values)+7


# # title_str = 'HbR'
# # hbx_brain_scalp = 'hbr_brain'
# # foo_img = X_hrf_mag_mean_weighted

# # title_str = 'HbR t-stat'
# # hbx_brain_scalp = 'hbr_brain'
# # foo_img = X_tstat


# # foo_img = xr.where(np.abs(foo_img) < 1.86, np.nan, foo_img) # one-tail is 1.86 and two tail is 2.3


# p0 = img_recon.plot_img_recon(foo_img, head, (2,3), (1,1), hbx_brain_scalp, 'scale_bar', None, title_str)
# p0 = img_recon.plot_img_recon(foo_img, head, (2,3), (0,0), hbx_brain_scalp, 'left', p0)
# p0 = img_recon.plot_img_recon(foo_img, head, (2,3), (0,1), hbx_brain_scalp, 'superior', p0)
# p0 = img_recon.plot_img_recon(foo_img, head, (2,3), (0,2), hbx_brain_scalp, 'right', p0)
# p0 = img_recon.plot_img_recon(foo_img, head, (2,3), (1,0), hbx_brain_scalp, 'anterior', p0)
# p0 = img_recon.plot_img_recon(foo_img, head, (2,3), (1,2), hbx_brain_scalp, 'posterior', p0)


# p0.screenshot( os.path.join(cfg_dataset['root_dir'], 'derivatives', 'plots', 'img_recon', f'{title_str}_IMG.png') )
# p0.close()






# %%

'''
X_foo = X_tstat.copy()
X_foo[:,0] = 0

# select parcels
# parcels with '_LH' at the end
parcels = np.unique(X_grp['parcel'].values)
parcels_LH = [x for x in parcels if x.endswith('_LH')]

parcels_sel = [x for x in parcels_LH if 'DefaultB_PFCv' in x]

X_foo[np.isin(X_foo['parcel'].values, parcels_sel), 0] = 1


p0 = img_recon.plot_img_recon(X_foo, head, 'hbo_brain', 'left')



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

p0 = img_recon.plot_img_recon(X_foo, head, 'hbo_brain', 'left')

'''

