#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Get DQR for individual subject

Created on Fri Mar 21 10:05:09 2025

@author: smkelley
"""

import os
import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality
import cedalion.xrutils as xrutils


import xarray as xr
import matplotlib.pyplot as p
import cedalion.plots as plots
from cedalion import units
import numpy as np
import pandas as pd
from math import ceil

import sys
sys.path.append('/projectnb/nphfnirs/ns/Shannon/Code/cedalion-dab-funcs2/modules')
import module_plot_DQR as dqr
import module_load_and_preprocess as preproc

#%%

root_dir = "/projectnb/nphfnirs/ns/Shannon/Data/Interactive_Walking_HD/" # CHANGE to your data directory

subj = '01' # CHANGE
task = 'IWHD' # CHANGE
runNum = '01' # CHANGE
stim_lst = ['ST', 'DT']  # CHANGE   - your stim names

cfg_prune = {
    'snr_thresh' : 5, # CHANGE
    'sd_threshs' : [1, 60]*units.mm, # CHANGE  # defines the lower and upper bounds for the source-detector separation that we would like to keep
    'amp_threshs' : [1e-5, 0.84], # CHANGE   # define whether a channel's amplitude is within a certain range
    'perc_time_clean_thresh' : 0.6, 
    'sci_threshold' : 0.6,
    'psp_threshold' : 0.1,
    'window_length' : 5 * units.s, 
    'flag_use_sci' : True,   # CHANGE
    'flag_use_psp' : False   # CHANGE
}


# cfg_motion_correct = {
#     'flag_do_tddr' : False,  # CHANGE 
#     'flag_do_imu_glm' : False, # CHANGE -- only True if paradigm includes walking
# }

cfg_bandpass = { 
    'fmin' : 0.01 * units.Hz, # CHANGE
    'fmax' : 0.5 * units.Hz  # CHANGE
}
        

cfg_preprocess = {
    'flag_prune_channels' : True, 
    'median_filt' : 3, # CHANGE  # set to 1 if you don't want to do median filtering
    'cfg_prune' : cfg_prune, 
    'cfg_bandpass' : cfg_bandpass,
    'flag_do_GLM_filter' : True, # CHANGE
}

#%% Load in data

subDir = os.path.join(root_dir, f'sub-{subj}', 'nirs')
run_nm = f'sub-{subj}_task-{task}_run-{runNum}'

snirf_path = os.path.join(subDir, run_nm + '_nirs.snirf' )

# check if the snirf file exists
if not os.path.exists( snirf_path ):
    print( f"Error: File {snirf_path} does not exist" )
else:
    records = cedalion.io.read_snirf( snirf_path ) 
    rec = records[0]

events_path = os.path.join(subDir, run_nm + '_events.tsv' )

# check if the events.tsv file exists
if not os.path.exists( events_path ):
    print( f"Error: File {events_path} does not exist" )
else:
    stim_df = pd.read_csv(events_path, sep='\t' )
    rec.stim = stim_df

#%% Preprocess 

# Preprocess data with median filt
rec = preproc.preprocess( rec, cfg_preprocess['median_filt'] )

# Prune channels
rec, chs_pruned, sci, psp = preproc.pruneChannels( rec, cfg_preprocess['cfg_prune'] )
pruned_chans = chs_pruned.where(chs_pruned != 0.4, drop=True).channel.values # get array of channels that were pruned


# Calculate OD 
# if flag pruned channels is True, then do rest of preprocessing on pruned amp, if not then do preprocessing on unpruned data
if cfg_preprocess['flag_prune_channels']:
    rec["od"] = cedalion.nirs.int2od(rec['amp_pruned'])                
else:
    rec["od"] = cedalion.nirs.int2od(rec['amp'])
    del rec.timeseries['amp_pruned']   # delete pruned amp from time series
rec["od_corrected"] = rec["od"]    # need to reassign to new rec_str to work w/ code

# Calculate GVTD on pruned data
amp_masked = preproc.prune_mask_ts(rec['amp'], pruned_chans)  # use chs_pruned to get gvtd w/out pruned data (could also zscore in gvtd func)
rec.aux_ts["gvtd"], _ = quality.gvtd(amp_masked) 


lambda0 = amp_masked.wavelength[0].wavelength.values
lambda1 = amp_masked.wavelength[1].wavelength.values
snr0, _ = quality.snr(amp_masked.sel(wavelength=lambda0), cfg_preprocess['cfg_prune']['snr_thresh'])
snr1, _ = quality.snr(amp_masked.sel(wavelength=lambda1), cfg_preprocess['cfg_prune']['snr_thresh'])


dqr.plotDQR( rec, chs_pruned, cfg_preprocess, run_nm, root_dir, stim_lst )





    