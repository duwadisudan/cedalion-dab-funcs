# %% Imports
##############################################################################
%matplotlib widget

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
import DABfuncs_load_and_preprocess as pfDAB
import DABfuncs_plot_DQR as pfDAB_dqr
import DABfuncs_group_avg as pfDAB_grp_avg
import DABfuncs_ERBM_ICA as pfDAB_ERBM
import DABfuncs_image_recon as pfDAB_img


# Turn off all warnings
import warnings
warnings.filterwarnings('ignore')


# %% 
##############################################################################
import importlib
importlib.reload(pfDAB_img)


# %% Initial root directory and analysis parameters
##############################################################################

# root directory for the BIDS dataset
rootDir_data = '/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data/Datasets/Esplanade/'

# subject and file IDs to process
subj_ids = ['01','02','03','04','05','06','07','08','09']
file_ids = ['STS_run-01']
#file_ids = ['IWHD_run-01']

# list of stim trial_types to include...
stim_lst_dqr = ['ST','DT'] # for DQR plots
stim_lst_hrf = ['ST','DT'] # for calculating HRFs

# pruning parameters
snr_thresh = 5 # the SNR (std/mean) of a channel. 
sd_threshs = [1, 60]*units.mm # defines the lower and upper bounds for the source-detector separation that we would like to keep
amp_threshs = [1e-5, 0.84] # define whether a channel's amplitude is within a certain range
                           # reduced from 0.89 to 0.84 because saturation goes lower

# motion correction parameters
flag_do_splineSG = False # if True, will do splineSG motion correction

# band pass filter parameters applied to OD after motion correction
fmin = 0.02 * units.Hz
fmax = 3 * units.Hz





# determine the number of subjects and files. Often used in loops.
n_subjects = len(subj_ids)
n_files_per_subject = len(file_ids)


# %% Load and preprocess the data
##############################################################################

# Load and preprocess the data
#
# This function will load all the data for the specified subject and file IDs, and preprocess the data.
# This function will also create several data quality report (DQR) figures that are saved in /derivatives/plots.
# The function will return the preprocessed data and a list of the filenames that were loaded, both as 
# two dimensional lists [subj_idx][file_idx].
# The data is returned as a recording container with the following fields:
#   timeseries - the data matrices with dimensions of ('channel', 'wavelength', 'time') 
#      or ('channel', 'HbO/HbR', 'time') depending on the data type. 
#      The following sub-fields are included:
#         'amp' - the original amplitude data slightly processed to remove negative and NaN values and to 
#            apply a 3 point median filter to remove outliers.
#         'amp_pruned' - the 'amp' data pruned according to the SNR, SD, and amplitude thresholds.
#         'od' - the optical density data
#         'od_tddr' - the optical density data after TDDR motion correction is applied
#         'conc_tddr' - the concentration data obtained from 'od_tddr'
#         'od_splineSG' and 'conc_splineSG' - returned if splineSG motion correction is applied (i.e. flag_do_splineSG=True)
#   stim - the stimulus data with 'onset', 'duration', and 'trial_type' fields and more from the events.tsv files.
#   aux_ts - the auxiliary time series data from the SNIRF files.
#      In addition, the following aux sub-fields are added during pre-processing:
#         'gvtd' - the global variance of the time derivative of the 'od' data.
#         'gvtd_tddr' - the global variance of the time derivative of the 'od_tddr' data.

rec, filenm_lst, chs_pruned_subjs = pfDAB.load_and_preprocess( rootDir_data, subj_ids, file_ids, snr_thresh, sd_threshs, amp_threshs, stim_lst_dqr, flag_do_splineSG, fmin, fmax )





# %% Block Average - unweighted
##############################################################################

trange_hrf = [5, 35] * units.s # time range for block averaging
trange_hrf_stat = [10, 20] # time range for t-stat
stim_lst_hrf = ['ST'] # for calculating HRFs

ica_lpf = 1.0 * units.Hz # MUST be the same as used when creating W_ica

subj_id_exclude = [] #['05','07'] # if you want to exclude a subject from the group average

flag_save_each_subj = False # if True, will save the block average data for each subject




# loop over subjects and files
blockaverage_subj = None
for subj_idx in range( n_subjects ):
    for file_idx in range( n_files_per_subject ):

        filenm = filenm_lst[subj_idx][file_idx]
        print( f"Running {subj_idx+1} of {n_subjects} subjects : {filenm}" )

        if flag_do_splineSG:
            conc_filt = rec[subj_idx][file_idx]['conc_splineSG'].copy()
        else:
            conc_filt = rec[subj_idx][file_idx]['conc_tddr'].copy()
        # LPF the data to match the ICA data
        conc_filt = cedalion.sigproc.frequency.freq_filter(conc_filt, 0 * units.Hz, ica_lpf )

        # check if rec[subj_idx][file_idx]['conc_tddr_ica'] exists
        if 'conc_tddr_ica' in rec[subj_idx][file_idx].timeseries.keys():
            conc_filt_ica = rec[subj_idx][file_idx]['conc_tddr_ica'].copy()
        else:
            conc_filt_ica = None

        #
        # block average
        #

        # select the stim for the given file
        stim = rec[subj_idx][file_idx].stim.copy()

        # get the epochs
        # conc_epochs_tmp = pfDAB_grp_avg.block_average( conc_filt, stim, rec[subj_idx][file_idx].geo3d, trange_hrf, glm_basis_func_param, glm_drift_order, flag_do_GLM, ssr_rho_thresh, stim_lst_hrf )
        conc_epochs_tmp = pfDAB_grp_avg.block_average( conc_filt, stim, rec[subj_idx][file_idx].geo3d, trange_hrf, [], [], False, [], stim_lst_hrf )
        # concatenate the different epochs from each file for each subject
        if file_idx == 0:
            conc_epochs_all = conc_epochs_tmp
        else:
            conc_epochs_all = xr.concat([conc_epochs_all, conc_epochs_tmp], dim='epoch')

        # get the epochs for ICA filtered data
        if conc_filt_ica is not None:
            stim_ica = rec[subj_idx][file_idx].stim.copy()
            stim_ica['trial_type'] = stim_ica['trial_type'] + '-ica'
            stim_lst_hrf_ica = [x + '-ica' for x in stim_lst_hrf]

            # get the epochs for ICA filtered data
            # conc_epochs_ica_tmp = pfDAB_grp_avg.block_average( conc_filt_ica, stim_ica, rec[subj_idx][file_idx].geo3d, trange_hrf, glm_basis_func_param, glm_drift_order, flag_do_GLM, ssr_rho_thresh, stim_lst_hrf_ica )
            conc_epochs_ica_tmp = pfDAB_grp_avg.block_average( conc_filt_ica, stim_ica, rec[subj_idx][file_idx].geo3d, trange_hrf, [], [], False, [], stim_lst_hrf_ica )
            # interpolate the ICA epochs to the same time points as the non-ICA epochs
            conc_epochs_ica_tmp = conc_epochs_ica_tmp.interp(reltime=conc_epochs_all.reltime) * units.micromolar
            # concatenate the different epochs from each file for each subject
            conc_epochs_all = xr.concat([conc_epochs_all, conc_epochs_ica_tmp], dim='epoch')

        if flag_save_each_subj:
            conc_epochs_tmp = conc_epochs_tmp.assign_coords(trial_type=('epoch', [x + '-' + subj_ids[subj_idx] for x in conc_epochs_tmp.trial_type.values]))
            conc_epochs_ica_tmp = conc_epochs_ica_tmp.assign_coords(trial_type=('epoch', [x + '-' + subj_ids[subj_idx] for x in conc_epochs_ica_tmp.trial_type.values]))

            conc_epochs_all = xr.concat([conc_epochs_all, conc_epochs_tmp], dim='epoch')
            conc_epochs_all = xr.concat([conc_epochs_all, conc_epochs_ica_tmp], dim='epoch')

        # DONE LOOP OVER FILES

    # Block Average
    baseline_conc = conc_epochs_all.sel(reltime=(conc_epochs_all.reltime < 0)).mean('reltime')
    conc_epochs = conc_epochs_all - baseline_conc
    blockaverage = conc_epochs.groupby('trial_type').mean('epoch')

    # gather the blockaverage across subjects
    if blockaverage_subj is None and subj_ids[subj_idx] not in subj_id_exclude:
        blockaverage_subj = blockaverage
        # add a subject dimension and coordinate
        blockaverage_subj = blockaverage_subj.expand_dims('subj')
        blockaverage_subj = blockaverage_subj.assign_coords(subj=[subj_ids[subj_idx]])
    elif subj_ids[subj_idx] not in subj_id_exclude:
        blockaverage_subj_tmp = blockaverage
        blockaverage_subj_tmp = blockaverage_subj_tmp.expand_dims('subj')
        blockaverage_subj_tmp = blockaverage_subj_tmp.assign_coords(subj=[subj_ids[subj_idx]])
        blockaverage_subj = xr.concat([blockaverage_subj, blockaverage_subj_tmp], dim='subj')
    else:
        print(f"   Subject {subj_ids[subj_idx]} excluded from group average")

    # DONE LOOP OVER SUBJECTS


# Save the block average data to a pickle file
# this can then be viewed with vis_plot_probe_from_pickle.py
# The file is saved in derivates/processed_data/blockaverage.pkl.gz
# blockaverage_subj_mean = np.nanmean(blockaverage_subj, axis=4)
# blockaverage.values = blockaverage_subj_mean
blockaverage = blockaverage_subj.mean('subj')
if flag_save_each_subj:
    # FIXME: this assumes the number of subjects and trial_type. Generalize this in the future.
    # blockaverage = blockaverage.sel(trial_type=['ST', 'ST-ica', 'ST-01', 'ST-ica-01', 'ST-02', 'ST-ica-02', 'ST-03', 'ST-ica-03', 'ST-04', 'ST-ica-04', 'ST-05', 'ST-ica-05', 'ST-06', 'ST-ica-06', 'ST-07', 'ST-ica-07', 'ST-08', 'ST-ica-08', 'ST-09', 'ST-ica-09'])
    blockaverage = blockaverage.sel(trial_type=['ST', 'ST-ica', 'ST-01', 'ST-ica-01', 'ST-02', 'ST-ica-02', 'ST-03', 'ST-ica-03', 'ST-04', 'ST-ica-04', 'ST-06', 'ST-ica-06', 'ST-08', 'ST-ica-08', 'ST-09', 'ST-ica-09'])
file_path_pkl = os.path.join(rootDir_data, 'derivatives', 'processed_data', 'blockaverage.pkl.gz')
file = gzip.GzipFile(file_path_pkl, 'wb')
file.write(pickle.dumps([blockaverage, rec[subj_idx][file_idx].geo2d, rec[subj_idx][file_idx].geo3d]))
file.close()

blockaverage_all = blockaverage.copy()
blockaverage_all_o = blockaverage_all.copy()

print('Saved group average HRF to ' + file_path_pkl)




# %% Block Average - variance of the mean weighted (aka MSE weighted)
##############################################################################

cov_min_thresh = 1e-6 # minimum value for the diagonal of the covariance matrix

y_mean = []
y_mean_weighted = []
cov_mean_weighted = []

for subj_idx in range(n_subjects):

    y_subj_mean = []
    y_subj_mean_weighted = []
    cov_subj_mean_weighted = []
    for file_idx in range(n_files_per_subject):

        filenm = filenm_lst[subj_idx][file_idx]
        print( f"Running {subj_idx+1} of {n_subjects} subjects : {filenm}" )

        od_filt = rec[subj_idx][file_idx]['od_o_tddr']
        geo3d = rec[subj_idx][file_idx].geo3d
        stim = rec[subj_idx][file_idx].stim.copy()
        od_epochs = pfDAB_grp_avg.block_average_od( od_filt, stim, geo3d, trange_hrf, stim_lst_hrf )

        od_epochs = od_epochs.stack(measurement=['channel', 'wavelength']).sortby('wavelength')
        od_epochs = od_epochs.transpose('measurement', 'epoch', 'reltime')

        # baseline subtract
        od_epochs = od_epochs - od_epochs.sel(reltime=slice(-trange_hrf[0].magnitude, 0)).mean('reltime')

        n_epochs = od_epochs.shape[1]
        n_reltime = od_epochs.shape[2]
        od_epochs_mean = od_epochs.mean('epoch')

        # calculate the covariance matrix over epochs
        foo = (od_epochs - od_epochs_mean) 
        foo_t = foo.transpose('measurement', 'reltime', 'epoch')
        foo_t = foo_t.values
        cov_t_diag = np.einsum('ijk,ijk->ij', foo_t, foo_t) / (n_epochs - 1)**2 # this is squared to get variance of the mean,
                                                                                # aka the MSE of the mean.

        foo = np.reshape(foo.values, (foo.shape[0], -1))
        cov_diag = np.sum( foo * foo, axis=1 ) / ((n_epochs * n_reltime - 1)*(n_epochs-1)) # divide by n_epochs to get variance of the mean,
                                                                                           # aka the MSE of the mean.
#        cov = foo @ foo.transpose() / (n_epochs * n_reltime - 1)

        # adjust the covariance matrix to handle bad data
        cov_val_on_diagonal = 1e0 # set this to a large value... FIXME: larger than good data
        cov_val_off_diagonal = 0 # I choose zero for the off diagonal elements to essentially ignore these channels
                                 # I also have to use truncated SVD to invert the covariance matrix since I have many 
                                 # fewer epochs than channels. This will also nicely handle the all the off diagonal elements
                                 # for which I have low confidence given # epochs << # channels.

        # list of elements in cov_diag that are nan
        # I found these to be amp=1e-6, which was set by convertBinToSnirf for negative data
        # and TDDR makes this a NaN for some reason.
        # I also found that there are a few channels per run with a mean amp very close to 1e-6 but the time series
        # has some spikes. These should also be set to a large value for cov. I can determine this by amp<1.1e-6 or look
        # for diff(y) having many zeros indicating a constant y
        idx_nan = np.where(np.isnan(cov_diag))[0] 
        cov_diag[idx_nan] = cov_val_on_diagonal
        cov_t_diag[idx_nan,:] = cov_val_on_diagonal
        # cov[idx_nan,:] = cov_val_off_diagonal # only necessary if we want the full covariance matrix
        # cov[:,idx_nan] = cov_val_off_diagonal
        od_epochs_mean[idx_nan,:] = 0

        amp = rec[subj_idx][file_idx]['amp'].mean('time') # I think this also gets the NaNs from idx_nan
        amp = amp.stack(measurement=['channel', 'wavelength']).sortby('wavelength').transpose()
        idx_amp = np.where(amp < 1.1e-6)[0]
        cov_diag[idx_amp] = cov_val_on_diagonal
        cov_t_diag[idx_amp,:] = cov_val_on_diagonal
        # cov[idx_amp,:] = cov_val_off_diagonal # only necessary if we want the full covariance matrix
        # cov[:,idx_amp] = cov_val_off_diagonal
        od_epochs_mean[idx_amp,:] = 0

        # FIXME: can remove this once amp_thresh set to 0.84 at top
        # idx_amp = np.where(amp > 0.84)[0] # reduced from 0.89 to 0.84 because saturation goes lower
        # cov_diag[idx_amp] = cov_val_on_diagonal
        # cov_t_diag[idx_amp,:] = cov_val_on_diagonal
        # cov[idx_amp,:] = cov_val_off_diagonal
        # cov[:,idx_amp] = cov_val_off_diagonal
        # od_epochs_mean[idx_amp,:] = 0

        # look at saturated channels
        idx_sat = np.where(chs_pruned_subjs[subj_idx][file_idx] == 0.0)[0]
        n_chs = int(len(cov_diag)//2)
        cov_diag[idx_sat] = cov_val_on_diagonal
        cov_diag[idx_sat + n_chs] = cov_val_on_diagonal
        cov_t_diag[idx_sat,:] = cov_val_on_diagonal
        cov_t_diag[idx_sat + n_chs,:] = cov_val_on_diagonal
        # cov[idx_sat,:] = cov_val_off_diagonal # only necessary if we want the full covariance matrix
        # cov[:,idx_sat] = cov_val_off_diagonal
        # cov[idx_sat + n_chs,:] = cov_val_off_diagonal
        # cov[:,idx_sat + n_chs] = cov_val_off_diagonal
        od_epochs_mean[idx_sat,:] = 0

        # set the minimum value of the diagonal of the covariance matrix
        cov_diag_o = cov_diag.copy()
        cov_diag[cov_diag < cov_min_thresh] = cov_min_thresh # FIXME: plot the histogram of cov_diag to determine a good thresh
        cov_t_diag[cov_t_diag < cov_min_thresh] = cov_min_thresh

        # for now we only use the diagonal elements of the covariance matrix
        # since we don't have enough epochs to estimate the full covariance matrix accurately
        # FIXME: But with reltime * epochs, maybe I can look at off-diagonal elements
        if 1:
            cov = np.diag(cov_diag)
            tsvd_drop = 0
        else:
            # replace the diag of cov with cov_diag keeping the off diagonal terms the same
            cov = np.diag(cov_diag) + cov - np.diag(np.diag(cov))
            tsvd_drop = -1

        if tsvd_drop < 0:
            U, S, Vt = np.linalg.svd(cov)
            rank = np.where(np.diff(np.log10(S)) < tsvd_drop)[0][0] + 1 # where the singular value drops by a factor of 10
            cov_inv_approx = U[:, :rank] @ np.diag(1 / S[:rank]) @ U[:, :rank].T
            print(f'subj {subj_idx}, rank {rank}')
        else:
            cov_inv_approx = np.linalg.inv(cov)
            cov_t_inv_approx = 1 / cov_t_diag # element wise division as the columns are cov diagonal values

        if subj_idx == 0 and file_idx == 0:
            # y_subjs = []
            cov_subjs = []
            cov_t_subjs = []
            # y_subjs.append( [od_epochs_mean] )
            cov_subjs.append( [cov_diag_o] )
            cov_t_subjs.append( [cov_t_diag] )
            y_subj_mean = od_epochs_mean
            y_subj_mean_weighted = cov_inv_approx @ od_epochs_mean.values
            cov_inv_subj_mean_weighted = cov_inv_approx
            cov_t_inv_subj_mean_weighted = cov_t_inv_approx
        elif file_idx == 0:
            # y_subjs.append( [od_epochs_mean] )
            cov_subjs.append( [cov_diag_o] )
            cov_t_subjs.append( [cov_t_diag] )
            y_subj_mean = od_epochs_mean
            y_subj_mean_weighted = cov_inv_approx @ od_epochs_mean.values
            cov_inv_subj_mean_weighted = cov_inv_approx
            cov_t_inv_subj_mean_weighted = cov_t_inv_approx
        else:
            # y_subjs[subj_idx].append( od_epochs_mean )
            cov_subjs[subj_idx].append( cov_diag_o )
            cov_t_subjs[subj_idx].append( cov_t_diag )
            y_subj_mean = y_subj_mean + od_epochs_mean
            y_subj_mean_weighted = y_subj_mean_weighted + cov_inv_approx @ od_epochs_mean.values
            cov_inv_subj_mean_weighted = cov_inv_subj_mean_weighted + cov_inv_approx
            cov_t_inv_subj_mean_weighted = cov_t_inv_subj_mean_weighted + cov_t_inv_approx
        
    # end loop over files

    if tsvd_drop < 0:
        U, S, Vt = np.linalg.svd(cov_inv_subj_mean_weighted)
        rank = np.where(np.diff(np.log10(S)) < tsvd_drop)[0][0] + 1# where the singular value drops by a factor of 10
        cov_subj_mean_weighted = U[:, :rank] @ np.diag(1 / S[:rank]) @ U[:, :rank].T
        cov_inv_subj_mean_weighted = U[:, :rank] @ np.diag(S[:rank]) @ U[:, :rank].T
        print(f'subj {subj_idx}, rank {rank}')
    else:
        cov_subj_mean_weighted = np.linalg.inv(cov_inv_subj_mean_weighted)
        cov_t_subj_mean_weighted = 1 / cov_t_inv_subj_mean_weighted # element wise division as the columns are cov diagonal values

    y_subj_mean_weighted = cov_subj_mean_weighted @ y_subj_mean_weighted
    y_subj_mean = y_subj_mean / n_files_per_subject

    if subj_idx == 0:
        y_mean = y_subj_mean
        y_mean_weighted = cov_inv_subj_mean_weighted @ y_subj_mean_weighted
        cov_inv_mean_weighted = cov_inv_subj_mean_weighted
        cov_t_inv_mean_weighted = cov_t_inv_subj_mean_weighted
    else:
        y_mean = y_mean + y_subj_mean
        y_mean_weighted = y_mean_weighted + cov_inv_subj_mean_weighted @ y_subj_mean_weighted
        cov_inv_mean_weighted = cov_inv_mean_weighted + cov_inv_subj_mean_weighted
        cov_t_inv_mean_weighted = cov_t_inv_mean_weighted + cov_t_inv_subj_mean_weighted

# end loop over subjects

if tsvd_drop < 0:
    U, S, Vt = np.linalg.svd(cov_inv_mean_weighted)
    rank = np.where(np.diff(np.log10(S)) < tsvd_drop)[0][0] + 1# where the singular value drops by a factor of 10
    cov_mean_weighted = U[:, :rank] @ np.diag(1 / S[:rank]) @ U[:, :rank].T
    print(f'rank {rank}')
else:
    cov_mean_weighted = np.linalg.inv(cov_inv_mean_weighted)
    cov_t_mean_weighted = 1 / cov_t_inv_mean_weighted # element wise division as the columns are cov diagonal values

y_mean_weighted = cov_mean_weighted @ y_mean_weighted
y_mean = y_mean / n_subjects

# convert to concentration
y_mean_tmp = y_mean.copy()
y_mean_tmp.values = y_mean_weighted
foo_conc, foo_conc_tmp = pfDAB_grp_avg.y_mean_to_conc( y_mean_tmp, rec[0][0].geo3d, rec[0][0]['amp'].wavelength, od_filt.source, stim_lst_hrf, cov_mean_weighted, trange_hrf )
foo_conc = foo_conc.assign_coords(trial_type=['ST-weighted'])
foo_conc_tmp = foo_conc_tmp.assign_coords(trial_type=['ST-weighted'])

blockaverage_all_tmp = xr.concat([blockaverage_all_o, foo_conc_tmp], dim='trial_type')
blockaverage_all = xr.concat([blockaverage_all_o, foo_conc], dim='trial_type')

# y_mean_tmp.values = y_mean
# foo_conc_tmp = pfDAB_grp_avg.y_mean_to_conc( y_mean_tmp, rec[0][0].geo3d, rec[0][0]['amp'].wavelength, od_filt.source, stim_lst_hrf, cov_mean_weighted )
# foo_conc = xr.concat([foo_conc, foo_conc_tmp], dim='trial_type')


# Save the block average data to a pickle file
# this can then be viewed with vis_plot_probe_from_pickle.py
# The file is saved in derivates/processed_data/blockaverage.pkl.gz
file_path_pkl = os.path.join(rootDir_data, 'derivatives', 'processed_data', 'blockaverage.pkl.gz')
file = gzip.GzipFile(file_path_pkl, 'wb')
file.write(pickle.dumps([blockaverage_all_tmp, rec[0][0].geo2d, rec[0][0].geo3d]))
file.close()

print('Saved group average HRF to ' + file_path_pkl)


# %% Plot the Histogram of the cov.diagonal() values
##############################################################################

f,ax = p.subplots(2,1,figsize=(9,10))

# plot the diagonals for all subjects
ax1 = ax[0]
for i in range(n_subjects):
    for j in range(n_files_per_subject):
        ax1.semilogy(cov_subjs[i][j], linewidth=0.5,alpha=0.5)
ax1.semilogy(cov_mean_weighted.diagonal(), label='grp cov', color='k', linewidth=1)
ax1.set_title('variance in the mean for all subjects')
ax1.set_xlabel('channel')
ax1.legend()

# histogram the diagonals
ax1 = ax[1]
foo = np.concatenate([cov_subjs[i][j] for i in range(n_subjects) for j in range(n_files_per_subject)])
ax1.hist(np.log10(foo), bins=100)
ax1.axvline(np.log10(cov_min_thresh), color='r', linestyle='--', label=f'cov_min_thresh={cov_min_thresh}')
ax1.legend()
ax1.set_title('histogram for all subjects of variance in the mean')
ax1.set_xlabel('log10(cov_diag)')

# give a title to the figure
dirnm = os.path.basename(os.path.normpath(rootDir_data))
p.suptitle(f'Data set - {dirnm}')

p.savefig( os.path.join(rootDir_data, 'derivatives', 'plots', "DQR_group_cov_histogram.png") )

p.show()

# %% Plot the weighted mean signal and the SE
##############################################################################

t_win = [10, 20]
idx_reltime = np.where((od_epochs_mean.reltime.values >= t_win[0]) & (od_epochs_mean.reltime.values <= t_win[1]))[0]
f,ax = p.subplots(1,1)
ax.semilogy( np.sqrt(cov_mean_weighted.diagonal()), label='SE' )
ax.semilogy( np.mean(y_mean_weighted[:,idx_reltime],axis=1), label='mean(y)' )
ax.legend()

p.show()



# %% Load the Sensitivity Matrix and Head Model
##############################################################################

path_to_dataset = '/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data/Datasets/BallSqueezing_WHHD/'
head_model = 'ICBM152'

Adot, head = pfDAB_img.load_Adot( path_to_dataset, head_model )



# %% Do the image reconstruction
##############################################################################

trial_type_img = 'ST-weighted' # 'DT', 'DT-ica', 'ST', 'ST-ica'
t_win = (10, 20)
file_save = True

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
alpha_spatial_list = [1e-3]#[1e-2, 1e-4, 1e-5, 1e-3, 1e-1] #[1e-3]


file_path0 = rootDir_data + 'derivatives/processed_data/'
wavelength = rec[0][0]['amp'].wavelength.values
spectrum = 'prahl'


#
# Get the group average image
#

# get the group average HRF over a time window
hrf_conc_mag = blockaverage_all.sel(trial_type=trial_type_img).sel(reltime=slice(t_win[0], t_win[1])).mean('reltime')
hrf_conc_ts = blockaverage_all.sel(trial_type=trial_type_img)

# convert back to OD
E = cedalion.nirs.get_extinction_coefficients(spectrum, wavelength)
hrf_od_mag = xr.dot(E, hrf_conc_mag * 1*units.mm * 1e-6*units.molar / units.micromolar, dim=["chromo"]) # assumes DPF = 1
hrf_od_ts = xr.dot(E, hrf_conc_ts * 1*units.mm * 1e-6*units.molar / units.micromolar, dim=["chromo"]) # assumes DPF = 1

# if 0:
#     idx_reltime = np.where((od_epochs_mean.reltime.values >= t_win[0]) & (od_epochs_mean.reltime.values <= t_win[1]))[0]
#     hrf_od_mag.values = np.reshape( np.mean(y_mean_weighted[:,idx_reltime],axis=1), (2,n_chs) )

trial_type_img_split = trial_type_img.split('-')
flag = False
if len(trial_type_img_split) > 1:
    if trial_type_img_split[1] == 'weighted':
        flag = True

if not flag:    
    X_grp, W, AAT_norm = pfDAB_img.do_image_recon( hrf_od_mag, head, Adot, None, wavelength, BRAIN_ONLY, SB, sb_cfg, alpha_spatial_list, alpha_meas_list, file_save, file_path0, trial_type_img)
else:
    X_grp, W, AAT_norm = pfDAB_img.do_image_recon( hrf_od_mag, head, Adot, cov_mean_weighted, wavelength, BRAIN_ONLY, SB, sb_cfg, alpha_spatial_list, alpha_meas_list, file_save, file_path0, trial_type_img)





# %% Calculate the image noise and image CNR
##############################################################################

thresh_noise = 2e-6
thresh_tstat = 20



cov_img_tmp = W @ cov_mean_weighted
cov_img_diag = np.nansum(cov_img_tmp * W, axis=1 )

nV = X_grp.shape[0]
cov_img_diag = np.reshape( cov_img_diag, (2,nV) ).T

# image noise
X_noise = X_grp.copy()
X_noise.values = np.sqrt(cov_img_diag)

X_noise[ np.where(X_noise[:,0]>thresh_noise)[0], 0 ] = thresh_noise
X_noise[ np.where(X_noise[:,1]>thresh_noise)[0], 1 ] = thresh_noise

X_noise[ np.where(X_noise[:,0]<-thresh_noise)[0], 0 ] = -thresh_noise
X_noise[ np.where(X_noise[:,1]<-thresh_noise)[0], 1 ] = -thresh_noise

filepath = os.path.join(file_path0, f'X_noise_{trial_type_img}_cov_alpha_spatial_{alpha_spatial_list[-1]:.0e}_alpha_meas_{alpha_meas_list[-1]:.0e}.pkl.gz')
print(f'   Saving to X_noise_{trial_type_img}_cov_alpha_spatial_{alpha_spatial_list[-1]:.0e}_alpha_meas_{alpha_meas_list[-1]:.0e}.pkl.gz')
file = gzip.GzipFile(filepath, 'wb')
file.write(pickle.dumps([X_noise, alpha_meas_list[-1], alpha_spatial_list[-1]]))
file.close()  

# image t-stat (i.e. CNR)
X_tstat = X_grp / np.sqrt(cov_img_diag)

X_tstat[ np.where(cov_img_diag[:,0]==0)[0], 0 ] = 0
X_tstat[ np.where(cov_img_diag[:,1]==0)[0], 1 ] = 0

X_tstat[ np.where(X_tstat[:,0]>thresh_tstat)[0], 0 ] = thresh_tstat
X_tstat[ np.where(X_tstat[:,1]>thresh_tstat)[0], 1 ] = thresh_tstat

X_tstat[ np.where(X_tstat[:,0]<-thresh_tstat)[0], 0 ] = -thresh_tstat
X_tstat[ np.where(X_tstat[:,1]<-thresh_tstat)[0], 1 ] = -thresh_tstat

filepath = os.path.join(file_path0, f'X_tstat_{trial_type_img}_cov_alpha_spatial_{alpha_spatial_list[-1]:.0e}_alpha_meas_{alpha_meas_list[-1]:.0e}.pkl.gz')
print(f'   Saving to X_tstat_{trial_type_img}_cov_alpha_spatial_{alpha_spatial_list[-1]:.0e}_alpha_meas_{alpha_meas_list[-1]:.0e}.pkl.gz')
file = gzip.GzipFile(filepath, 'wb')
file.write(pickle.dumps([X_tstat, alpha_meas_list[-1], alpha_spatial_list[-1]]))
file.close()     



# %% Plot the images
##############################################################################

p0 = pfDAB_img.plot_image_recon(X_tstat, head, 'hbo_brain', 'left')

#p0.disable()



# %% plot SVS of the MSE compared with that of A A.T
##############################################################################

u,s,v = np.linalg.svd(AAT_norm)
u1,s1,v1 = np.linalg.svd(cov_mean_weighted*alpha_meas_list[-1])

f,ax = p.subplots(2,1,figsize=(8,10))

ax1 = ax[0]
ax1.semilogy(AAT_norm.diagonal(),label='A A.T')
ax1.semilogy(cov_mean_weighted.diagonal()*alpha_meas_list[-1],label='MSE')
ax1.legend()
ax1.set_title(fr'Diagonal of (A A.T)/norm and MSE*$\alpha_{{meas}}$={alpha_meas_list[-1]:.2e}')

ax1 = ax[1]
ax1.semilogy(s,label='A A.T')
ax1.semilogy(s1,label='MSE')
ax1.legend()
ax1.set_title(r'Singular Values of (A A.T)/norm and MSE*$\alpha_{{meas}}$')

p.show()

# give a title to the figure
dirnm = os.path.basename(os.path.normpath(rootDir_data))
p.suptitle(f'Data set - {dirnm}')

p.savefig( os.path.join(rootDir_data, 'derivatives', 'plots', "DQR_group_AAT_svs.png") )

p.show()




# %% Parcels
##############################################################################
# list unique parcels in X
parcels = np.unique(X_grp['parcel'].values)
#parcels

# parcels with '_LH' at the end
parcels_LH = [x for x in parcels if x.endswith('_LH')]
#parcels_LH

# %%

Xo = X_tstat.sel(chromo='HbO')

# Create a mapping from vertex to parcel
vertex_to_parcel = Xo['parcel'].values

# Add the parcel information as a coordinate to the DataArray/Dataset
Xo = Xo.assign_coords(parcel=('vertex', vertex_to_parcel))

# Group by the parcel coordinate and calculate the mean over the vertex dimension
Xo_parcel = Xo.groupby('parcel').mean(dim='vertex')

# Verify the result
#display(Xo)
#display(Xo_parcel)

# find Xo_parcel values > 2 and from parcels_LH
Xo_parcel_2 = Xo_parcel.where(np.abs(Xo_parcel) > 5).dropna('parcel').where(Xo_parcel['parcel'].isin(parcels_LH)).dropna('parcel')

# Xo_parcel_2 = Xo_parcel.where(Xo_parcel > 3)
# Xo_parcel_2 = Xo_parcel_2.dropna(dim='parcel')

# %%
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


# %%
# get the time series for the parcels
Xo_ts = X_grp_ts #.sel(chromo='HbO')
vertex_to_parcel = Xo_ts['parcel'].values
Xo_ts = Xo_ts.assign_coords(parcel=('vertex', vertex_to_parcel))
Xo_ts_parcel = Xo_ts.groupby('parcel').mean(dim='vertex')

# plot the significant parcels
foo = Xo_ts_parcel.sel(parcel=Xo_parcel_2.parcel.values)

f, ax = p.subplots(1, 1, figsize=(10, 5))
for i in range(foo.sizes['parcel']):
    line, = ax.plot(foo['reltime'], foo.sel(parcel=foo['parcel'][i], chromo='HbO'), label=foo['parcel'][i].values)
    ax.plot(foo['reltime'], foo.sel(parcel=foo['parcel'][i], chromo='HbR'), linestyle='--', color=line.get_color())
ax.set_title('Significant parcels')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Concentration (M)')
ax.legend()
p.show()










# %% Old Code
##############################################################################
##############################################################################


# %% Plot the HRF CNR for one channel
##############################################################################

ich = 10+567

foo = np.sqrt(cov_mean_weighted.diagonal())

f,ax = p.subplots(1,2)
ax1 = ax[0]
#ax1.plot( od_epochs_mean.reltime.values, np.mean(y_mean_weighted,axis=0) )
ax1.plot( od_epochs_mean.reltime.values, y_mean_weighted[ich,:], label='y_mean_weighted' )
ax1.plot( od_epochs_mean.reltime.values, foo[ich] * np.ones_like(od_epochs_mean.reltime.values), label='SE_o' )
ax1.plot( od_epochs_mean.reltime.values, np.sqrt(cov_t_mean_weighted[ich,:]), label='SE(t)' )
ax1.legend()

ax1 = ax[1]
ax1.plot( od_epochs_mean.reltime.values, y_mean_weighted[ich,:] / foo[ich] * np.ones_like(od_epochs_mean.reltime.values), label='constant noise' )
ax1.plot( od_epochs_mean.reltime.values, y_mean_weighted[ich,:] / (np.sqrt(cov_t_mean_weighted[ich,:])), label='varying noise' )
ax1.legend()
ax1.set_title('CNR')

p.show()

# %% Save the HRF CNR to visualize with plot probe
##############################################################################

stderr_constant = np.sqrt(cov_mean_weighted.diagonal()) / np.sqrt(n_subjects)
stderr_varying = np.sqrt(cov_t_mean_weighted) / np.sqrt(n_subjects)


foo_conc_tmp = foo_conc.copy()

# w1
stderr_varying_da = xr.DataArray(stderr_varying[:n_chs,:]*units.micromolar, dims=foo_conc_tmp[1,0,:,:].dims, coords=foo_conc_tmp[1,0,:,:].coords)
foo_conc_tmp[0,0,:,:] = stderr_varying_da

stderr_constant_t = stderr_constant[:n_chs]
stderr_constant_t = np.tile(stderr_constant_t, (len(od_epochs_mean.reltime.values),1)).T
stderr_constant_t_da = xr.DataArray(stderr_constant_t*units.micromolar, dims=foo_conc_tmp[1,0,:,:].dims, coords=foo_conc_tmp[1,0,:,:].coords)
foo_conc_tmp[0,1,:,:] = stderr_constant_t_da

# w1
stderr_varying_da = xr.DataArray(stderr_varying[n_chs:,:]*units.micromolar, dims=foo_conc_tmp[1,0,:,:].dims, coords=foo_conc_tmp[1,0,:,:].coords)
foo_conc_tmp[1,0,:,:] = stderr_varying_da

stderr_constant_t = stderr_constant[n_chs:]
stderr_constant_t = np.tile(stderr_constant_t, (len(od_epochs_mean.reltime.values),1)).T
stderr_constant_t_da = xr.DataArray(stderr_constant_t*units.micromolar, dims=foo_conc_tmp[1,0,:,:].dims, coords=foo_conc_tmp[1,0,:,:].coords)
foo_conc_tmp[1,1,:,:] = stderr_constant_t_da


foo_conc_tmp = foo_conc_tmp.assign_coords(trial_type=['stderr w1', 'stderr w2'])

file_path_pkl = os.path.join(rootDir_data, 'derivatives', 'processed_data', 'blockaverage.pkl.gz')
file = gzip.GzipFile(file_path_pkl, 'wb')
file.write(pickle.dumps([foo_conc_tmp, rec[0][0].geo2d, rec[0][0].geo3d]))
file.close()

print('Saved group average HRF to ' + file_path_pkl)

