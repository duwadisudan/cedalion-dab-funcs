import os

import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality
import cedalion.sigproc.frequency as frequency
import cedalion.sigproc.motion_correct as motion_correct
import cedalion.xrutils as xrutils
import cedalion.datasets as datasets
import xarray as xr
#import cedalion.vis.time_series as vTimeSeries
from cedalion import units
import numpy as np


from sklearn.decomposition import PCA
from cedalion.sigdecomp.ERBM import ERBM
from cedalion.sigdecomp.ICA_EBM import ICA_EBM as EBM
from scipy import stats



def ERBM_run_ica( rec, filenm_lst, flag_ICA_use_pruned_data, ica_lpf, ica_downsample, cov_amp_thresh, chs_pruned_subjs, pca_var_thresh, flag_do_pca_filter, flag_calculate_ICA_matrix, flag_ERBM_vs_EBM, p_ica, rootDir_data, flag_do_ica_filter, ica_spatial_mask_thresh, ica_tstat_thresh, trange_hrf, trange_hrf_stat, stim_lst_hrf_ica ):

    n_subjects = len(rec)
    n_files_per_subject = len(rec[0])

    print('')
    print(f'Starting ICA calculation for {n_subjects} subjects with {n_files_per_subject} files per subject')
    if flag_calculate_ICA_matrix:
        if flag_ERBM_vs_EBM:
            print('Calculating ICA ERBM matrix for each file')
        else:
            print('Calculating ICA EBM matrix for each file')
    print(f'   ICA low pass filter: {ica_lpf}')
    print(f'   ICA downsample factor: {ica_downsample}')
    print(f'   ICA use pruned data: {flag_ICA_use_pruned_data}')

    for subj_idx in range( n_subjects ):
        for file_idx in range(n_files_per_subject):

            filenm = filenm_lst[subj_idx][file_idx]
            print(f'Processing {filenm}')

            # the TS data to get the ICA of. 
            if flag_ICA_use_pruned_data:
                foo = rec[subj_idx][file_idx]["od_tddr"].copy()
            else:
                foo = rec[subj_idx][file_idx]["od_o_tddr"].copy()

            # filter foo with ica_lpf and then downsample and stack it
            foo = cedalion.sigproc.frequency.freq_filter(foo, 0 * units.Hz, ica_lpf )
            foo = foo[:,:,::ica_downsample]
            TS = foo.stack(measurement = ['channel', 'wavelength']).sortby('wavelength')

            if flag_ICA_use_pruned_data:
                S_pca_thresh, W_pca, num_components = ERBM_pca_step( TS, pca_var_thresh, flag_ICA_use_pruned_data )
            else:
                amp = rec[subj_idx][file_idx]['amp'].mean('time') 
                amp = amp.stack(measurement=['channel', 'wavelength']).sortby('wavelength').transpose()
                idx_amp = np.where(amp < cov_amp_thresh)[0] # list of channels with too low signal
                idx_sat = np.where(chs_pruned_subjs[subj_idx][file_idx] == 0.0)[0] # list of saturated channels
                n_chs = int(len(amp)//2)

                TS[:,idx_amp] = 0
                TS[:,idx_sat] = 0
                TS[:,idx_sat+n_chs] = 0

                S_pca_thresh, W_pca, num_components = ERBM_pca_step( TS, pca_var_thresh, flag_ICA_use_pruned_data )
            print(f'   number of PCA components kept: {num_components}')

            if flag_do_pca_filter:
                # scale the columns of new_ts by ts_std
                ts_mean = TS.mean('time') # needed for projecting back to channel space from PCA space
                ts_std = TS.std('time')
                ts_zscore = stats.zscore(TS.values, axis=0)

                # get indices of mean_ts_zscore with NaN
                mean_ts_zscore = ts_zscore.mean(axis=0)
                idx_not_nan = np.where(~np.isnan(mean_ts_zscore))[0]

                # project back to channel space
                new_ts = np.full((ts_zscore.shape[0],ts_std.shape[0]), np.nan)
                if flag_ICA_use_pruned_data:
                    new_ts[:,idx_not_nan] = S_pca_thresh @ W_pca
                else:
                    new_ts = S_pca_thresh @ W_pca

                ts_std_values = ts_std.values
                if flag_ICA_use_pruned_data:
                    new_ts[:,idx_not_nan] = new_ts[:,idx_not_nan] @ np.diag(ts_std_values[idx_not_nan]) + ts_mean[idx_not_nan].values
                else:
                    new_ts[:,idx_not_nan] = new_ts[:,idx_not_nan] @ np.diag(ts_std_values[idx_not_nan]**2) + ts_mean[idx_not_nan].values

                new_xr = xr.zeros_like(TS)
                new_xr.values = new_ts
                new_xr = new_xr.unstack("measurement")

                detector_coord = new_xr["detector"].data[:, 0]
                new_xr = new_xr.assign_coords(detector=("channel", detector_coord))
                source_coord = new_xr["source"].data[:, 0]
                new_xr = new_xr.assign_coords(source=("channel", source_coord))

                new_xr = new_xr.transpose("channel", "wavelength", "time")
                new_xr.time.attrs['units'] = 'second'

                # convert to concentration
                dpf = xr.DataArray(
                    [1, 1],
                    dims="wavelength",
                    coords={"wavelength": rec[subj_idx][file_idx]['amp'].wavelength},
                )

                if flag_ICA_use_pruned_data:
                    rec[subj_idx][file_idx]['od_tddr_pca'] = new_xr
                    rec[subj_idx][file_idx]['conc_tddr_pca'] = cedalion.nirs.od2conc(rec[subj_idx][file_idx]['od_tddr_pca'], rec[subj_idx][file_idx].geo3d, dpf, spectrum="prahl")
                else:
                    rec[subj_idx][file_idx]['od_o_tddr_pca'] = new_xr
                    rec[subj_idx][file_idx]['conc_o_tddr_pca'] = cedalion.nirs.od2conc(rec[subj_idx][file_idx]['od_o_tddr_pca'], rec[subj_idx][file_idx].geo3d, dpf, spectrum="prahl")  


            if flag_calculate_ICA_matrix:
                # ICA-ERBM on PCs
                import time
                from datetime import datetime
                start_time = time.time()
                if flag_ERBM_vs_EBM:
                    print(f'   start calculating ICA ERBM matrix at {datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")}')
                    W_ica = ERBM(S_pca_thresh.T, p_ica )
                else:
                    print(f'   start calculating ICA EBM matrix at {datetime.fromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")}')
                    W_ica = EBM(S_pca_thresh.T )
                end_time = time.time()
                execution_time = end_time - start_time
                print( f"   ICA execution time: {execution_time/60:0.1f} minutes")
            
                # Save W_ica to a file
                file_path = os.path.join(rootDir_data, 'derivatives', 'ica', filenm )
                if flag_ERBM_vs_EBM:
                    file_path = file_path + '_Wica_ERBM'
                else:
                    file_path = file_path + '_Wica_EBM'
                if flag_ICA_use_pruned_data:
                    np.savez(file_path + f'_od_tddr_ds{ica_downsample}.npz', W_ica=W_ica )
                else:
                    np.savez(file_path + f'_od_o_tddr_ds{ica_downsample}.npz', W_ica=W_ica )
                    

            if flag_do_ica_filter:
                # load W_ica from file
                file_path = os.path.join(rootDir_data, 'derivatives', 'ica', filenm )
                if flag_ERBM_vs_EBM:
                    file_path = file_path + '_Wica_ERBM'
                else:
                    file_path = file_path + '_Wica_EBM'
                if flag_ICA_use_pruned_data:
                    W_ica = np.load(file_path + f'_od_tddr_ds{ica_downsample}.npz')['W_ica']
                else:
                    W_ica = np.load(file_path + f'_od_o_tddr_ds{ica_downsample}.npz')['W_ica']

                # project to ICA space
                S_ica = W_ica @ S_pca_thresh.T

                # do the ICA filter
                stim = rec[subj_idx][file_idx].stim.copy()
                # FIXME: this must properly scale back to channel space if NOT using pruned data. I think I did this, but it needs checking
                if flag_ICA_use_pruned_data:
                    rec[subj_idx][file_idx]['od_tddr_ica'], num_components_sig_ica, num_components_remove, num_components_sig_minus_remove = ERBM_ica_step(
                        TS, stim, W_pca, W_ica, S_ica, trange_hrf, trange_hrf_stat, ica_spatial_mask_thresh, ica_tstat_thresh, stim_lst_hrf_ica, flag_ICA_use_pruned_data
                    )
                    # interpolate back to original time points if necessary
                    if ica_downsample > 1:
                        foo = rec[subj_idx][file_idx]['od_tddr_ica']
                        foo = foo.interp(time=rec[subj_idx][file_idx]['amp'].time) 
                        foo = foo.assign_coords(samples=("time", np.arange(foo.shape[2])))
                        rec[subj_idx][file_idx]['od_tddr_ica'] = foo

                    # convert to concentration
                    dpf = xr.DataArray(
                        [1, 1],
                        dims="wavelength",
                        coords={"wavelength": rec[subj_idx][file_idx]['amp'].wavelength},
                    )
                    rec[subj_idx][file_idx]['conc_tddr_ica'] = cedalion.nirs.od2conc(rec[subj_idx][file_idx]['od_tddr_ica'], rec[subj_idx][file_idx].geo3d, dpf, spectrum="prahl")
                else:
                    rec[subj_idx][file_idx]['od_o_tddr_ica'], num_components_sig_ica, num_components_remove, num_components_sig_minus_remove = ERBM_ica_step(
                        TS, stim, W_pca, W_ica, S_ica, trange_hrf, trange_hrf_stat, ica_spatial_mask_thresh, ica_tstat_thresh, stim_lst_hrf_ica, flag_ICA_use_pruned_data
                    )
                    # interpolate back to original time points if necessary
                    if ica_downsample > 1:
                        foo = rec[subj_idx][file_idx]['od_o_tddr_ica']
                        foo = foo.interp(time=rec[subj_idx][file_idx]['amp'].time)
                        foo = foo.assign_coords(samples=("time", np.arange(foo.shape[2])))
                        rec[subj_idx][file_idx]['od_o_tddr_ica'] = foo

                    # convert to concentration
                    dpf = xr.DataArray(
                        [1, 1],
                        dims="wavelength",
                        coords={"wavelength": rec[subj_idx][file_idx]['amp'].wavelength},
                    )
                    rec[subj_idx][file_idx]['conc_o_tddr_ica'] = cedalion.nirs.od2conc(rec[subj_idx][file_idx]['od_o_tddr_ica'], rec[subj_idx][file_idx].geo3d, dpf, spectrum="prahl")

                print(f'   number of significant ICA components: {num_components_sig_ica}')
                print(f'   number of ICA components identified by spatial mask: {num_components_remove}')
                print(f'   number of significant ICA components removed: {num_components_sig_ica-num_components_sig_minus_remove}')
                print(f'   number of ICA components kept: {num_components_sig_minus_remove}')

    print('Done with ERBM_run_ica()')
    return rec




def ERBM_pca_step( TS, var_thresh = 0.99, flag_ICA_use_pruned_data = True ):

    ts_zscore = stats.zscore(TS.values, axis=0)
    if not flag_ICA_use_pruned_data:
        ts_zscore = ts_zscore/np.sqrt(np.var(TS.values,axis=0))

    # get indices of mean_ts_zscore with NaN
    mean_ts_zscore = ts_zscore.mean(axis=0)
    idx_nan = np.where(np.isnan(mean_ts_zscore))[0]

    # remove columns with NaN
    if flag_ICA_use_pruned_data:
        ts_zscore = np.delete(ts_zscore, idx_nan, axis=1)
    else:
        ts_zscore[:,idx_nan] = 0 # instead of pruning, set to zero to indicate no signal

    #% run ICA algorithm
    # PCA on segment
    pca = PCA()
    S_pca = pca.fit_transform(ts_zscore)
    explained = pca.explained_variance_ratio_

    # Calculate cumulative explained variance
    cumulative_explained = np.cumsum(explained) #*100

    # Find the number of components required to meet the variance threshold
    num_components = np.argmax(cumulative_explained >= var_thresh) + 1

    # Calculate cumulative explained variance
    cumulative_explained = np.cumsum(explained) #*100

    # Find the number of components required to meet the variance threshold
    num_components = np.argmax(cumulative_explained >= var_thresh) + 1
 
    # Select the subset of principal components that meet the variance threshold
    S_pca_thresh = S_pca[:, :num_components]
    W_pca = pca.components_[:num_components, :]

    return S_pca_thresh, W_pca, num_components





def ERBM_ica_step( TS, stim, W_pca, W_ica, S_ica, trange_hrf, trange_hrf_stat, ica_spatial_mask_thresh, ica_tstat_thresh, stim_lst_hrf, flag_ICA_use_pruned_data = True  ):

    ts_mean = TS.mean('time') # needed for projecting back to channel space from PCA space
    ts_std = TS.std('time')
    ts_zscore = stats.zscore(TS.values, axis=0)

    # get indices of mean_ts_zscore with NaN
    mean_ts_zscore = ts_zscore.mean(axis=0)
    idx_nan = np.where(np.isnan(mean_ts_zscore))[0]
    idx_not_nan = np.where(~np.isnan(mean_ts_zscore))[0]

    # remove columns with NaN
    if flag_ICA_use_pruned_data:
        ts_zscore = np.delete(ts_zscore, idx_nan, axis=1)
    else:    
        ts_zscore[:,idx_nan] = 0

    #
    # block average the ICA components
    #

    # create xarray
    S_ica_xr = xr.DataArray(
        S_ica.reshape(S_ica.shape[0], 1, S_ica.shape[1]),
        dims=["channel", "subject", "time"],
    #        coords={"channel": np.arange(1, S_ica.shape[0]+1), "subject": [1], "time": rec[subj_idx][file_idx]["conc"].time.values * units.s}
        coords={"channel": np.arange(1, S_ica.shape[0]+1), "subject": [1], "time": TS.time.values * units.s}
    )
    S_ica_xr = S_ica_xr.assign_coords(samples=("time", np.arange(S_ica_xr.shape[2])))
    S_ica_xr = S_ica_xr.transpose("subject", "channel", "time")
    S_ica_xr.time.attrs['units'] = 'second'

    # get the blocks
    S_ica_xr_epochs = S_ica_xr.cd.to_epochs(
        stim,  # stimulus dataframe
        stim_lst_hrf,  # select events    
        before=trange_hrf[0],  # seconds before stimulus 
        after=trange_hrf[1]  # seconds after stimulus
    )

    # baseline subtract
    S_ica_xr_epochs = S_ica_xr_epochs - S_ica_xr_epochs.sel(reltime=slice(-trange_hrf[0].magnitude, 0)).mean("reltime")

    # get the mean and std
    blockaverage_mean = S_ica_xr_epochs.groupby("trial_type").mean("epoch")
    blockaverage_std = S_ica_xr_epochs.groupby("trial_type").std("epoch") 

    # t-stat
    t_stat = blockaverage_mean / (blockaverage_std / np.sqrt(S_ica_xr_epochs.sizes["epoch"]))

    # get the mean t-stat over the time range
    t_stat_mean = t_stat.sel(reltime=slice(trange_hrf_stat[0], trange_hrf_stat[1])).mean("reltime")

    # get component indices that are significant
    t_stat_mean_max = np.nanmax(np.abs(t_stat_mean), axis=0) # max over trial types
    idx_sig = np.where(np.abs(t_stat_mean_max[0,:]) > ica_tstat_thresh)[0]

    #
    # get the components with broad spatial distribution in the same direction
    #

    # set new_ts to NaN with the same shape as ts_zscore
    T_ica_to_ch = np.full((W_ica.shape[0], ts_std.shape[0]), np.nan)
    # project back to channel space
    if flag_ICA_use_pruned_data:
        T_ica_to_ch[:,idx_not_nan] = np.linalg.pinv(W_ica).T @ W_pca
    else:
        T_ica_to_ch = np.linalg.pinv(W_ica).T @ W_pca


    # get the norm of each component ignoring the NaNs
    norm = np.sqrt( np.nansum( T_ica_to_ch * T_ica_to_ch, axis=1 ) )

    foo = T_ica_to_ch / norm[:, np.newaxis]
    metric = np.nanmean(foo, axis=1)

    idx_remove = np.where( abs(metric) > (ica_spatial_mask_thresh * np.std(metric))  )[0]

    # keep indices from idx_sig that are not in idx_remove
    idx_sig_minus_remove = np.setdiff1d(idx_sig, idx_remove)

    # plot the metric
    # ax[idx_file//ax_ncol][idx_file%ax_ncol].plot(abs(metric))
    # ax[idx_file//ax_ncol][idx_file%ax_ncol].plot(idx_sig, abs(metric[idx_sig]), 'go', markerfacecolor='none')
    # ax[idx_file//ax_ncol][idx_file%ax_ncol].plot(idx_remove, abs(metric[idx_remove]), 'rx')
    # ax[idx_file//ax_ncol][idx_file%ax_ncol].set_title(f"s{idx_file}, #C={len(idx_sig_minus_remove)}")

    num_components_sig = len(idx_sig)
    num_components_remove = len(idx_remove)
    num_components_sig_minus_remove = len(idx_sig_minus_remove)

    #
    # project ICA back to channel space
    #

    # get the indices of the components to project back
    idx = idx_sig_minus_remove
                        
    # S_ica_idx contains the components you want to project back, otherwise zeros 
    S_ica_idx = np.zeros_like(S_ica)
    S_ica_idx[idx, :] = S_ica[idx, :]
                                    
    # project back to PC space
    S_pca_idx = np.linalg.pinv(W_ica) @ S_ica_idx

    # set new_ts to NaN with the same shape as ts_zscore
    new_ts = np.full((ts_zscore.shape[0],ts_std.shape[0]), np.nan)

    # project back to channel space
    if flag_ICA_use_pruned_data:
        new_ts[:,idx_not_nan] = S_pca_idx.T @ W_pca
    else:
        new_ts = S_pca_idx.T @ W_pca

    # scale the columns of new_ts by ts_std
    ts_std_values = ts_std.values
    if flag_ICA_use_pruned_data:
        new_ts[:,idx_not_nan] = new_ts[:,idx_not_nan] @ np.diag(ts_std_values[idx_not_nan]) + ts_mean[idx_not_nan].values
    else:
        new_ts[:,idx_not_nan] = new_ts[:,idx_not_nan] @ np.diag(ts_std_values[idx_not_nan]**2) + ts_mean[idx_not_nan].values


    new_xr = xr.zeros_like(TS)
    new_xr.values = new_ts
    new_xr = new_xr.unstack("measurement")

    detector_coord = new_xr["detector"].data[:, 0]
    new_xr = new_xr.assign_coords(detector=("channel", detector_coord))
    source_coord = new_xr["source"].data[:, 0]
    new_xr = new_xr.assign_coords(source=("channel", source_coord))

    new_xr = new_xr.transpose("channel", "wavelength", "time")
    new_xr.time.attrs['units'] = 'second'

    return new_xr, num_components_sig, num_components_remove, num_components_sig_minus_remove






