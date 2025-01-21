# group_avg_GLM()
# group_avg_block()


import os
import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality

import cedalion.models.glm as glm

from cedalion import units
import numpy as np
import xarray as xr

import matplotlib.pyplot as p



def run_group_block_average( rec, filenm_lst, rec_str, ica_lpf, trange_hrf, stim_lst_hrf, flag_save_each_subj, subj_ids, subj_id_exclude, chs_pruned_subjs, rootDir_data ):

    n_subjects = len(rec)
    n_files_per_subject = len(rec[0])

    print(f"Running group block average for trial_type = '{rec_str}'")

    # loop over subjects and files
    blockaverage_subj = None
    for subj_idx in range( n_subjects ):
        for file_idx in range( n_files_per_subject ):

            filenm = filenm_lst[subj_idx][file_idx]
            print( f"   Running {subj_idx+1} of {n_subjects} subjects : {filenm}" )

            # do the block average on the data in rec[subj_idx][file_idx][rec_str]
            # rec_str could be 'conc_tddr', 'conc_tddr_ica' or 'od' or even conc_splineSG
            conc_filt = rec[subj_idx][file_idx][rec_str].copy()
            # FIXME: sould I do this here now that this code has evolved? Seems like it should be handled before this function.
            # FIXME: oddly it is turning my tddr-ica to nan
            # conc_filt = cedalion.sigproc.frequency.freq_filter(conc_filt, 0 * units.Hz, ica_lpf ) # LPF the data to match the ICA data

            #
            # block average
            #

            # select the stim for the given file
            stim = rec[subj_idx][file_idx].stim.copy()

            # get the epochs
            conc_filt = conc_filt.transpose('chromo', 'channel', 'time')
            conc_filt = conc_filt.assign_coords(samples=('time', np.arange(len(conc_filt.time))))
            conc_filt['time'] = conc_filt.time.pint.quantify(units.s)     

            conc_epochs_tmp = conc_filt.cd.to_epochs(
                                        stim,  # stimulus dataframe
                                        set(stim[stim.trial_type.isin(stim_lst_hrf)].trial_type), # select events
                                        before=trange_hrf[0],  # seconds before stimulus
                                        after=trange_hrf[1],  # seconds after stimulus
                                    )
                
            # concatenate the different epochs from each file for each subject
            if file_idx == 0:
                conc_epochs_all = conc_epochs_tmp
            else:
                conc_epochs_all = xr.concat([conc_epochs_all, conc_epochs_tmp], dim='epoch')

            if flag_save_each_subj:
                conc_epochs_tmp = conc_epochs_tmp.assign_coords(trial_type=('epoch', [x + '-' + subj_ids[subj_idx] for x in conc_epochs_tmp.trial_type.values]))
                conc_epochs_all = xr.concat([conc_epochs_all, conc_epochs_tmp], dim='epoch')

            # DONE LOOP OVER FILES

        # Block Average
        baseline_conc = conc_epochs_all.sel(reltime=(conc_epochs_all.reltime < 0)).mean('reltime')
        conc_epochs = conc_epochs_all - baseline_conc
        blockaverage = conc_epochs.groupby('trial_type').mean('epoch')



        # get MSE for weighting across subjects
        # FIXME: this at the moment assumes all epochs are the same trial_type
        #        so we need one trial type and flag_save_each_subj=False
        blockaverage_weighted = blockaverage.copy()
        n_epochs = conc_epochs.shape[0]
        n_chs = conc_epochs.shape[2]
        foo = conc_epochs - blockaverage_weighted[0,:,:,:] # FIXME: assuming one trial_type
        foo_t = foo.stack(measurement=['channel','chromo']).sortby('chromo')
        foo_t = foo_t.transpose('measurement', 'reltime', 'epoch')
        mse_t = (foo_t**2).sum('epoch') / (n_epochs - 1)**2 # this is squared to get variance of the mean, aka MSE of the mean
        # remove the units from mse_t
        # mse_t = mse_t.pint.dequantify()

        # adjust the MSE to handle bad data
        mse_val_for_bad_data = 1e7 * units.micromolar**2 # FIXME: this should probably be passed
        mse_amp_thresh = 1.1e-6 # FIXME: this should probably be passed
        mse_min_thresh = 1e0 * units.micromolar**2 # FIXME: this should probably be passed
                                                    # FIXME: what if OD units?

        # list of channel elements in mse corresponding to channels with amp < mse_amp_thresh
        amp = rec[subj_idx][file_idx]['amp'].mean('time').min('wavelength')
        idx_amp = np.where(amp < mse_amp_thresh)[0]
        mse_t[idx_amp,:] = mse_val_for_bad_data
        mse_t[idx_amp + n_chs,:] = mse_val_for_bad_data
        blockaverage_weighted[0,0,idx_amp,:] = 0 * units.micromolar # FIXME: first dimension is trial_type
        blockaverage_weighted[0,1,idx_amp,:] = 0 * units.micromolar # FIXME: what if is OD units?
        
        # look at saturated channels
        idx_sat = np.where(chs_pruned_subjs[subj_idx][file_idx] == 0.0)[0]
        mse_t[idx_sat,:] = mse_val_for_bad_data
        mse_t[idx_sat + n_chs,:] = mse_val_for_bad_data
        blockaverage_weighted[0,0,idx_sat,:] = 0 * units.micromolar # FIXME: first dimension is trial_type
        blockaverage_weighted[0,1,idx_sat,:] = 0 * units.micromolar # FIXME: what if is OD units?

        # set the minimum value of mse_t
        mse_t = mse_t.unstack('measurement').transpose('chromo','channel','reltime')
        mse_t = mse_t.expand_dims('trial_type')
        source_coord = blockaverage['source']
        mse_t = mse_t.assign_coords(source=('channel',source_coord.data))
        detector_coord = blockaverage['detector']
        mse_t = mse_t.assign_coords(detector=('channel',detector_coord.data))

        mse_t_o = mse_t.copy()
        mse_t = xr.where(mse_t < mse_min_thresh, mse_min_thresh, mse_t)

        # gather the blockaverage across subjects
        if blockaverage_subj is None and subj_ids[subj_idx] not in subj_id_exclude:
            blockaverage_subj = blockaverage
            # add a subject dimension and coordinate
            blockaverage_subj = blockaverage_subj.expand_dims('subj')
            blockaverage_subj = blockaverage_subj.assign_coords(subj=[subj_ids[subj_idx]])

            blockaverage_mse_subj = mse_t_o.expand_dims('subj') 
            blockaverage_mse_subj = blockaverage_mse_subj.assign_coords(subj=[subj_ids[subj_idx]])

            blockaverage_mean_weighted = blockaverage_weighted / mse_t
            blockaverage_mse_inv_mean_weighted = 1 / mse_t

        elif subj_ids[subj_idx] not in subj_id_exclude:
            blockaverage_subj_tmp = blockaverage
            blockaverage_subj_tmp = blockaverage_subj_tmp.expand_dims('subj')
            blockaverage_subj_tmp = blockaverage_subj_tmp.assign_coords(subj=[subj_ids[subj_idx]])
            blockaverage_subj = xr.concat([blockaverage_subj, blockaverage_subj_tmp], dim='subj')

            blockaverage_mse_subj_tmp = mse_t_o.expand_dims('subj')
            blockaverage_mse_subj_tmp = blockaverage_mse_subj_tmp.assign_coords(subj=[subj_ids[subj_idx]])
            blockaverage_mse_subj = xr.concat([blockaverage_mse_subj, blockaverage_mse_subj_tmp], dim='subj')

            blockaverage_mean_weighted = blockaverage_mean_weighted + blockaverage_weighted / mse_t
            blockaverage_mse_inv_mean_weighted = blockaverage_mse_inv_mean_weighted + 1 / mse_t

        else:
            print(f"   Subject {subj_ids[subj_idx]} excluded from group average")

        # DONE LOOP OVER SUBJECTS

    # get the average
    blockaverage_mean = blockaverage_subj.mean('subj')

    # get the weighted average
    blockaverage_mean_weighted = blockaverage_mean_weighted / blockaverage_mse_inv_mean_weighted
    blockaverage_stderr_weighted = np.sqrt(1 / blockaverage_mse_inv_mean_weighted)


    # plot the MSE histogram
    f,ax = p.subplots(2,1,figsize=(6,10))

    # plot the diagonals for all subjects
    ax1 = ax[0]
    foo = blockaverage_mse_subj.mean('reltime')
    foo = foo.stack(measurement=['channel','chromo']).sortby('chromo')
    for i in range(n_subjects):
        ax1.semilogy(foo[i,0,:], linewidth=0.5,alpha=0.5)
    ax1.set_title('variance in the mean for all subjects')
    ax1.set_xlabel('channel')
    ax1.legend()

    # histogram the diagonals
    ax1 = ax[1]
    foo1 = np.concatenate([foo[i][0] for i in range(n_subjects)])
    foo1 = np.where(foo1 < 1e-2, 1e-2, foo1) # set the minimum value to 1e-2
    ax1.hist(np.log10(foo1), bins=100)
    ax1.axvline(np.log10(mse_min_thresh.magnitude), color='r', linestyle='--', label=f'cov_min_thresh={mse_min_thresh.magnitude:.2e}')
    ax1.legend()
    ax1.set_title('histogram for all subjects of variance in the mean')
    ax1.set_xlabel('log10(cov_diag)')

    # give a title to the figure and save it
    dirnm = os.path.basename(os.path.normpath(rootDir_data))
    p.suptitle(f'Data set - {dirnm}')

    p.savefig( os.path.join(rootDir_data, 'derivatives', 'plots', f'DQR_group_mse_histogram_{rec_str}.png') )
    p.close()


    return blockaverage_mean, blockaverage_mean_weighted, blockaverage_stderr_weighted, blockaverage_mse_subj




def block_average_od( od_filt, stim, geo3d, trange_hrf, stim_lst_hrf ):

    # get the epochs
    od_tmp = od_filt.transpose('wavelength', 'channel', 'time')
    od_tmp = od_tmp.assign_coords(samples=('time', np.arange(len(od_tmp.time))))
    od_tmp['time'] = od_tmp.time.pint.quantify(units.s)     
    od_tmp['source'] = od_filt.source
    od_tmp['detector'] = od_filt.detector

    od_epochs = od_tmp.cd.to_epochs(
                                stim,  # stimulus dataframe
                                set(stim[stim.trial_type.isin(stim_lst_hrf)].trial_type), # select events
#                                set(stim.trial_type),  # select events
                                before=trange_hrf[0],  # seconds before stimulus
                                after=trange_hrf[1],  # seconds after stimulus
                            )
        
    return od_epochs


def block_average( conc_filt, stim, geo3d, trange_hrf, glm_basis_func_param, glm_drift_order, flag_do_GLM, ssr_rho_thresh, stim_lst_hrf ):

    # Do GLM or Block Average
    # Right now the GLM doesn't handle NaN's, but pruned channels have NaN's.. curious.
    if not flag_do_GLM:
        # do block average
        pred_hrf = conc_filt

    else:
        # do the GLM

        # get the short separation channels
        ts_long, ts_short = cedalion.nirs.split_long_short_channels(
            conc_filt, geo3d, distance_threshold=ssr_rho_thresh
        )

        # if stim has a column named 'amplitude' then rename it to 'value'
        # I noticed that some events files have 'amplitude' and some have 'value' and some have neither
        if 'amplitude' in stim.columns:
            stim = stim.rename(columns={"amplitude": "value"})
        elif 'amplitude' not in stim.columns and 'value' not in stim.columns:
            stim['value'] = 1

        # make the design matrix
        # ’closest’: Use the closest short channel 
        # ‘max_corr’: Use the short channel with the highest correlation 
        # ‘mean’: Use the average of all short channels.
        dm, channel_wise_regressors = glm.make_design_matrix(
            conc_filt,
            ts_short,
            stim[stim.trial_type.isin(stim_lst_hrf)],
            geo3d,
            basis_function = glm.GaussianKernels(trange_hrf[0], trange_hrf[1], t_delta=glm_basis_func_param, t_std=glm_basis_func_param), 
            drift_order = glm_drift_order, 
            short_channel_method='mean'
        )

        # fit the GLM model
        betas = glm.fit(conc_filt, dm, channel_wise_regressors, noise_model="ols")

        # prediction of all HRF regressors, i.e. all regressors that start with 'HRF '
        pred_hrf = glm.predict(
            conc_filt,
            betas.sel(regressor=betas.regressor.str.startswith("HRF ")),
            dm,
            channel_wise_regressors
        )

    # get the HRF prediction 
    # This is a simple way to get HRF stats by getting epochs and then simple block average, 
    # but it will not work for event related designs with overlapping epochs.
    # In the future we will have to update glm.predict to provide HRF stats.
    pred_hrf = pred_hrf.transpose('chromo', 'channel', 'time')
    pred_hrf = pred_hrf.assign_coords(samples=('time', np.arange(len(pred_hrf.time))))
    pred_hrf['time'] = pred_hrf.time.pint.quantify(units.s)     
    pred_hrf['source'] = conc_filt.source
    pred_hrf['detector'] = conc_filt.detector

    conc_epochs_tmp = pred_hrf.cd.to_epochs(
                                stim,  # stimulus dataframe
                                set(stim[stim.trial_type.isin(stim_lst_hrf)].trial_type), # select events
#                                set(stim.trial_type),  # select events
                                before=trange_hrf[0],  # seconds before stimulus
                                after=trange_hrf[1],  # seconds after stimulus
                            )
        
    return conc_epochs_tmp




def y_mean_to_conc( y_mean_tmp, geo3d, wavelength, source, stim_lst_hrf, cov_mean_weighted, trange_hrf ):

    n_chs = y_mean_tmp.shape[0] // 2

    foo = y_mean_tmp.unstack()
    foo = foo.transpose('channel', 'wavelength', 'reltime')
    foo = foo.rename({'reltime':'time'})
    foo['time'].attrs['units'] = 's'

    dpf = xr.DataArray(
        [1, 1],
        dims="wavelength",
        coords={"wavelength": wavelength},
    )
    foo_conc = cedalion.nirs.od2conc(foo, geo3d, dpf, spectrum="prahl")
    foo_conc = foo_conc.rename({'time':'reltime'})
    foo_conc = foo_conc.assign_coords(source=source)
    foo_conc = foo_conc.expand_dims('trial_type')
    foo_conc = foo_conc.assign_coords(trial_type=stim_lst_hrf)

    # baseline subtract
    foo_conc = foo_conc - foo_conc.sel(reltime=slice(-trange_hrf[0].magnitude, 0)).mean('reltime')

    # set to NaN the noisy channels for viewing purposes
    cov_mean_weighted_diag = cov_mean_weighted.diagonal()
    idx_cov = np.where(cov_mean_weighted_diag > 1e-3)[0]
    idx_cov1 = idx_cov[idx_cov<n_chs]
    idx_cov2 = idx_cov[idx_cov>=n_chs] - n_chs
    idx_cov = np.union1d(idx_cov1, idx_cov2)

    foo_conc_tmp = foo_conc.copy()
    foo_conc_tmp[:,:,idx_cov,:] = np.nan * np.ones(foo_conc[:,:,idx_cov,:].shape) * units.micromolar

    return foo_conc, foo_conc_tmp




def GLM_extract_estimated_hrf( conc_filt, geo3d, stim, trange_hrf, glm_basis_func_param, betas ):

    pred_hrf = conc_filt

    conc_epochs_tmp = pred_hrf.cd.to_epochs(
                                stim,  # stimulus dataframe
                                set(stim.trial_type),  # select events
                                before=trange_hrf[0],  # seconds before stimulus
                                after=trange_hrf[1],  # seconds after stimulus
                            )

    blockaverage = conc_epochs_tmp.groupby("trial_type").mean("epoch")

    # Get GLM prediction for a single trial
    conc_hrf = blockaverage.copy() # I take this from blockaverage because I want timing from 0 as done in to_epochs()
    conc_hrf = conc_hrf.rename({"reltime": "time"})
    conc_hrf = conc_hrf.sel(trial_type=conc_hrf.trial_type[0])
    conc_hrf = conc_hrf.drop("trial_type")
    conc_hrf.values = np.zeros_like(conc_hrf.values)
    conc_hrf = conc_hrf.assign_coords(samples=("time", np.arange(len(conc_hrf.time))))
    conc_hrf['time'] = conc_hrf.time.pint.quantify(units.s)

    stim_fake = stim.copy()
    stim_fake = stim_fake.iloc[0:1]
    stim_fake.onset = 0
    stim_fake.trial_type = 'mnt'

    dm_fake, _ = glm.make_design_matrix(
        conc_hrf,
        None,
        stim_fake,
        geo3d,
        basis_function = glm.GaussianKernels(trange_hrf[0], trange_hrf[1], t_delta=glm_basis_func_param, t_std=glm_basis_func_param), 
        drift_order = None,
        short_channel_method=None,
    )

    pred_hrf = glm.predict(
        conc_hrf,
        betas.sel(regressor=betas.regressor.str.startswith("HRF ")),
        dm_fake,
        None
    )

    pred_hrf = pred_hrf.transpose('chromo','channel','time')
    pred_hrf = pred_hrf.rename({"time": "reltime"})
    pred_hrf = pred_hrf.expand_dims('trial_type')
    pred_hrf = pred_hrf.assign_coords(trial_type=('trial_type',['mnt']))
    pred_hrf = pred_hrf.assign_coords(source=('channel',blockaverage['source'].data))
    pred_hrf = pred_hrf.assign_coords(detector=('channel',blockaverage['detector'].data))

    return pred_hrf
