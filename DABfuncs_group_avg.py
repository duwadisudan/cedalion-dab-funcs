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
