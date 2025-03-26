import cedalion
import cedalion.nirs
import cedalion.sigproc.quality as quality
import cedalion.sigproc.frequency as frequency
import cedalion.xrutils as xrutils
import cedalion.datasets as datasets
import xarray as xr
import matplotlib.pyplot as p
import cedalion.plots as plots
from cedalion import units
import numpy as np

from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform




def getCorrMatrix( conc_hbo = None, flag_GMS = "all" ):

        # global mean subtract conc_hbo
    if flag_GMS == "all": # do GMS with all channels or SS channels
        # select all channels
        gms = conc_hbo.mean(dim='channel')

        # fit GMS to the channel data
        numerator = (conc_hbo * gms).sum(dim="time")
        denomenator = (gms * gms).sum(dim="time")
        scl = numerator/denomenator

        conc_hbo_gms = conc_hbo - scl*gms

    elif flag_GMS == "ss": # do GMS with SS channels
        # select SS channels only
        gms = conc_hbo.sel(channel=conc_hbo.channel.str.contains("D31|D32|D33|D34|D35|D36|D37|D38")).mean(dim='channel')
        # FIX ME. Quality.sd_dist should be used to determine which channels are SS and LS

        # fit GMS to the channel data
        numerator = (conc_hbo * gms).sum(dim="time")
        denomenator = (gms * gms).sum(dim="time")
        scl = numerator/denomenator

        conc_hbo_gms = conc_hbo - scl*gms

    elif flag_GMS == "ss_ls": # do GMS with SS channels and then LS channels
        # SS channels
        gms = conc_hbo.sel(channel=conc_hbo.channel.str.contains("D31|D32|D33|D34|D35|D36|D37|D38")).mean(dim='channel')

        numerator = (conc_hbo * gms).sum(dim="time")
        denomenator = (gms * gms).sum(dim="time")
        scl = numerator/denomenator
        conc_hbo_gms = conc_hbo - scl*gms

        # LS channels
        gms = conc_hbo_gms.sel(channel=~conc_hbo.channel.str.contains("D31|D32|D33|D34|D35|D36|D37|D38")).mean(dim='channel')

        numerator = (conc_hbo_gms * gms).sum(dim="time")
        denomenator = (gms * gms).sum(dim="time")
        scl = numerator/denomenator
        conc_hbo_gms = conc_hbo_gms - scl*gms

    else:
        conc_hbo_gms = conc_hbo
        
    # calculate the correlation matrix of conc_hbo
    conc_hbo_array = conc_hbo_gms.values

    if 1: # correlation matrix
        corr_matrix = np.corrcoef(conc_hbo_array, rowvar=True)
        corr_max = 1

    else: # partial correlation matrix
        regularization = 1e2
        # Step 1: Calculate the covariance matrix
        cov_matrix = np.cov(conc_hbo_array, rowvar=True)    
        # Step 2: Regularize by adding a small value to the diagonal
        cov_matrix += np.eye(cov_matrix.shape[0]) * regularization    
        # Step 3: Invert the covariance matrix to get the precision matrix
        precision_matrix = np.linalg.inv(cov_matrix)
        # Step 4: Convert the precision matrix to the partial correlation matrix
        partial_corr = precision_matrix / np.sqrt(np.outer(np.diag(precision_matrix), np.diag(precision_matrix)))
        # Step 5: Ensure the diagonal is all zeros
        np.fill_diagonal(partial_corr, 0)

        corr_matrix = partial_corr
#        corr_max = np.max(np.abs(corr_matrix))

        

    # Ensure corr_matrix_xr has two dimensions, both of which are channel
    corr_matrix_xr = xr.DataArray(
        corr_matrix,
    #    cov_matrix,
        dims=["channel", "channel"],
        coords={"channel": conc_hbo.channel.values, "channel": conc_hbo.channel.values}
    )

    return corr_matrix_xr, conc_hbo_gms


def calc_dFC( conc_hbo_array_clusters = None, t = None, window_size_s = 20 ):

    fs = 1 / np.mean(np.diff(t))
    window_size = window_size_s * fs
    window_size = int(window_size.round())

    max_cluster_label = conc_hbo_array_clusters.shape[0]

    # initialize the correlation matrix
    corr_time_clusters = np.zeros([max_cluster_label, max_cluster_label, conc_hbo_array_clusters.shape[1] - window_size])

    # get the tcorr vector
    # tcorr = np.zeros(corr_time_clusters.shape[2])
    # for i in range(0, corr_time_clusters.shape[2], 1):
    #     tcorr[i] = t[i + window_size // 2]
    # Assuming tcorr and t are NumPy arrays and window_size is defined
    tcorr = t[window_size // 2 : corr_time_clusters.shape[2] + window_size // 2]

    # For loop over c1 
    for c1 in range(0, max_cluster_label):
        for c2 in range(c1, max_cluster_label):

            if c1 == c2:
                corr_time_clusters[c1,c2,:] = conc_hbo_array_clusters[c1,window_size//2:-window_size//2]
            else:
                # calculate the correlation between the two clusters over a sliding window
                y1 = conc_hbo_array_clusters[c1,:]
                y2 = conc_hbo_array_clusters[c2,:]

                # calculate the correlation between the two channels over a sliding window
                for i in range(0, y1.shape[0] - window_size, 1):
                    corr_time_clusters[c1,c2,i] = np.corrcoef(y1[i:i + window_size], y2[i:i + window_size])[0, 1]

    return corr_time_clusters, tcorr


def block_average_clusters( corr_time_clusters, tcorr, stim, events_str, t_before = 2, t_after = 20 ):
    
    max_cluster_label = corr_time_clusters.shape[0]

    if len(events_str) > 2:
        print("Only two events are allowed, but you can fix the code")
        return
    
    duration = np.zeros(2)
    stim_trial_type = stim.loc[stim.trial_type == events_str[0]]
    duration[0] = stim_trial_type.duration.mean()
    stim_trial_type = stim.loc[stim.trial_type == events_str[1]]
    duration[1] = stim_trial_type.duration.mean()

    if duration[0] != duration[1]:
        print("The duration of the two events must be the same, but you can fix the code")
        return

    # put corr_time_clusters into an xarray
    corr_time_clusters_xr = xr.DataArray(
        corr_time_clusters,
        dims=["cluster1", "cluster2", "time"],
        coords={"cluster1": np.arange(1, max_cluster_label+1), "cluster2": np.arange(1, max_cluster_label+1), "time": tcorr}
    )
    corr_time_clusters_xr = corr_time_clusters_xr.assign_coords(samples=("time",np.arange(corr_time_clusters_xr.shape[2])))

    # get the blocks
    corr_time_clusters_xr_epochs = corr_time_clusters_xr.cd.to_epochs(
        stim,  # stimulus dataframe
        events_str,  # select events
        t_before,  # seconds before stimulus
        t_after,  # seconds after stimulus
    )

    # block average
    blockaverage_clusters = corr_time_clusters_xr_epochs.groupby("trial_type").mean("epoch")

    # get t-stat
    # FIX ME - this assumes stim duration is the same for all trials
    foo = corr_time_clusters_xr_epochs.sel(reltime=(corr_time_clusters_xr_epochs.reltime > 0) & (corr_time_clusters_xr_epochs.reltime < stim.duration.mean())).mean("reltime")
    blockaverage_clusters_mean = foo.groupby("trial_type").mean("epoch")
    blockaverage_clusters_std = foo.groupby("trial_type").std("epoch")
    blockaverage_clusters_sem = foo.groupby("trial_type").std("epoch") / np.sqrt(foo.epoch.size)
    # FIX ME - CHECK that this is dividing by the correct number of epochs 

    # FIX ME - this assumes only two trials types
    # calculate the t-statistic between the two trial_types
    blockaverage_clusters_tstat = ( blockaverage_clusters_mean[0,:,:] - blockaverage_clusters_mean[1,:,:] ) / np.sqrt( blockaverage_clusters_sem[0,:,:]**2 + blockaverage_clusters_sem[1,:,:]**2 )

    return blockaverage_clusters, blockaverage_clusters_tstat, blockaverage_clusters_mean, blockaverage_clusters_std, blockaverage_clusters_sem



def corr_cluster( corr_matrix_xr, cluster_threshold ):
    
    # calculate the distance matrix
    #dist_matrix = 1 - np.abs(corr_matrix)
    dist_matrix = 1 - corr_matrix_xr.values

    # Fill NaN values with a specific value (e.g., 0)
    dist_matrix = np.nan_to_num(dist_matrix, nan=2)

    # Ensure the distance matrix is symmetric
    dist_matrix = (dist_matrix + dist_matrix.T) / 2

    # Ensure the diagonal of the distance matrix is zero
    np.fill_diagonal(dist_matrix, 0)

    # calculate the linkage matrix
    linkage_matrix = linkage(squareform(dist_matrix), method="average")

    return linkage_matrix
