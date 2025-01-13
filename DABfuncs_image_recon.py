import cedalion_parcellation.datasets as datasets
import cedalion_parcellation.imagereco.forward_model as fw
import cedalion.io as io
import cedalion.nirs as nirs
import xarray as xr
from cedalion import units
import cedalion.dataclasses as cdc 
import numpy as np
import os.path
import pickle
from cedalion.imagereco.solver import pseudo_inverse_stacked

import matplotlib.pyplot as p
import pyvista as pv
from matplotlib.colors import ListedColormap

import gzip

import tkinter as tk
from tkinter import filedialog

import sys
sys.path.append('/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data')
import spatial_basis_funs_ced as sbf 



#
# load in head model and sensitivity profile 
#
def load_Adot( path_to_dataset = None, head_model = 'ICBM152' ):

    # Load the sensitivity profile
    with open(os.path.join(path_to_dataset, 'derivatives', 'fw',  head_model, 'Adot_wParcels.pkl'), 'rb') as f:
        Adot = pickle.load(f)
        
    #% LOAD HEAD MODEL 
    if head_model == 'ICBM152':
        SEG_DATADIR, mask_files, landmarks_file = datasets.get_icbm152_segmentation()
    elif head_model == 'colin27':
        SEG_DATADIR, mask_files, landmarks_file = datasets.get_colin27_segmentation()()

    masks, t_ijk2ras = io.read_segmentation_masks(SEG_DATADIR, mask_files)

    head = fw.TwoSurfaceHeadModel.from_surfaces(
        segmentation_dir=SEG_DATADIR,
        mask_files = mask_files,
        brain_surface_file= os.path.join(SEG_DATADIR, "mask_brain.obj"),
        scalp_surface_file= os.path.join(SEG_DATADIR, "mask_scalp.obj"),
        landmarks_ras_file=landmarks_file,
        smoothing=0.5,
        fill_holes=True,
    ) 
    head.scalp.units = units.mm
    head.brain.units = units.mm

    return Adot, head





def do_image_recon( hrf_od = None, head = None, Adot = None, C_meas = None, wavelength = [760,850], BRAIN_ONLY = False, SB = False, sb_cfg = None, alpha_spatial_list = [1e-3], alpha_meas_list = [1e-3], file_save = False, file_path = None, trial_type_img = None, W = None ):

    print( 'Starting Image Reconstruction')

    #
    # prune the data and sensitivity profile
    #
    od_mag = hrf_od.stack(measurement=('channel', 'wavelength')).sortby('wavelength')

    if len(hrf_od.dims) == 2: # not a time series else it is a time series
        pruning_mask = ~hrf_od.sel(wavelength=850).isnull()
    else: 
        pruning_mask = ~hrf_od.sel(wavelength=850, reltime=0).isnull()

    if BRAIN_ONLY:
        Adot_pruned = Adot[pruning_mask.values, Adot.is_brain.values, :] 
    else:
        Adot_pruned = Adot[pruning_mask.values, :, :]
        
    od_mag_pruned = od_mag.dropna('measurement')


    #
    # create the sensitivity matrix for HbO and HbR
    #
    ec = nirs.get_extinction_coefficients("prahl", wavelength)

    nchannel = Adot_pruned.shape[0]
    nvertices = Adot_pruned.shape[1]
    n_brain = sum(Adot.is_brain.values)

    A = np.zeros((2 * nchannel, 2 * nvertices))

    wl1, wl2 = wavelength
    A[:nchannel, :nvertices] = ec.sel(chromo="HbO", wavelength=wl1).values * Adot_pruned.sel(wavelength=wl1) # noqa: E501
    A[:nchannel, nvertices:] = ec.sel(chromo="HbR", wavelength=wl1).values * Adot_pruned.sel(wavelength=wl1) # noqa: E501
    A[nchannel:, :nvertices] = ec.sel(chromo="HbO", wavelength=wl2).values * Adot_pruned.sel(wavelength=wl2) # noqa: E501
    A[nchannel:, nvertices:] = ec.sel(chromo="HbR", wavelength=wl2).values * Adot_pruned.sel(wavelength=wl2) # noqa: E501

    A = xr.DataArray(A, dims=("flat_channel", "flat_vertex"))
    A = A.assign_coords({"parcel" : ("flat_vertex", np.concatenate((Adot_pruned.coords['parcel'].values, Adot_pruned.coords['parcel'].values)))})


    #
    # spatial basis functions
    #
    if SB:
        M = sbf.get_sensitivity_mask(Adot_pruned, sb_cfg['mask_threshold'])

        G = sbf.get_G_matrix(head, 
                                M,
                                sb_cfg['threshold_brain'], 
                                sb_cfg['threshold_scalp'], 
                                sb_cfg['sigma_brain'], 
                                sb_cfg['sigma_scalp']
                                )
        
        nbrain = Adot_pruned.is_brain.sum().values
        nscalp = Adot.shape[1] - nbrain 
        
        nkernels_brain = G['G_brain'].kernel.shape[0]
        nkernels_scalp = G['G_scalp'].kernel.shape[0]

        nkernels = nkernels_brain + nkernels_scalp

        H = np.zeros((2 * nchannel, 2 * nkernels))

        A_hbo_brain = A[:, :nbrain]
        A_hbr_brain = A[:, nbrain+nscalp:2*nbrain+nscalp]
        
        A_hbo_scalp = A[:, nbrain:nscalp+nbrain]
        A_hbr_scalp = A[:, 2*nbrain+nscalp:]
        
        H[:,:nkernels_brain] = A_hbo_brain.values @ G['G_brain'].values.T
        H[:, nkernels_brain+nkernels_scalp:2*nkernels_brain+nkernels_scalp] = A_hbr_brain.values @ G['G_brain'].values.T
        
        H[:,nkernels_brain:nkernels_brain+nkernels_scalp] = A_hbo_scalp.values @ G['G_scalp'].values.T
        H[:,2*nkernels_brain+nkernels_scalp:] = A_hbr_scalp.values @ G['G_scalp'].values.T

        H = xr.DataArray(H, dims=("channel", "kernel"))

        A = H.copy()


    #
    # Do the Image Reconstruction
    #

    # Ensure A is a numpy array
    A = np.array(A)
    B = np.sum((A ** 2), axis=0)
    b = B.max()

    for alpha_spatial in alpha_spatial_list:
                        
        if not BRAIN_ONLY and W is None:

            print( f'   Doing spatial regularization with alpha_spatial = {alpha_spatial}')
            # GET A_HAT
            lambda_spatial = alpha_spatial * b
            
            L = np.sqrt(B + lambda_spatial)
            Linv = 1/L
            Linv = np.diag(Linv)
            
            A_hat = A @ Linv
            
            #% GET W
            F = A_hat @ A_hat.T
            f = max(np.diag(F)) 
            print(f'   f = {f}')
            
            C = F #A @ (Linv ** 2) @ A.T
            D = Linv**2 @ A.T
            
        for alpha_meas in alpha_meas_list:
            
            print(f'   Doing image recon with alpha_meas = {alpha_meas}')
            

            if BRAIN_ONLY and W is None:
                Adot_stacked = xr.DataArray(A, dims=("flat_channel", "flat_vertex"))
                W = pseudo_inverse_stacked(Adot_stacked, alpha=alpha_meas)
                W = W.assign_coords({"chromo" : ("flat_vertex", ["HbO"]*nvertices  + ["HbR"]* nvertices)})
                W = W.set_xindex("chromo")
            elif W is None:
                if C_meas is None:
                    lambda_meas = alpha_meas * f 
                    W = D @ np.linalg.inv(C  + lambda_meas * np.eye(A.shape[0]) )
                else:
                    lambda_meas = alpha_meas * f
                    W = D @ np.linalg.inv(C + lambda_meas * C_meas)
            nvertices = W.shape[0]//2
        
            #% GENERATE IMAGES FOR DIFFERENT IMAGE PARAMETERS AND ALSO FOR THE FULL TIMESERIES
            X = W @ od_mag_pruned.values.T
            
            split = len(X)//2

            if BRAIN_ONLY:
                if len(hrf_od.dims) == 2: # not a time series else it is a time series
                    X = xr.DataArray(X, 
                                    dims = ('vertex'),
                                    coords = {'parcel':("vertex",np.concatenate((Adot.coords['parcel'].values, Adot.coords['parcel'].values)))},
                                    )
                else:
                    X = xr.DataArray(X, 
                                    dims = ('vertex', 'reltime'),
                                    coords = {'parcel':("vertex",np.concatenate((Adot.coords['parcel'].values, Adot.coords['parcel'].values))),
                                            'reltime': od_mag_pruned.reltime.values},
                                    )
                
            else:
                if SB:
                    X_hbo = X[:split]
                    X_hbr = X[split:]
                    sb_X_brain_hbo = X_hbo[:nkernels_brain]
                    sb_X_brain_hbr = X_hbr[:nkernels_brain]
                    
                    sb_X_scalp_hbo = X_hbo[nkernels_brain:]
                    sb_X_scalp_hbr = X_hbr[nkernels_brain:]
                    
                    #% PROJECT BACK TO SURFACE SPACE 
                    X_hbo_brain = G['G_brain'].values.T @ sb_X_brain_hbo
                    X_hbo_scalp = G['G_scalp'].values.T @ sb_X_scalp_hbo
                    
                    X_hbr_brain = G['G_brain'].values.T @ sb_X_brain_hbr
                    X_hbr_scalp = G['G_scalp'].values.T @ sb_X_scalp_hbr
                    
                    # concatenate them back together
                    if len(hrf_od.dims) == 2: # not a time series else it is a time series
                        X = np.stack([np.concatenate([X_hbo_brain, X_hbo_scalp]),np.concatenate([ X_hbr_brain, X_hbr_scalp])], axis=1)
                    else:
                        X = np.stack([np.vstack([X_hbo_brain, X_hbo_scalp]), np.vstack([X_hbr_brain, X_hbr_scalp])], axis =2)
                    
                else:
                    if len(hrf_od.dims) == 2: # not a time series else it is a time series
                        X = X.reshape([2, split]).T
                    else:
                        X = X.reshape([2, split, X.shape[1]])
                        X = X.transpose(1,2,0)
                    
                if len(hrf_od.dims) == 2: # not a time series else it is a time series
                    X = xr.DataArray(X, 
                                    dims = ('vertex', 'chromo'),
                                    coords = {'chromo': ['HbO', 'HbR'],
                                            'parcel': ('vertex',Adot.coords['parcel'].values),
                                            'is_brain':('vertex', Adot.coords['is_brain'].values)},
                                    )
                    X = X.set_xindex('parcel')
                else:                
                    X = xr.DataArray(X,
                                        dims = ('vertex', 'reltime', 'chromo'),
                                        coords = {'chromo': ['HbO', 'HbR'],
                                                'parcel': ('vertex',Adot.coords['parcel'].values),
                                                'is_brain':('vertex', Adot.coords['is_brain'].values),
                                                'reltime': od_mag_pruned.reltime.values},
                                        )
                    X = X.set_xindex("parcel")


            # save the results
            if file_save:
                if C_meas is None:
                    filepath = os.path.join(file_path, f'X_{trial_type_img}_alpha_spatial_{alpha_spatial:.0e}_alpha_meas_{alpha_meas:.0e}.pkl.gz')
                    print(f'   Saving to X_{trial_type_img}_alpha_spatial_{alpha_spatial:.0e}_alpha_meas_{alpha_meas:.0e}.pkl.gz')
                else:
                    filepath = os.path.join(file_path, f'X_{trial_type_img}_cov_alpha_spatial_{alpha_spatial:.0e}_alpha_meas_{alpha_meas:.0e}.pkl.gz')
                    print(f'   Saving to X_{trial_type_img}_cov_alpha_spatial_{alpha_spatial:.0e}_alpha_meas_{alpha_meas:.0e}.pkl.gz')
                file = gzip.GzipFile(filepath, 'wb')
                file.write(pickle.dumps([X, alpha_meas, alpha_spatial]))
                file.close()     

            # end loop over alpha_meas
        # end loop over alpha_spatial

    C_norm = C / f

    return X, W, C_norm



def plot_image_recon( X, head, flag_hbx='hbo_brain', view_position='superior' ):
    # pos_names = ['superior', 'left']

    #
    # Plot the results
    #
    cmap = p.get_cmap("jet", 256)
    new_cmap_colors = np.vstack((cmap(np.linspace(0, 1, 256))))
    custom_cmap = ListedColormap(new_cmap_colors)

    X_hbo_brain = X[X.is_brain.values, 0]
    X_hbr_brain = X[X.is_brain.values, 1]

    X_hbo_scalp = X[~X.is_brain.values, 0]
    X_hbr_scalp = X[~X.is_brain.values, 1]

    pos_names = ['superior', 'left', 'right', 'anterior', 'posterior']
    positions = [ 'xy',
        [(-547.808328867038, 96.55047760772226, 130.5057670434646),
        (96.98938441628879, 115.4642870176181, 165.68507873066255),
        (-0.05508155730503299, 0.021062586851317233, 0.9982596803838084)],
        [(547.808328867038, 96.55047760772226, 130.5057670434646),
        (96.98938441628879, 115.4642870176181, 165.68507873066255),
        (-0.05508155730503299, 0.021062586851317233, 0.9982596803838084)],
        [(100, 1000, 100),
        (96.98938441628879, 115.4642870176181, 165.68507873066255),
        (-0.05508155730503299, 0.021062586851317233, 0.9982596803838084)],
        [(100, -1000, 100),
        (96.98938441628879, 115.4642870176181, 165.68507873066255),
        (-0.05508155730503299, 0.021062586851317233, 0.9982596803838084)]
    ]
    clim=(-X_hbo_brain.max(), X_hbo_brain.max())

    # get index of pos_names that matches view_position
    idx = [i for i, s in enumerate(pos_names) if view_position in s]

    pos = positions[idx[0]]
    p0 = pv.Plotter(shape=(1,1), window_size = [600, 600])
#        p.add_text(f"Group average with alpha_meas = {alpha_meas} and alpha_spatial = {alpha_spatial}", position='upper_left', font_size=12, viewport=True)
    
    if flag_hbx == 'hbo_brain': # hbo brain 
        surf = cdc.VTKSurface.from_trimeshsurface(head.brain)
        surf = pv.wrap(surf.mesh)
        p0.subplot(0,0)
        p0.add_mesh(surf, scalars=X_hbo_brain, cmap=custom_cmap, clim=clim, show_scalar_bar=True )
        p0.camera_position = pos
        p0.add_text('HbO Brain', position='lower_left', font_size=10)

    elif flag_hbx == 'hbr_brain': # hbr brain
        surf = cdc.VTKSurface.from_trimeshsurface(head.brain)
        surf = pv.wrap(surf.mesh)   
        p0.subplot(0,0)      
        p0.add_mesh(surf, scalars=X_hbr_brain, cmap=custom_cmap, clim=clim, show_scalar_bar=True )
        p0.camera_position = pos
        p0.add_text('HbR Brain', position='lower_left', font_size=10)

    elif flag_hbx == 'hbo_scalp': # hbo scalp
        surf = cdc.VTKSurface.from_trimeshsurface(head.scalp)
        surf = pv.wrap(surf.mesh)
        p0.subplot(0,0)         
        p0.add_mesh(surf, scalars=X_hbo_scalp, cmap=custom_cmap, clim=clim, show_scalar_bar=True )
        p0.camera_position = pos
        p0.add_text('HbO Scalp', position='lower_left', font_size=10)

    elif flag_hbx == 'hbr_scalp': # hbr scalp
        surf = cdc.VTKSurface.from_trimeshsurface(head.scalp)
        surf = pv.wrap(surf.mesh)
        p0.subplot(0,0)         
        p0.add_mesh(surf, scalars=X_hbr_scalp, cmap=custom_cmap, clim=clim, show_scalar_bar=True )
        p0.camera_position = pos
        p0.add_text('HbR Scalp', position='lower_left', font_size=10)

    p0.show( )

    return p0