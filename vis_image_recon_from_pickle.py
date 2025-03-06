import os
import gzip
import pickle
import tkinter as tk
from tkinter import filedialog
import numpy as np

import cedalion_parcellation.datasets as datasets
import cedalion_parcellation.imagereco.forward_model as fw
import cedalion.io as io
from cedalion import units
import cedalion.dataclasses as cdc 

import matplotlib.pyplot as p
import pyvista as pv
from matplotlib.colors import ListedColormap


flag_choose_file = 0

path2results = "/projectnb/nphfnirs/ns/Shannon/Data/Interactive_Walking_HD/derivatives/processed_data/"

fileName = "X_ST-o_cov_alpha_spatial_1e-01_alpha_meas_1e+00.pkl.gz"

#
#  Load the image recon results
#
if os.path.exists('image_results.pkl.gz'):
    with gzip.open('image_results.pkl.gz', 'rb') as f:
        X, alpha_meas, alpha_spatial  = pickle.load(f)

elif flag_choose_file:

    # set initialdir to current directory
    initialdir = os.getcwd()

    # Create a Tkinter root window (it will not be shown)
    root = tk.Tk()
    root.withdraw()

    # Open a file dialog to select a file
    file_path = filedialog.askopenfilename(
        initialdir=initialdir,
        title='Select a data file',
        filetypes=(('gZip files', '*.gz'), ('Pickle files', '*.pkl'), ('All files', '*.*'))
    )
    # Check if a file was selected
    if file_path:
        with gzip.open(file_path, 'rb') as f:
            X, alpha_meas, alpha_spatial  = pickle.load(f)
    else:
        print("No file selected.")
        quit()
        
else:        # open file by name and loc
    filepath = os.path.join(path2results, fileName)
    with gzip.open(filepath, 'rb') as f:
        X, alpha_meas, alpha_spatial  = pickle.load(f)



#
# Load the head model
#
#%% load in head model and sensitivity profile 

# root directory
#rootDir_data = '/Users/dboas/Documents/People/2024/BoasDavid/NN22_Data/Datasets/BallSqueezing_WHHD/'
probe_dir = "/projectnb/nphfnirs/ns/lcarlton/DATA/probes/NN22_WHHD/12NN/fw/"

head_model = 'ICBM152'

# Load the sensitivity profile
# with open(os.path.join(rootDir_data, 'derivatives', 'fw',  head_model, 'Adot_wParcels.pkl'), 'rb') as f:
#     Adot = pickle.load(f)
    
with open(probe_dir + head_model + '/Adot_wParcels.pkl', 'rb') as f:
    Adot = pickle.load(f) 
    
    
    
# recordings = io.read_snirf(probe_path + 'fullhead_56x144_System1.snirf')
# rec = recordings[0]
# geo3d = rec.geo3d
# amp = rec['amp']
# meas_list = rec._measurement_lists['amp']

#% GET EXTINCTION COEFFICIENTS
# E = nirs.get_extinction_coefficients("prahl", amp.wavelength)

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



print('plotting ...')

positions = [ 'xy',
    [(-547.808328867038, 96.55047760772226, 130.5057670434646),
    (96.98938441628879, 115.4642870176181, 165.68507873066255),
    (-0.05508155730503299, 0.021062586851317233, 0.9982596803838084)]
]
pos_names = ['superior'] #['superior', 'left']
clim=(-X_hbo_brain.max(), X_hbo_brain.max())

print('before the loop')
for name, camera_position in zip(pos_names, positions):

    print(f'plotting {name} ...')

    pos = positions[0]
#    p = pv.Plotter(shape=(2,2), off_screen=True, window_size = [1000, 600])
    p = pv.Plotter(shape=(2,2), window_size = [2400, 1600])
    p.add_text(f"Group average with alpha_meas = {alpha_meas} and alpha_spatial = {alpha_spatial}", position='upper_left', font_size=12, viewport=True)

    
    # hbo brain 
    surf = cdc.VTKSurface.from_trimeshsurface(head.brain)
    surf = pv.wrap(surf.mesh)
    p.subplot(0,0)
    p.add_mesh(surf, scalars=X_hbo_brain, cmap=custom_cmap, clim=clim, show_scalar_bar=True )
    p.camera_position = camera_position
    p.add_text('HbO Brain', position='lower_left', font_size=10)

    # hbr brain 
    surf = cdc.VTKSurface.from_trimeshsurface(head.brain)
    surf = pv.wrap(surf.mesh)   
    p.subplot(0,1)      
    p.add_mesh(surf, scalars=X_hbr_brain, cmap=custom_cmap, clim=clim, show_scalar_bar=True )
    p.camera_position = camera_position
    p.add_text('HbR Brain', position='lower_left', font_size=10)

    # # hbo scalp
    surf = cdc.VTKSurface.from_trimeshsurface(head.scalp)
    surf = pv.wrap(surf.mesh)
    p.subplot(1,0)         
    p.add_mesh(surf, scalars=X_hbo_scalp, cmap=custom_cmap, clim=clim, show_scalar_bar=True )
    p.camera_position = camera_position
    p.add_text('HbO Scalp', position='lower_left', font_size=10)

    # # hbr scalp
    surf = cdc.VTKSurface.from_trimeshsurface(head.scalp)
    surf = pv.wrap(surf.mesh)
    p.subplot(1,1)         
    p.add_mesh(surf, scalars=X_hbr_scalp, cmap=custom_cmap, clim=clim, show_scalar_bar=True )
    p.camera_position = camera_position
    p.add_text('HbR Scalp', position='lower_left', font_size=10)

    # link the axes
    p.link_views()

    # wait until the user closes the window
    print('go interactive ...')
    p.show(interactive=True)


print('done')

