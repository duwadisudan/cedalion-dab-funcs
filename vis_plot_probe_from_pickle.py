import os
import gzip
import pickle
import tkinter as tk
from tkinter import filedialog

import cedalion.vis.plot_probe as vPlotProbe

#%%

flag_choose_file = 0

path2results = "/projectnb/nphfnirs/ns/Shannon/Data/Interactive_Walking_HD/derivatives/processed_data/"

fileName = "blockaverage_STS_OD.pkl.gz"

#
# does blockaverage.pkl.gz exist in the current directory?
if os.path.exists('blockaverage.pkl.gz'):
    with gzip.open('blockaverage.pkl.gz', 'rb') as f:
        blockaverage_all, geo2d, geo3d  = pickle.load(f)
    vPlotProbe.run_vis(blockaverage = blockaverage_all, geo2d = geo2d, geo3d = geo3d)

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
            blockaverage_all, geo2d, geo3d  = pickle.load(f)

        vPlotProbe.run_vis(blockaverage = blockaverage_all, geo2d = geo2d, geo3d = geo3d)
    else:
        print("No file selected.")


else:        # open file by name and loc
    filepath = os.path.join(path2results, fileName)
    with gzip.open(file_path, 'rb') as f:
        blockaverage_all, geo2d, geo3d  = pickle.load(f)

    vPlotProbe.run_vis(blockaverage = blockaverage_all, geo2d = geo2d, geo3d = geo3d)
        
        