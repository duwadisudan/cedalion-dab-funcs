import os
import gzip
import pickle
import tkinter as tk
from tkinter import filedialog

import cedalion.vis.time_series as vTS


# does blockaverage.pkl.gz exist in the current directory?
if os.path.exists('rec.pkl.gz'):
    with gzip.open('rec.pkl.gz', 'rb') as f:
        rec  = pickle.load(f)
    vTS.run_vis(rec)

else:

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
            rec = pickle.load(f)

        vTS.run_vis(rec)
    else:
        print("No file selected.")