# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 13:11:02 2025

@author: binae
"""

import napari
import tifffile
import dask.array as da

from PyQt5.QtWidgets import QFileDialog, QApplication
app = QApplication([])
input_file,_= QFileDialog.getOpenFileName(None, "Choose file")

# You can use memmap for faster access if file is large:
try:
    arr = tifffile.memmap(input_file)

except Exception:
    arr = tifffile.imread(input_file, aszarr=True)
    arr = da.from_zarr(arr)


napari.view_image(arr) 