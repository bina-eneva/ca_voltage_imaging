# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 12:07:49 2025

@author: be320
"""
import tifffile as tiff
import os
from PyQt5.QtWidgets import QFileDialog, QApplication
import numpy as np
 
def cut_brightfiled(image_path,rows,columns,output_path):
    image= tiff.imread(image_path)
    rows=np.load(rows,allow_pickle=True)
    columns=np.load(columns,allow_pickle=True)
    image_cut=image[rows.min():rows.max()+1,columns.min():columns.max()+1]
     
    with tiff.TiffWriter(os.path.join(output_path,os.path.basename(image_path.replace('.ome.tif','_cut.tif'))), bigtiff=True) as writer: 
          writer.write(image_cut) 




# Initialize QApplication (required for Qt)
app = QApplication([])

# Prompt the user to select a folder
image, _ = QFileDialog.getOpenFileName(None, "Choose brightfield")
rows, _ = QFileDialog.getOpenFileName(None, "Choose rows")
columns, _ = QFileDialog.getOpenFileName(None, "Choose columns")
output_path = QFileDialog.getExistingDirectory(None, "Choose an output folder")

# Check if a folder was selected
if image:
    print(f"Folder selected: {image}")
else:
    print("No folder selected.")
    
cut_brightfiled(image, rows, columns, output_path)
