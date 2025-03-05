# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 11:26:51 2025

@author: be320
"""
import os
import numpy as np
import tifffile as tiff
image_stack=r"D:/Ca_Voltage_Imaging_working/20250127/20250127_slip5_area1_exp3ms_1_MMStack_Default_combined_voltage.tif"
def get_every_100th_image(image_stack):
    frames_to_align=[]
    with tiff.TiffFile(image_stack) as tif:
        for i, page in enumerate(tif.pages):  # Process pages inside the open block
            img = page.asarray()  
            if i % 100 == 0:
                    frames_to_align.append(img)
                
           
                   
    frames_to_align=np.stack(frames_to_align, axis=0)
    print(f"Returning {frames_to_align.shape[0]} frames to align")
    
    return frames_to_align
 
frames= get_every_100th_image(image_stack)
