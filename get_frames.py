# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 12:27:30 2025

@author: be320
"""

import os
import numpy as np
import tifffile as tiff
import xml.etree.ElementTree as ET

def get_last_level_folders(root_dir):
    last_level_folders = set()

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if filenames and any(f.endswith('.tif') for f in filenames):  # Check if files exist and include .tif
            if not dirnames:  # Ensure this folder has no subdirectories (i.e., true last level)
                last_level_folders.add(dirpath)  # Store the folder name

    return sorted(last_level_folders)



def update_ome_metadata(metadata, new_size_t):
    """
    Updates the OME-XML metadata to reflect the new number of frames (SizeT or SizeZ).
    
    Parameters:
    - metadata: Original OME-XML metadata string.
    - new_size_t: New total number of frames.
    
    Returns:
    - Updated OME-XML metadata string.
    """
    # Parse the OME-XML metadata
    root = ET.fromstring(metadata)

    # OME XML Namespace (used to find correct elements)
    namespace = "{http://www.openmicroscopy.org/Schemas/OME/2016-06}"
    
    # Find the <Pixels> tag that contains the SizeT or SizeZ attributes
    pixels = root.find(f".//{namespace}Pixels")
    
    if pixels is not None:
        # Update SizeT (timepoints) or SizeZ (z-slices) as needed
        if "SizeT" in pixels.attrib:
            pixels.set("SizeT", str(new_size_t))
        elif "SizeZ" in pixels.attrib:  # Some OME-TIFFs store frames as Z-slices
            pixels.set("SizeZ", str(new_size_t))

    # Convert back to string
    updated_metadata = ET.tostring(root, encoding="unicode")
    return updated_metadata

def combine_ome_tiffs(input_folder, output_folder):
  
    tif_files = [f for f in sorted(os.listdir(input_folder)) if f.endswith('.tif')]
    
    output_filename=tif_files[0].replace('.ome.tif', '_combined.tif')
    output_path=os.path.join(output_folder, output_filename)
    
    combined_frames = []  # List to store image frames
    # metadata = None  # Variable to store OME metadata
    # total_frames = 0  # Counter for total number of frames

    for file in tif_files:
        with tiff.TiffFile(os.path.join(input_folder, file)) as tif:
            for page in tif.pages: 
                combined_frames.append(page.asarray())  # Store each frame
            
            # Extract metadata from the first file
            # if metadata is None:
            #     metadata = tif.ome_metadata

    combined_stack = np.stack(combined_frames, axis=0) 
    # total_frames = combined_frames.shape[0]  # Count frames
    # updated_metadata = update_ome_metadata(metadata, total_frames)
    # updated_metadata=tiff.xml2dict(updated_metadata)

    with tiff.TiffWriter(output_path, bigtiff=True) as writer: 
         writer.write(combined_stack) # Add metadata=updated_metadata if you need it
    
    print(f"Combined stack shape: {combined_stack.shape}")
    print(f"Combined file saved as: {output_filename}")
    


def deinterleave_tif(input_file):
    
    with tiff.TiffFile(input_file) as tif:
        # Create output files
        output_file1 = input_file.replace('.tif', '_voltage.tif')
        output_file2 = input_file.replace('.tif', '_ca.tif')
        
        voltage=[];
        calcium=[];
       
        for i, page in enumerate(tif.pages):  # Process pages inside the open block
            img = page.asarray()  
            if i % 2 == 0:
                    voltage.append(img)
                
            else:
                    calcium.append(img)
                   
        voltage=np.stack(voltage, axis=0)
        calcium=np.stack(calcium, axis=0) 
        
    with tiff.TiffWriter(output_file1, bigtiff=True) as tif1, tiff.TiffWriter(output_file2, bigtiff=True) as tif2:
          tif1.write(voltage)
          tif2.write(calcium)

    print(f"Processed: {os.path.basename(input_file)} → {os.path.basename(output_file1)} and {os.path.basename(output_file2)}")

# specify folders
root_directory = "R:/projects/thefarm2/live/Firefly/Calcium_Voltage_Imaging/MDA_MB_468/20250304/slip4" 
output_folder=r"R:\projects\thefarm2\live\Firefly\ca_voltage_imaging_working\20250304"
os.makedirs(output_folder, exist_ok=True)

last_level_folders = get_last_level_folders(root_directory)
folders = [ s for s in last_level_folders if not (s.startswith('20250208') or s.startswith('20250209') or s.endswith('after') or s.endswith('before') or s.endswith('brightfield'))]


for folder in folders: 
    print(f"Working on folder {folder}")
    combine_ome_tiffs(folder,output_folder) 
    
for filename in os.listdir(output_folder): 
    if filename.lower().endswith('combined.tif'):  # Only process .tif files
        input_file = os.path.join(output_folder, filename)
        deinterleave_tif(input_file)

 