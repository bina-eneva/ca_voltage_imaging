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
    """
    Combines multiple OME-TIFF files into a single OME-TIFF while preserving metadata.

    Parameters:
    - input_files: List of OME-TIFF file paths to be combined.
    - output_file: Path for the combined OME-TIFF output.
    """
    tif_files = [f for f in sorted(os.listdir(input_folder)) if f.endswith('.tif')]
    
    output_filename=tif_files[0].replace('.ome.tif', '_combined.ome.tif')
    output_path=os.path.join(output_folder, output_filename)
    
    combined_frames = []  # List to store image frames
    metadata = None  # Variable to store OME metadata
    total_frames = 0  # Counter for total number of frames

    # Step 1: Read images and extract metadata
    for file in tif_files:
        with tiff.TiffFile(os.path.join(input_folder, file)) as tif:
            images = tif.asarray()  # Read all frames as NumPy array
            combined_frames.append(images)  # Store frames
            total_frames += images.shape[0]  # Count frames

            # Extract metadata from the first file
            if metadata is None:
                metadata = tif.ome_metadata

    # Step 2: Stack images along the correct axis (Z or T)
    combined_stack = np.concatenate(combined_frames, axis=0)  # Adjust axis as needed

    # Step 3: Update metadata with new frame count
    updated_metadata = update_ome_metadata(metadata, total_frames)
    updated_metadata=tiff.xml2dict(updated_metadata)

    # Step 4: Save combined image stack as an OME-TIFF
    with tiff.TiffWriter(output_path, bigtiff=True) as writer: 
        writer.write(combined_stack)

    print(f"Combined OME-TIFF saved as: {output_filename}")


def deinterleave_tif(input_folder,filename, output_folder):
    input_file = os.path.join(input_folder, filename)
    with tiff.TiffFile(input_file) as tif:
        # Create output files
        output_file1 = filename.replace('.ome.tif', '_voltage.ome.tif')
        output_file2 = filename.replace('.ome.tif', '_ca.ome.tif')
        
        output_path1=os.path.join(output_folder, output_file1)
        output_path2=os.path.join(output_folder, output_file2)

        with tiff.TiffWriter(output_path1) as tif1, tiff.TiffWriter(output_path2) as tif2:
            for i, page in enumerate(tif.pages):  # Process pages inside the open block
                img = page.asarray()  # Read frame while file is open
                if i % 2 == 0:
                    tif1.write(img)
                else:
                    tif2.write(img)

    print(f"Processed: {input_file} → {output_file1}, {output_file2}")

#specify folders
# root_directory = "R:/projects/thefarm2/live/Firefly/Calcium_Voltage_Imaging/MDA_MB_468/20250127/slip1" 
root_directory = r"R:\projects\thefarm2\live\Firefly\Calcium_Voltage_Imaging\MDA_MB_468\20250227\slip3\area1\20250227_slip3_area1_cytochalasin_1uM_60min_1"
output_folder=r"D:/Ca_Voltage_Imaging_working/20250127"
os.makedirs(output_folder, exist_ok=True)

last_level_folders = get_last_level_folders(root_directory)
folders = [ s for s in last_level_folders if not (s.startswith('20250208') or s.startswith('20250209') or s.endswith('after') or s.endswith('before') or s.endswith('brightfield'))]


# for folder in folders:
#     print(f"Working on folder {folder}")
#     combine_ome_tiffs(folder,output_folder) 
    
for folder in folders:
    for filename in os.listdir(folder):
         if filename.lower().endswith('default.ome.tif'):  # Only process .tif files
            deinterleave_tif(folder, filename,output_folder)

 