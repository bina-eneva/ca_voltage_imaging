# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 14:28:27 2025

@author: binae
"""
import os
import numpy as np
import tifffile as tiff
#import functions as canf
import cv2
def get_last_level_folders(root_dir):
    last_level_folders = set()

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if filenames and any(f.endswith('.tif') for f in filenames):  # Check if files exist and include .tif
            if not dirnames:  # Ensure this folder has no subdirectories (i.e., true last level)
                last_level_folders.add(dirpath)  # Store the folder name

    return sorted(last_level_folders)


def combine_and_deinterleave_tiffs(input_folder, output_folder):
    
    tif_files = [f for f in sorted(os.listdir(input_folder)) if f.endswith('.tif')]
    
    # generate output files and paths for the two files
    output_filename_v=tif_files[0].replace('.ome.tif', '_voltage.tif')
    output_path_v=os.path.join(output_folder, output_filename_v)
    output_filename_ca=tif_files[0].replace('.ome.tif', '_ca.tif')
    output_path_ca=os.path.join(output_folder, output_filename_ca)
    
    combined_frames = []  # List to store image frames


    for file in tif_files:
        image_stack=tiff.imread(os.path.join(input_folder, file))
        combined_frames.append(image_stack)  # Store each frame

    combined_stack = np.stack(combined_frames, axis=0) 
    
    #deinterleave
    combined_voltage = combined_stack[::2, :, :]  # Take every even z
    combined_calcium= combined_stack[1::2, :, :]
    

    with tiff.TiffWriter(output_path_v, bigtiff=True) as writer: 
         writer.write(combined_voltage) # Add metadata=updated_metadata if you need it
    
    with tiff.TiffWriter(output_path_ca, bigtiff=True) as writer: 
         writer.write(combined_calcium) # Add metadata=updated_metadata if you need it
         
    print(f"Combined voltage shape: {combined_voltage.shape}")
    print(f"Combined file saved as: {output_filename_v}")
    print(f"Combined calcium shape: {combined_calcium.shape}")
    print(f"Combined file saved as: {output_filename_ca}")
    return combined_voltage,combined_calcium

def save_to_avi(stack, filename, save_dir, fps=1000):
    """
    Save a 3D NumPy array as an AVI file.
    
    :param stack: 3D NumPy array (Z, H, W)
    :param filename: Name of the output AVI file
    :param save_dir: Directory where the file will be saved
    :param fps: Frames per second (default 1000)
    """
    filepath = str(Path(save_dir) / filename)
    
    # Get image size
    height, width = stack.shape[1], stack.shape[2]
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Use MJPG for compatibility
    out = cv2.VideoWriter(filepath, fourcc, fps, (width, height), isColor=False)
    
    # Write each frame
    for frame in stack:
        out.write(cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype('uint8'))  # Normalize to 8-bit
    
    out.release()  # Close the video file
    print(f"✅ Saved {filename} at {fps} FPS in {save_dir}")

def enhance_contrast(image, clip_limit=3.5, grid_size=(8,8)):
    """
    Equivalent to ImageJ's Enhance Contrast with saturation adjustment.
    Uses CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    :param image: Input grayscale image (NumPy array)
    :param clip_limit: Controls contrast enhancement strength (default 3.5)
    :param grid_size: Defines local contrast adjustment regions (default 8x8)
    :return: Contrast-enhanced image
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    return clahe.apply(image)


def adjust_brightness(image, clip_limit=2.0, gamma=1.0):
    """
    Adjust brightness using CLAHE and Gamma Correction (less aggressive).
    """
    # Ensure image is grayscale and uint8
    if len(image.shape) != 2:
        print(f"❌ Error: Expected 2D grayscale image, got shape {image.shape}")
        return image
    
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Apply CLAHE (weaker)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    enhanced = clahe.apply(image)

    # Gamma Correction
    gamma_table = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in np.arange(256)], dtype=np.uint8)
    brightened = cv2.LUT(enhanced, gamma_table)

    return brightened

def enhance_sharpness(image, alpha=1.0, apply_denoise=True):
    """
    Enhances sharpness with a gentler filter, and optionally reduces noise.
    """
    # Define a softer sharpening kernel
    sharpening_kernel = np.array([
        [ 0, -1,  0],
        [-1,  5 + alpha, -1],
        [ 0, -1,  0]
    ], dtype=np.float32)

    sharpened = cv2.filter2D(image, -1, sharpening_kernel)

    # Apply denoising filter to reduce noise after sharpening
    if apply_denoise:
        sharpened = cv2.GaussianBlur(sharpened, (3, 3), 0.5)  # Smooth out noise

    return sharpened

# from PyQt5.QtWidgets import QFileDialog, QApplication
# app = QApplication([])
# input_folder= QFileDialog.getExistingDirectory(None, "Choose folder")
# output_folder = r'R:\projects\thefarm2\live\Firefly\ca_voltage_imaging_working\tests'

root_directory = '$RDS\projects\thefarm2\live\Firefly\Calcium_Voltage_Imaging\MDA_MB_468\20250307\slip2\area1'
output_folder='$RDS/projects/thefarm2/live/Firefly/ca_voltage_imaging_working/tests'
input_folder = get_last_level_folders(root_directory)

voltage_channel, calcium_channel=combine_and_deinterleave_tiffs(input_folder, output_folder)
     
voltage_channel = np.array([adjust_brightness(slice) for slice in voltage_channel])
calcium_channel = np.array([adjust_brightness(slice) for slice in calcium_channel])
#voltage_channel = np.array([canf.enhance_sharpness(slice) for slice in voltage_channel])
#calcium_channel = np.array([canf.enhance_sharpness(slice) for slice in calcium_channel])
#voltage_channel= np.array([canf.enhance_contrast(slice) for slice in voltage_channel])
#calcium_channel = np.array([canf.enhance_contrast(slice) for slice in calcium_channel])

save_to_avi(voltage_channel, "enhanced_voltage_video.avi", output_folder)
save_to_avi(calcium_channel, "enhanced_calcium_video.avi", output_folder)