# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:07:11 2025

@author: be320
"""
import numpy as np
import pandas as pd
import tifffile as tiff
import os
import re
from scipy.ndimage import center_of_mass
import matplotlib.pyplot as plt 
from skimage.measure import regionprops
import scipy.ndimage as ndimage
import argparse

def separate_masks(seg):
    masks = []
    for i in range(1, seg.max() + 1):
        masks.append((seg == i).astype(int))
    return np.array(masks)


def t_course_from_roi(nd_stack, roi):
    if len(roi.shape) != 2:
        raise NotImplementedError("Only works for 2d ROIs")
    wh = np.where(roi)
    return np.mean(nd_stack[..., wh[0], wh[1]], -1)


def median_t_course_from_roi(nd_stack, roi):
    if len(roi.shape) != 2:
        raise NotImplementedError("Only works for 2d ROIs")
    wh = np.where(roi)
    return np.median(nd_stack[..., wh[0], wh[1]], -1)

def std_t_course_from_roi(nd_stack, roi):
    if len(roi.shape) != 2:
        raise NotImplementedError("Only works for 2d ROIs")
    wh = np.where(roi)

    return np.std(nd_stack[..., wh[0], wh[1]], -1)

def get_neighbour_map(masks):
    n_cells=np.unique(masks).shape[0]-1
    neighbour_map=np.zeros((n_cells,n_cells))
    for i in range(1,n_cells+1):
        neighbours=[]
        p_idx=np.argwhere(masks==i)
        for p in range(0,p_idx.shape[0]):
            px=p_idx[p,1]
            py=p_idx[p,0]
            for k in [-1,0,1]:
                for l in [-1,0,1]:
                    if k==0 and l==0:
                       continue
                    if px+k>=0 and px+k<=masks.shape[1]-1 and py+l>=0 and py+l<=masks.shape[0]-1:
                       if masks[py+l,px+k]!=i and masks[py+l,px+k]!=0:
                          neighbours.append(masks[py+l,px+k])
        nb=set(neighbours)
        for n in nb:
            neighbour_map[i-1,n-1]=1

    return neighbour_map 

def moving_average(signal, window_size):
    """
    Apply a symmetric moving average filter to a signal with adaptive edge handling.

    Parameters:
        signal (numpy array): The input signal.
        window_size (int): The number of points to average over (must be odd).

    Returns:
        numpy array: The filtered signal.
    """
    half_window = window_size // 2
    smoothed_signal = np.zeros_like(signal)

    for i in range(len(signal)):
        # Adjust window at the edges
        start = max(0, i - half_window)
        end = min(len(signal), i + half_window + 1)
        smoothed_signal[i] = np.mean(signal[start:end])  # Average over available points

    return smoothed_signal

def plot_masks(masks):
    # # plot masks with numbers
    plt.figure()
    plt.imshow(masks, cmap="jet")

    # Get unique mask labels (excluding background 0)
    mask_ids = np.unique(masks)
    mask_ids = mask_ids[mask_ids > 0]  # Remove background (0)


    props = regionprops(masks)

    for prop in props:
        y, x = prop.centroid  # Get the centroid of each mask
        plt.text(x, y, str(int(prop.label)), color="black", fontsize=5, ha="center", va="center")

    # Hide axes and add title
    plt.axis("off")
    plt.title("Cellpose Masks with IDs")
    plt.colorbar()
    plt.show()
    
    
def exclude_cells(individual_masks):
    #exclude cells outside a fitted elipse due to calcium FOV, circle if image 512x512
    h=individual_masks[0].shape[0]
    w=individual_masks[0].shape[1]

    # Create an empty array of zeros
    ellipse_array = np.zeros((h, w), dtype=np.uint8)

    # Create grid of coordinates
    y, x = np.ogrid[:h, :w]

    # Define ellipse center and radii
    center_y, center_x = h / 2, w / 2  # Center of the ellipse
    radius_y, radius_x = h / 2, w / 2  # Radii along y and x

    # Equation of an ellipse: (x - cx)^2 / rx^2 + (y - cy)^2 / ry^2 <= 1
    ellipse_mask = ((x - center_x) ** 2 / radius_x ** 2) + ((y - center_y) ** 2 / radius_y ** 2) <= 1

    # Set values inside the ellipse to 1
    ellipse_array[ellipse_mask] = 1
    plt.imshow(ellipse_array, cmap="gray")
    plt.axis("off")
    plt.show()

    selected_masks=[]
    renumbered_masks=np.zeros((h,w))
    count=0
    for mask in individual_masks:
        px_idx=np.argwhere(mask)
        if mask[px_idx[:, 0], px_idx[:, 1]].all() and  ellipse_array[px_idx[:, 0], px_idx[:, 1]].all():
           count=count+1
           selected_masks.append(mask)
           renumbered_masks[px_idx[:, 0], px_idx[:, 1]]=count
           
    renumbered_masks=renumbered_masks.astype(int)
    selected_masks=np.stack(selected_masks, axis=0)
    #np.save(selected_masks)
    return selected_masks, renumbered_masks 
#%% 
# Set up argument parser
parser = argparse.ArgumentParser(description="Align files in input folder and save in output folder")
parser.add_argument("input_folder", type=str, help="Path to the input folder")
parser.add_argument("output_folder", type=str, help="Path to the output folder")

# Parse arguments
args = parser.parse_args()
input_folder= args.input_folder
output_folder= args.output_folder

# Check if input folder exists
if not os.path.isdir(input_folder):
    print(f"Error: Input folder '{input_folder}' does not exist.")
    exit(1)

# Create output folder if it doesn't exist
current_directory = os.getcwd()
output_path=os.path.join(current_directory, output_folder)
os.makedirs(output_folder, exist_ok=True)
#%% for individual file        
# brightfield=np.load(r'D:\Ca_Voltage_Imaging_working\20250227_slip1_area1_brightfield_before_MMStack_Default_cut_seg.npy',allow_pickle=True)


# input_folder=r'R:\projects\thefarm2\live\Firefly\ca_voltage_imaging_working\20250227_aligned'
# file='20250227_slip1_area1_cytochalasin_1uM_1_MMStack_Default_combined_voltage_aligned.tif'

# file_path=os.path.join(input_folder, file)
# aligned_frames=tiff.imread(file_path)
#%% for files in folder
# input_folder=r'R:\projects\thefarm2\live\Firefly\ca_voltage_imaging_working\test_folder'
# output_path=r'R:\projects\thefarm2\live\Firefly\ca_voltage_imaging_working\test_folder'
for file in os.listdir(input_folder):
    if file.endswith("_ca_aligned.tif") or file.endswith("_voltage_aligned.tif"):
        print(f"Extracting traces from file: {file}")
        #load file
        file_path=os.path.join(input_folder,file)
        aligned_frames=tiff.imread(file_path)
        #load masks
        found_trial=re.search(r'(^[^_]*_[^_]*_[^_]*)(?=_)',file)
        trial=found_trial.group(1)
        brightfield_file=trial+'_brightfield_before_MMStack_Default_cut_seg.npy'
        brightfield=np.load(os.path.join(input_folder,brightfield_file),allow_pickle=True)
        
        
        #%% get individual masks 
        brightfield=brightfield.item()
        masks=brightfield['masks']
        individual_masks=separate_masks(masks) 
        #%% exculde masks outside ROI ellipse/circle
        individual_masks,masks=exclude_cells(individual_masks)       
        
        #%% initialise dataframe
        trial_length=aligned_frames.shape[0]
        n_cells=individual_masks.shape[0]
        
        # Create numeric labels starting from 1 for the remaining columns
        column_labels = ["cell_id", "cell_string", "trial_string","ca/voltage","exp_stage","cell_x","cell_y"] 
        numeric_columns = [str(i) for i in range(1, trial_length + 1)]
        
        # Combine both lists of column names
        all_columns = column_labels + numeric_columns
        
        # Initialize the empty DataFrame with the combined column names
        df = pd.DataFrame(columns=all_columns)
        
        #%% from masks - get cell id number, get cell id name, trial string, ca/voltage, exp_stage (celline_drug), x,y, optional - area, dimensions 
        cell_id = np.arange(1, n_cells+1)
        cell_id=cell_id.astype(str)
        df['cell_id']=cell_id
        
        trial_string=file.replace('_aligned.tif','')
        df['trial_string']=trial_string
        
        temp=np.char.add('_cell',cell_id)
        cell_string=np.char.add(trial_string, temp)
        df['cell_string']=cell_string
        
        if trial_string.endswith('_ca'):
            df['ca/voltage']='ca'
        else:
            df['ca/voltage']='v' 
        
        found_exp_stage= re.search(r'^[^_]*_[^_]*_[^_]*_(\S+)(?=_1_)', trial_string)
        exp_stage=found_exp_stage.group(1)
        df['exp_stage']=exp_stage
        
        for i in range(0,n_cells):
            coordinates=center_of_mass(individual_masks[i])
            df.loc[i,'cell_x']=coordinates[0] 
            df.loc[i,'cell_y']=coordinates[1]
            #1.04um/px
        #%% create neighbour map
        # nm=get_neighbour_map(masks)
        plot_masks(masks)
        
        #%% get mean, median and std traces from eroded masks and write to the .csv file 
        structure = np.zeros((3, 3, 3))
        structure[1, :, :] = 1
        eroded_masks = ndimage.binary_erosion(individual_masks, structure)
        tc_mean=[t_course_from_roi(aligned_frames, mask) for mask in eroded_masks]
        #tc_median=[median_t_course_from_roi(aligned_frames, mask) for mask in eroded_masks]
        #tc_std=[std_t_course_from_roi(aligned_frames, mask) for mask in eroded_masks]
        df.loc[:,'1':]=tc_mean
        #%%
        # Save to CSV
        df.to_csv(os.path.join(output_path,f"{trial_string}.csv"), index=False)
        np.save(os.path.join(output_path,brightfield_file.replace('.npy','_renumbered.npy')),masks)
        # plt.figure(2)
        # plt.plot(df.loc[226,'1':].values)
        # filtered=moving_average(df.loc[226,'1':].values,2000)
        # plt.figure(3)
        # plt.plot(filtered)
        # plt.show()
        # plt.figure(4)
        # plt.plot(df.loc[226,'1':].values/filtered)
        # plt.show()
