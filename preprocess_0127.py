# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 17:14:10 2025

@author: binae
"""
import os
import numpy as np
import tifffile as tiff
import matplotlib.pyplot as plt
#import functions as canf
import cv2
from cellpose import models
from cellpose.io import imread
import SimpleITK as sitk
import argparse
import cv2

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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


#     for file in tif_files:
#         image_stack=tiff.imread(os.path.join(input_folder, file))
#         combined_frames.append(image_stack)  # Store each frame

    print(tif_files[0])
    image_stack=tiff.imread(os.path.join(input_folder, tif_files[0]))
    print(f"Combined shape: {image_stack.shape}")
#     plt.imshow(image_stack[18000],cmap='gray')
#     combined_frames.append(image_stack)  # Store each frame


#     combined_stack = np.concatenate(combined_frames, axis=0) 
    
    #deinterleave
    combined_voltage = image_stack[::2, :, :]  # Take every even z
    combined_calcium= image_stack[1::2, :, :]
    

    with tiff.TiffWriter(output_path_v, bigtiff=True) as writer: 
         writer.write(combined_voltage) # Add metadata=updated_metadata if you need it
    
    with tiff.TiffWriter(output_path_ca, bigtiff=True) as writer: 
         writer.write(combined_calcium) # Add metadata=updated_metadata if you need it
         
    print(f"Combined voltage shape: {combined_voltage.shape}")
    print(f"Combined file saved as: {output_filename_v}")
    print(f"Combined calcium shape: {combined_calcium.shape}")
    print(f"Combined file saved as: {output_filename_ca}")
    return combined_voltage,combined_calcium
#%% align functions 
def get_every_100th_image(image_stack):
    
    frames_to_align=tiff.imread(image_stack)
    frames_to_align=frames_to_align[::100]
    output_file_name=image_stack.replace('voltage.tif','voltage_frames_to_align.tif')
    
    with tiff.TiffWriter(output_file_name) as tif:
         tif.write(frames_to_align)
         
    # with tiff.TiffWriter(r"D:/Ca_Voltage_Imaging_working/20250127/frame1.tif") as tif:
    #      tif.write(frames_to_align[0])
         
    # with tiff.TiffWriter(r"D:/Ca_Voltage_Imaging_working/20250127/frame25.tif") as tif:
    #      tif.write(frames_to_align[24])
         
    # with tiff.TiffWriter(r"D:/Ca_Voltage_Imaging_working/20250127/frame50.tif") as tif:
    #      tif.write(frames_to_align[49])
         
         
    print(f"Returning {frames_to_align.shape[0]} frames to align")
    
    return output_file_name

def segment_voltage_frames(frames_to_align,output_path):
    model = models.Cellpose(gpu=False, model_type='cyto')


    imgs = imread(frames_to_align)
    masks, flows, styles, diams = model.eval(imgs, diameter=30, channels=[0,0],
                                             flow_threshold=0.4, cellprob_threshold=0.5, do_3D=False)

    frames_to_align_binary_masks=frames_to_align.replace('frames_to_align.tif','binary_masks.npy')
    # If the mask values are not binary, convert to binary
    binary_masks = (masks> 0).astype(np.uint8)  # 1 for cell, 0 for background
    np.save(os.path.join(output_path,os.path.basename(frames_to_align_binary_masks)), binary_masks)
    print('Returning binary masks')
    return binary_masks

def get_transformation_parameters(fixed_img, moving_img):
    # Step 1: Load the binary images from .npy files
    # masks= np.load(r"R:\projects\thefarm2\live\Firefly\ca_voltage_imaging_working\20250127\binary_masks.npy")
    
    # Step 2: Convert NumPy arrays to SimpleITK images
    fixed_image = sitk.GetImageFromArray(fixed_img)
    moving_image = sitk.GetImageFromArray(moving_img)
    # fixed_image = sitk.SignedMaurerDistanceMap(fixed_image, insideIsPositive=True, squaredDistance=False)
    # moving_image = sitk.SignedMaurerDistanceMap(moving_image, insideIsPositive=True, squaredDistance=False)
    
    # Ensure the images have the correct datatype (UInt8 for binary images)
    fixed_image = sitk.Cast(fixed_image, sitk.sitkFloat32)
    moving_image = sitk.Cast(moving_image, sitk.sitkFloat32)
    
    
    # Step 3: Initialize a rigid transform
    # initial_transform = sitk.Euler2DTransform()  # Use Euler2DTransform() for 2D images
    initial_transform = sitk.TranslationTransform(fixed_image.GetDimension())
    
    # Step 4: Set up the registration method
    registration = sitk.ImageRegistrationMethod()
    
    # For binary images, Mean Squares Difference is commonly used:
    # registration.SetMetricAsMeanSquares()  # For binary images of the same modality
    # registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricAsCorrelation() 
    # Use linear interpolation since it's binary
    registration.SetInterpolator(sitk.sitkLinear)
    
    # Set the optimizer
    # registration.SetOptimizerAsGradientDescent(learningRate=0.1, numberOfIterations=1000)
    registration.SetOptimizerAsRegularStepGradientDescent(
    learningRate=0.1,  # Adjust if needed
    minStep=0.001,  # Controls when to stop refining
    numberOfIterations=1000)
    
    registration.SetInitialTransform(initial_transform, inPlace=False)
    
    # Step 5: Perform the registration
    final_transform = registration.Execute(fixed_image, moving_image)
    
    # Step 6: Apply the final transformation to the moving image
    resampled_image = sitk.Resample(moving_image, fixed_image, final_transform, sitk.sitkLinear, 26 , moving_image.GetPixelID())
    parameters=final_transform.GetParameters()


    
    # Step 8: Print the transformation parameters
    # print("Final Parameters:", final_transform.GetParameters())
    
    # Convert SimpleITK images to NumPy for visualization
    fixed_image= sitk.GetArrayFromImage(fixed_image)
    resampled_moving = sitk.GetArrayFromImage(resampled_image)
    
    # Step 9: Visualization
    
    # # Plotting the images to inspect the registration visually
    # fig, axes = plt.subplots(1, 1, figsize=(8, 8))
    
    # # Display the fixed image with a colormap (e.g., gray)
    # axes.imshow(fixed_image, cmap='gray')
    
    # # Overlay the registered image with some transparency (alpha)
    # axes.imshow(resampled_moving, cmap='jet', alpha=0.5)  # 'jet' colormap is used for the overlay
    
    # # Add title and remove axis labels
    # axes.set_title('Fixed Image and Registered Image Overlay')
    # axes.axis('off')
    
    # plt.tight_layout()
    # plt.show()
    #print('Returning translation parameters')
    
    return parameters

 

def get_all_translation_vectors(binary_masks,image_stack,output_path):
    n_frames=np.size(binary_masks,0)
    translation=np.zeros([n_frames,2])
    #for every i=length and i-1 find parameters
    for i in range(0,(n_frames-1)):
        translation[i,:]=get_transformation_parameters(binary_masks[i],binary_masks[i+1])
    np.save(os.path.join(output_path,os.path.basename(image_stack.replace('.tif','_translation.npy'))), translation)
    print('Returning translation vector')
    return translation

def align_frames(image_stack, translation,output_path):
    resampled_stack=[]
    with tiff.TiffFile(image_stack) as tif:
        first_frame = tif.pages[0].asarray()
        resampled_stack.append(first_frame)
        for i, page in enumerate(tif.pages[1:], start=1):  # Process pages inside the open block
            img = page.asarray()  
            quotient=i//100 #whole part
            fraction=i/100-quotient #fractional part
#             if i<=100:
#                 translation_x=np.round(fraction*translation[quotient,0])
#                 translation_y=np.round(fraction*translation[quotient,1])
#             else:
#                 translation_x=np.round(np.sum(translation[0:quotient,0])+fraction*translation[quotient,0]) #sum is start:stop excluding the stop
#                 translation_y=np.round(np.sum(translation[0:quotient,1])+fraction*translation[quotient,1]) #sum is start:stop excluding the stop
            if i<=100:
                translation_x=fraction*translation[quotient,0]
                translation_y=fraction*translation[quotient,1]
            else:
                translation_x=np.sum(translation[0:quotient,0])+fraction*translation[quotient,0] #sum is start:stop excluding the stop
                translation_y=np.sum(translation[0:quotient,1])+fraction*translation[quotient,1] #sum is start:stop excluding the stop 
            
            
            #transform image
            translation_transform = sitk.TranslationTransform(2) # Set translation parameters
            translation_transform.SetOffset([translation_x, translation_y])

            # Apply the translation to the moving image
            resampled_image = sitk.Resample(sitk.GetImageFromArray(img), sitk.GetImageFromArray(first_frame), translation_transform, sitk.sitkLinear, 0)

            # # Ensure the resampled image is of integer type (for binary images, UInt8)
            # resampled_image = sitk.Cast(resampled_image, sitk.sitkUInt16)

            resampled_image = sitk.GetArrayFromImage(resampled_image)
            resampled_stack.append(resampled_image)
           
    resampled_stack = np.stack(resampled_stack, axis=0) 
    to_cut=resampled_stack.all(0)
    print(f"to cut {to_cut.shape}")
    columns=np.where(to_cut.any(0))[0]
    rows=np.where(to_cut.any(1))[0]
    resampled_stack_cut=resampled_stack[:,rows.min():rows.max()+1,columns.min():columns.max()+1]
    if image_stack.endswith('voltage.tif'):
       np.save(os.path.join(output_path,os.path.basename(image_stack.replace('_voltage.tif','_rows.npy'))), rows)
       np.save(os.path.join(output_path,os.path.basename(image_stack.replace('_voltage.tif','_columns.npy'))),columns)
    with tiff.TiffWriter(os.path.join(output_path,os.path.basename(image_stack.replace('.tif','_aligned.tif'))), bigtiff=True) as writer: 
         writer.write(resampled_stack_cut) 
    print('Returning aligned stack')
    return resampled_stack_cut
#%% video functions 
def save_to_avi(stack, filename, save_dir, fps=1000):
    """
    Save a 3D NumPy array as an AVI file.
    
    :param stack: 3D NumPy array (Z, H, W)
    :param filename: Name of the output AVI file
    :param save_dir: Directory where the file will be saved
    :param fps: Frames per second (default 1000)
    """
    filepath = os.path.join(save_dir,filename)
    
    # Get image size
    height, width = stack.shape[1], stack.shape[2]
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use MJPG for compatibility
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

#%%deinterleave

input_folder= r'/rds/general/user/be320/projects/thefarm2/live/Firefly/Calcium_Voltage_Imaging/MDA_MB_468/20250127'
output_folder=r'/rds/general/user/be320/projects/thefarm2/live/Firefly/ca_voltage_imaging_working/20250127_processed'
os.makedirs(output_folder, exist_ok=True)

last_level_folders = get_last_level_folders(input_folder)
folders = [ s for s in last_level_folders if not (s.startswith('20250208') or s.startswith('20250209') or s.endswith('after') or s.endswith('before') or s.endswith('brightfield') or s.endswith('post'))]
print(folders)
for folder in folders:
    voltage_channel, calcium_channel=combine_and_deinterleave_tiffs(folder, output_folder)
         
    voltage_channel = np.array([adjust_brightness(slice) for slice in voltage_channel])
    calcium_channel = np.array([adjust_brightness(slice) for slice in calcium_channel])
    voltage_channel = np.array([enhance_sharpness(slice) for slice in voltage_channel])
    #calcium_channel = np.array([enhance_sharpness(slice) for slice in calcium_channel])
    #voltage_channel= np.array([enhance_contrast(slice) for slice in voltage_channel])
    #calcium_channel = np.array([enhance_contrast(slice) for slice in calcium_channel])
    
    save_to_avi(voltage_channel, folder.replace('_1','_1_MMStack_Default_voltage.mp4'), output_folder,fps=1000)
    save_to_avi(calcium_channel, folder.replace('_1','_1_MMStack_Defaul_ca.mp4'), output_folder,fps=1000)

#%% align 
for file in os.listdir(output_folder):
    if file.endswith("_voltage.tif"):
        print(f"Processing voltage file: {file}")
        image_stack=os.path.join(output_folder,file)
        #%% get frames to align    
        frames_to_align=get_every_100th_image(image_stack)
        #%% segment to align
        binary_masks=segment_voltage_frames(frames_to_align,output_folder)
        #%% masks registration 
        translation =get_all_translation_vectors(binary_masks,image_stack,output_folder)
        # translation = np.load('20250307_slip2_area1_Thapsigargin_1uM_1_MMStack_Default_translation.npy')
        voltage_channel=align_frames(image_stack,translation,output_folder)
        calcium_sister_file=file.replace('voltage.tif','ca.tif')
        image_stack=os.path.join(output_folder,calcium_sister_file)
        print("Processing calcium sister file")
        calcium_channel=align_frames(image_stack,translation,output_folder)
        voltage_channel = np.array([adjust_brightness(slice) for slice in voltage_channel])
        calcium_channel = np.array([adjust_brightness(slice) for slice in calcium_channel])
        voltage_channel = np.array([enhance_sharpness(slice) for slice in voltage_channel])
        #calcium_channel = np.array([enhance_sharpness(slice) for slice in calcium_channel])
#         voltage_channel= np.array([enhance_contrast(slice) for slice in voltage_channel])
#         calcium_channel = np.array([enhance_contrast(slice) for slice in calcium_channel])
    
        save_to_avi(voltage_channel, file.replace('.tif','_aligned.mp4'), output_folder,fps=1000)
        save_to_avi(calcium_channel, calcium_sister_file.replace('.tif','_aligned.mp4'), output_folder,fps=1000)