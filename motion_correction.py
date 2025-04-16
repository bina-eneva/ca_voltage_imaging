# -*- coding: utf-8 -*-
"""
Created on Wed Mar  5 11:26:51 2025

@author: be320
"""
import os
import numpy as np
import tifffile as tiff
from cellpose import models
from cellpose.io import imread
import matplotlib.pyplot as plt
import SimpleITK as sitk

#%% Get frames to align
# image_stack=r"R:\projects\thefarm2\live\Firefly\ca_voltage_imaging_working\20250127\20250127_slip4_area1_exp3ms_4AP_5mM_washout_1_MMStack_Default_combined_voltage.tif"

# def get_every_100th_image(image_stack):
#     frames_to_align=[]
#     with tiff.TiffFile(image_stack) as tif:
#         for i, page in enumerate(tif.pages):  # Process pages inside the open block
#             img = page.asarray()  
#             if i % 100 == 0:
#                     frames_to_align.append(img)
                
           
                   
#     frames_to_align=np.stack(frames_to_align, axis=0)
#     with tiff.TiffWriter(r"R:\projects\thefarm2\live\Firefly\ca_voltage_imaging_working\20250127\20250127_slip4_area1_4AP_washout_frames_to_align.tif") as tif:
#          tif.write(frames_to_align)
         
#     # with tiff.TiffWriter(r"D:/Ca_Voltage_Imaging_working/20250127/frame1.tif") as tif:
#     #      tif.write(frames_to_align[0])
         
#     # with tiff.TiffWriter(r"D:/Ca_Voltage_Imaging_working/20250127/frame25.tif") as tif:
#     #      tif.write(frames_to_align[24])
         
#     # with tiff.TiffWriter(r"D:/Ca_Voltage_Imaging_working/20250127/frame50.tif") as tif:
#     #      tif.write(frames_to_align[49])
         
         
#     print(f"Returning {frames_to_align.shape[0]} frames to align")
    
#     return frames_to_align
 
# frames= get_every_100th_image(image_stack)

#%% Segmentation
# model = models.Cellpose(gpu=False, model_type='cyto')


# imgs = imread(r"R:\projects\thefarm2\live\Firefly\ca_voltage_imaging_working\20250127\20250127_slip4_area1_4AP_washout_frames_to_align.tif")
# masks, flows, styles, diams = model.eval(imgs, diameter=30, channels=[0,0],
#                                          flow_threshold=0.4, cellprob_threshold=0.5, do_3D=False)


# # If the mask values are not binary, convert to binary
# binary_masks = (masks> 0).astype(np.uint8)  # 1 for cell, 0 for background

# # Display the binary mask: 1 is white, 0 is black
# plt.imshow(binary_masks[0], cmap='gray')  # 'gray' colormap for binary display
# plt.title("First Mask")
# plt.axis('off')  # Hide axis for better visualization
# plt.show()

#%% Registration
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
    print("Final Parameters:", final_transform.GetParameters())
    
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
    
    return parameters

binary_masks= np.load(r"R:\projects\thefarm2\live\Firefly\ca_voltage_imaging_working\20250127\binary_masks.npy")
#initialise parameters 
n_frames=np.size(binary_masks,0)
translation=np.zeros([n_frames,2])
#for every i=length and i-1 find parameters
for i in range(0,(n_frames-1)):
    translation[i,:]=get_transformation_parameters(binary_masks[i],binary_masks[i+1])
    
#transform each i times using 0 to length-i parameters
#plot first and last image overlayed 
translation_x=np.round(np.sum(translation[:100,0]))
translation_y=np.round(np.sum(translation[:100,1]))

translation_transform = sitk.TranslationTransform(2) # Set translation parameters
translation_transform.SetOffset([translation_x, translation_y])

# Apply the translation to the moving image
resampled_image = sitk.Resample(sitk.GetImageFromArray(binary_masks[100]), sitk.GetImageFromArray(binary_masks[0]), translation_transform, sitk.sitkLinear, 0)

# Ensure the resampled image is of integer type (for binary images, UInt8)
resampled_image = sitk.Cast(resampled_image, sitk.sitkUInt8)

resampled_image = sitk.GetArrayFromImage(resampled_image)

#plot overlayed 
fig, axes = plt.subplots(1, 1, figsize=(8, 8))
axes.imshow(binary_masks[0], cmap='gray')
axes.imshow(resampled_image, cmap='jet', alpha=0.5)  
axes.set_title('Fixed Image and Registered Image Overlay')
axes.axis('off')
plt.tight_layout()


