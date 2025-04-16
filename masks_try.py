# -*- coding: utf-8 -*-
"""
Created on Mon Mar 10 16:52:42 2025

@author: binae
"""

import os
import numpy as np
import tifffile as tiff
from cellpose import models
from cellpose.io import imread
import SimpleITK as sitk
# os.makedirs('Results', exist_ok=True)
import argparse


def get_every_100th_image(image_stack):
    frames_to_align=[]
    with tiff.TiffFile(image_stack) as tif:
        for i, page in enumerate(tif.pages):  # Process pages inside the open block
            img = page.asarray()  
            if i % 100 == 0:
                    frames_to_align.append(img)
                
           
                   
    frames_to_align=np.stack(frames_to_align, axis=0)
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
    print('Returning translation parameters')
    
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
            if i<=100:
                translation_x=np.round(fraction*translation[quotient,0])
                translation_y=np.round(fraction*translation[quotient,1])
            else:
                translation_x=np.round(np.sum(translation[0:quotient,0])+fraction*translation[quotient,0]) #sum is start:stop excluding the stop
                translation_y=np.round(np.sum(translation[0:quotient,1])+fraction*translation[quotient,1]) #sum is start:stop excluding the stop
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
    columns=np.where(to_cut.any(0))[0]
    rows=np.where(to_cut.any(1))[0]
    resampled_stack_cut=resampled_stack[:,rows.min():rows.max()+1,columns.min():columns.max()+1]
    if image_stack.endswith('voltage.tif'):
       np.save(os.path.join(output_path,os.path.basename(image_stack.replace('_voltage.tif','_rows.npy'))), rows)
       np.save(os.path.join(output_path,os.path.basename(image_stack.replace('_voltage.tif','_columns.npy'))),columns)
    with tiff.TiffWriter(os.path.join(output_path,os.path.basename(image_stack.replace('.tif','_aligned.tif'))), bigtiff=True) as writer: 
         writer.write(resampled_stack_cut) 
    print('Returning aligned stack')

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

#%% 
# Process files
for file in os.listdir(input_folder):
    if file.endswith("_voltage.tif"):
        print(f"Processing voltage file: {file}")
        image_stack=os.path.join(input_folder,file)
    # image_stack=r"20250307_slip2_area1_Thapsigargin_1uM_1_MMStack_Default_combined_voltage.tif"
        #%% get frames to align    
        frames_to_align=get_every_100th_image(image_stack)
        #%% segment to align
        binary_masks=segment_voltage_frames(frames_to_align,output_path)
        #%% masks registration 
        translation =get_all_translation_vectors(binary_masks,image_stack,output_path)
        # np.save(image_stack.replace('combined_voltage.tif','translation.npy'),translation)
        # translation = np.load('20250307_slip2_area1_Thapsigargin_1uM_1_MMStack_Default_translation.npy')
        align_frames(image_stack,translation,output_path)
        calcium_sister_file=file.replace('voltage.tif','ca.tif')
        image_stack=os.path.join(input_folder,calcium_sister_file)
        print("Processing calcium sister file")
        align_frames(image_stack,translation,output_path)
        
        
        
        