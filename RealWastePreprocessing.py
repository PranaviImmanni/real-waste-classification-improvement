# Install augmentor library for geometric augmentations
!pip install Augmentor
# Import directory navigation libraries
from os import listdir, rmdir
# Import image preprocessing libraries
import numpy as np
import cv2
from shutil import move, copy
from random import randint
from PIL import Image
from tensorflow.image import resize
from tensorflow.io import decode_jpeg, encode_jpeg, write_file
from tensorflow.dtypes import saturate_cast
from random import randint
# Import augmentor for geometric data augmentations
import Augmentor
# Import Colab runtime disconnection
from google.colab import runtime
# Import sleep to allow time for copying to process
from time import sleep

# ------------------------------- Constants -------------------------------- #
# Original dataset path
real_waste_original_path = ''
# RealWaste training and validation paths
real_waste_training_path = ''
real_waste_validation_path = ''
# RealWaste temporary training and validation paths for augmentation stages
temp_real_waste_training_path = ''
temp_real_waste_validation_path = ''
# Test path
test_path = ''
# Labels
labels = ['Cardboard', 'Food Organics', 'Glass', 'Metal', 'Paper', 'Plastic', 'Miscellaneous Trash', 'Textile Trash', 'Vegetation']
# Processed image dimensions
img_height = 524
img_width = 524

# ------------------------------- Functions -------------------------------- #
# Count images in directory
def count_images(path):
    directory = listdir(path)
    return len(directory)

# Split datasets into training, validation and testing
def split_dataset(input_path, train_path, val_path, test_path, val_prop, test_prop, label):
    # Count images in directory and get files names
    num_files = count_images(input_path+label)
    file_names = listdir(input_path+label)

    # Calculated number of validation and testing images
    num_val = round(num_files*val_prop)
    num_test = round(num_files*test_prop)
    
    # Initialise empty lists for image indexes
    val_list = []
    test_list = []
    
    # Randomly assign file indexes for validation dataset
    i = 0
    while i < num_val:
        r = randint(1, num_files - 1)
        if r not in val_list:
            val_list.append(r)
            i += 1

    # Move validation dataset files and rename
    name_incrementer = 1
    for j in val_list:
        copy(input_path+label+'/'+file_names[j], val_path+label+'/'+label+'_'+str(name_incrementer)+'.jpg')
        sleep(0.05)
        name_incrementer += 1
    
    # Randomly assign file indexes for testing dataset
    i = 0
    while i < num_test:
        r = randint(1, num_files - 1)
        if r not in test_list and r not in val_list:
            test_list.append(r)
            i += 1

    # Move test dataset files and rename
    name_incrementer = 1
    for j in test_list:
        copy(input_path+label+'/'+file_names[j], test_path+label+'/'+label+'_'+str(name_incrementer)+'.jpg')
        sleep(0.05)
        name_incrementer += 1
    
    # Move remaining files into training dataset
    i = 0
    while i < num_files:
        if i not in val_list and i not in test_list:
            copy(input_path+label+'/'+file_names[i], train_path+label+'/'+label+'_'+str(name_incrementer)+'.jpg')
            sleep(0.05)
            name_incrementer += 1
        i += 1

# Resize images
def resize_image(path, width, height):
    directory = listdir(path)
    for file in directory:
        # Read image as binary file
        image = open(path+'/'+file, 'rb')                          
        binary_representation = image.read()                        
        decoded_representation = decode_jpeg(binary_representation) 
        # Resize binary represnetation
        resized_image = resize(decoded_representation, [width, height])
        resized_image = saturate_cast(resized_image, 'uint8')
        encoded_image = encode_jpeg(resized_image)                  
        write_file(path+'/'+file, encoded_image)
        sleep(0.05)

# Check images for RGB mode and convert if required
def convert_rgb_images(path):
    print(path)
    directory = listdir(path)
    for file in directory:
        img = Image.open(path+'/'+file)
        # Convert to RGB mode if not already in format
        if img.mode != 'RGB':
            img.convert('RGB').save(path+'/'+file)
            sleep(0.05)
            
# Geometric augmentation function: distort and flip
def distort_and_flip(path):
    pipe = Augmentor.Pipeline(path)           # Create pipeline in path of dataset
    pipe.flip_left_right(1)                   # Horizontally flip every image
    pipe.random_distortion(1, 5, 5, 4)        # Randomly distort flipped images
    pipe.process()                            # Process images in pipeline

# Geometric augmentation: rotate and shear
def rotate_and_shear(path):
    pipe = Augmentor.Pipeline(path)           # Create pipeline in path of dataset
    pipe.rotate(1, 25, 25)                    # Rotate every image by 45 degrees
    pipe.shear(1, 15, 15)                     # Randomly shear between 0 and 15 degrees
    pipe.process()                            # Process images in pipeline

# Move files between directories
def move_files(augmented_path, dataset_path, new_file_name):
    directory = listdir(augmented_path)
    name_incrementer = 1
    for file in directory:
        move(augmented_path+'/'+file, dataset_path+'/'+new_file_name+'_'+str(name_incrementer)+'.jpg')
        name_incrementer += 1
        sleep(0.05)

# ---------------------------- Split RealWaste ----------------------------- #
print('\n|----------------------------------| RealWaste Splitting |---------------------------------|\n')

# Split datasets
for label in labels:
    split_dataset(real_waste_original_path, real_waste_training_path, real_waste_validation_path, test_path, 0.2, 0.1, label)

# Print file counts in paths
for label in labels:
    print('Number of files in RealWaste raw ' + label + ': ' + str(count_images(real_waste_original_path + label)))
    print('Number of files in RealWaste training ' + label + ': ' + str(count_images(real_waste_training_path + label)))
    print('Number of files in RealWaste validation ' + label + ': ' + str(count_images(real_waste_validation_path + label)))
    print('Number of files in RealWaste test ' + label + ': ' + str(count_images(test_path + label)))
    print()

# ---------------------- Pre-Process Images RealWaste ---------------------- #
print('|--------------------------------| RealWaste Pre-Processing |------------------------------|\n')
# Preprocess images
for label in labels:
    # RealWaste training
    print('\nBefore preprocessing RealWaste training ' + label + ': ' + str(count_images(real_waste_training_path + label)))
    convert_rgb_images(real_waste_training_path+label)
    resize_image(real_waste_training_path+label, img_width, img_height)
    print('After preprocessing RealWaste training ' + label + ': ' + str(count_images(real_waste_training_path + label)))
    
    # RealWaste validation
    print('Before preprocessing RealWaste validation ' + label + ': ' + str(count_images(real_waste_validation_path + label)))
    convert_rgb_images(real_waste_validation_path+label)
    resize_image(real_waste_validation_path+label, img_width, img_height)
    print('After preprocessing RealWaste validation ' + label + ': ' + str(count_images(real_waste_validation_path + label)))
    
    # Testing
    print('Before preprocessing test ' + label + ': ' + str(count_images(test_path + label)))
    convert_rgb_images(test_path+label)
    resize_image(test_path+label, img_width, img_height)
    print('After preprocessing test ' + label + ': ' + str(count_images(test_path + label)))
    print()

# ---------------------- Distort and Flip RealWaste ------------------------ #
print('|----------------------| RealWaste Augmentation - Distort and Flip |---------------------|\n')

# Perform augmentation
for label in labels:
    # RealWaste training
    print('\nTraining count RealWaste '+ label + ': ' + str(count_images(real_waste_training_path+label)))
    distort_and_flip(real_waste_training_path+label)
    print('Augmented training count RealWaste '+ label + ': ' + str(count_images(real_waste_training_path+label+'/output')))
    sleep(60)
    # RealWaste validation
    print('\nValidation count RealWaste '+ label + ': ' + str(count_images(real_waste_validation_path+label)))
    distort_and_flip(real_waste_validation_path+label)
    print('Augmented validation count RealWaste '+ label + ': ' + str(count_images(real_waste_validation_path+label+'/output')))
    sleep(60)
    # Move to temp directories
    move_files(real_waste_training_path+label+'/output', temp_real_waste_training_path+label+'/', label+'_Distort_Flip')
    sleep(60)
    move_files(real_waste_validation_path+label+'/output', temp_real_waste_validation_path+label+'/', label+'_Distort_Flip')
    sleep(60)
    # Delete augmented output folder
    rmdir(real_waste_training_path+label+'/output')
    rmdir(real_waste_validation_path+label+'/output')

# ---------------------- Rotate and Shear RealWaste ------------------------ #
print('|---------------------| RealWaste Augmentation - Rotate and Shear |--------------------|\n')

# Perform augmentation
for label in labels:
    # RealWaste training
    print('\nTraining count RealWaste '+ label + ': ' + str(count_images(real_waste_training_path+label)))
    rotate_and_shear(real_waste_training_path+label)
    print('Augmented training count RealWaste '+ label + ': ' + str(count_images(real_waste_training_path+label+'/output')))
    sleep(60)
    # RealWaste validation
    print('\nValidation count RealWaste '+ label + ': ' + str(count_images(real_waste_validation_path+label)))
    rotate_and_shear(real_waste_validation_path+label)
    print('Augmented validation count RealWaste '+ label + ': ' + str(count_images(real_waste_validation_path+label+'/output')))
    sleep(60)
    # Move files back to container
    move_files(real_waste_training_path+label+'/output', real_waste_training_path+label+'/', label+'_Rotate_Shear')
    sleep(60)
    move_files(real_waste_validation_path+label+'/output', real_waste_validation_path+label+'/', label+'_Rotate_Shear')
    sleep(60)
    # Delete augmented output folder
    rmdir(real_waste_training_path+label+'/output')
    rmdir(real_waste_validation_path+label+'/output')

# Move first augmentation back to directory
for label in labels:
    move_files(temp_real_waste_training_path+label, real_waste_training_path+label, label+'_Distort_Flip')
    sleep(60)
    move_files(temp_real_waste_validation_path+label, real_waste_validation_path+label, label+'_Distort_Flip')
    sleep(60)

# Print final label counts
print('Final training count RealWaste ' + label + ': ' + str(count_images(real_waste_training_path)))
print('Final validation count RealWaste ' + label + ': ' + str(count_images(real_waste_validation_path)))
print()

# -------------------------- Print Final Counts ---------------------------- #
# Print training counts for each label in training dataset
print('\n|-----------| Total Training Label Counts |-----------|')
for label in labels:
    print('RealWaste ' + label + ': ' + str(count_images(real_waste_training_path+label)))
print('\n')

# Print validation counts for each label
print('\|----------| Total Validation Label Counts |----------|')
for label in labels:
    print('RealWaste ' + label + ': ' + str(count_images(real_waste_validation_path+label)))
print('\n')

# Print testing counts for each label
print('\n|------------| Total Testing Label Counts |-----------|')
for label in labels:
    print('Test '+ label + ': ' + str(count_images(test_path+label)))

# -------------------------- Disconnect Runtime ---------------------------- #
runtime.unassign()   