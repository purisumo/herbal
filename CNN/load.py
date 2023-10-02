import os
import cv2
from PIL import Image
import numpy as np
# temporal storage for labels and images
data=[]
labels=[]

# Get the herb directory
none = os.listdir(os.getcwd() + "/CNN/data/ADELFA")
for x in none:
    """
    Loop through all the images in the directory
    1. Convert to arrays
    2. Resize the images
    3. Add image to dataset
    4. Add the label
    """
    imag=cv2.imread(os.getcwd() + "/CNN/data/ADELFA/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((224, 224))
    data.append(np.array(resized_image))
    labels.append(0)

none = os.listdir(os.getcwd() + "/CNN/data/ALOE VERA")
for x in none:
    """
    Loop through all the images in the directory
    1. Convert to arrays
    2. Resize the images
    3. Add image to dataset
    4. Add the label
    """
    imag=cv2.imread(os.getcwd() + "/CNN/data/ALOE VERA/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((224, 224))
    data.append(np.array(resized_image))
    labels.append(1)

none = os.listdir(os.getcwd() + "/CNN/data/sambong")
for x in none:
    """
    Loop through all the images in the directory
    1. Convert to arrays
    2. Resize the images
    3. Add image to dataset
    4. Add the label
    """
    imag=cv2.imread(os.getcwd() + "/CNN/data/sambong/" + x)
    img_from_ar = Image.fromarray(imag, 'RGB')
    resized_image = img_from_ar.resize((224, 224))
    data.append(np.array(resized_image))
    labels.append(2)


# load in herbs and labels
herbs=np.array(data)
labels=np.array(labels)
# save
np.save("herbs",herbs)
np.save("labels",labels)

# import os
# import cv2
# from PIL import Image
# import numpy as np

# def process_images_in_folders(root_directory):
#     """
#     Process all images in subfolders of the specified root directory.
    
#     Args:
#     root_directory (str): The root directory containing subfolders, each representing a category.
    
#     Returns:
#     data (list): List of processed images.
#     labels (list): List of corresponding labels.
#     """
#     data = []
#     labels = []

#     # Get the list of subdirectories (each representing a category)
#     subdirectories = [d for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]

#     for label, subdirectory in enumerate(subdirectories):
#         # Construct the full path to the subdirectory
#         subdirectory_path = os.path.join(root_directory, subdirectory)

#         # Get the list of image files in the subdirectory
#         image_files = os.listdir(subdirectory_path)

#         for image_file in image_files:
#             # Construct the full path to the image file
#             image_path = os.path.join(subdirectory_path, image_file)

#             # Read the image using OpenCV
#             imag = cv2.imread(image_path)

#             # Convert to a PIL Image and resize
#             img_from_ar = Image.fromarray(imag, 'RGB')
#             resized_image = img_from_ar.resize((50, 50))

#             # Append the resized image to the data list
#             data.append(np.array(resized_image))

#             # Append the label to the labels list
#             labels.append(label)

#     return data, labels

# # Specify the root directory containing subfolders for each category
# root_directory = "C:\herbal\CNN\data"

# # Process the images in subfolders
# data, labels = process_images_in_folders(root_directory)

# # Convert data and labels to NumPy arrays
# herbs = np.array(data)
# labels = np.array(labels)

# # Save the processed data and labels
# np.save("herbs", herbs)
# np.save("labels", labels)
