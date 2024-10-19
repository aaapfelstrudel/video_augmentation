import os
import matplotlib.pyplot as plt
import torch
from torchvision.transforms.functional import to_pil_image
from data.gta5_dataset import GTA5DataSet
from data.cityscapes_dataset import cityscapesDataSet
import torchvision.transforms as transforms

# # Instantiate the dataset
# city = os.listdir(r'C:\Users\DR\video_augmentation\dataset\cityscapes')
# print(city[2])
# dataset = cityscapesDataSet(root=r'C:\Users\DR\video_augmentation\dataset\cityscapes', color='RGB', city=city[2])

# # Mean and std used in normalization
# mean = [0.485, 0.456, 0.406]
# std = [0.229, 0.224, 0.225]

# # Function to reverse normalization
# def reverse_normalize(tensor, mean, std):
#     for t, m, s in zip(tensor, mean, std):
#         t.mul_(s).add_(m)  # Undo the normalization
#     return tensor

# # Manually loop through the dataset and display images
# while True:
#     # Get the next image, label, and status
#     image, label, status = dataset.__getitem__()

#     if status:
#         print(f"Loaded Image Label: {label}")
#         break
#     print(f"Loaded Image Label: {label}")

#     # # Reverse the normalization to visualize the image correctly
#     # image = reverse_normalize(image.clone(), mean, std)

#     # # Convert the tensor image back to a PIL image
#     # image_pil = to_pil_image(image)

#     # # Display the image using PIL's show method or matplotlib
#     # plt.imshow(image_pil)
#     # plt.title(label)
#     # plt.axis('off')  # Hide axes for better visualization
#     # plt.show()

#     # # Optional: Add a break or input to pause before showing next image
#     # # input("Press Enter to show the next image...")






# Instantiate the dataset
dataset = GTA5DataSet(root=r'C:\Users\DR\video_augmentation\dataset\gta5', color='RGB', index=31)

# Mean and std used in normalization
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Function to reverse normalization
def reverse_normalize(tensor, mean, std):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Undo the normalization
    return tensor

# Manually loop through the dataset and display images
while True:
    # Get the next image, label, and status
    image, label, status = dataset.__getitem__()

    if status:
        print(f"Loaded Image Label: {label}")
        break

    print(f"Loaded Image Label: {label}")

    # # Reverse the normalization to visualize the image correctly
    # image = reverse_normalize(image.clone(), mean, std)

    # # Convert the tensor image back to a PIL image
    # image_pil = to_pil_image(image)

    # # Display the image using PIL's show method or matplotlib
    # plt.imshow(image_pil)
    # plt.title(label)
    # plt.axis('off')  # Hide axes for better visualization
    # plt.show()

    # # Optional: Add a break or input to pause before showing next image
    # # input("Press Enter to show the next image...")
