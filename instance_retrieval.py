import os
import faiss
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from data.gta5_dataset import GTA5DataSet
from data.cityscapes_dataset import cityscapesDataSet

# Paths to the query and base image directories
query_path = r'C:\Users\DR\video_augmentation\dataset\gta5'
base_path = r'C:\Users\DR\video_augmentation\dataset\cityscapes'
result_path = r'C:\Users\DR\video_augmentation\result'

# Load the DINOV2 model
# Replace with the actual way to load the DINOV2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
model = model.to(device)
model.eval()

# Define functions to extract features using DINOV2
def extract_features(image, device):
    image_tensor = image.unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(image_tensor).float()
    return feature


# Function to perform nearest neighbor search
def nearest_neighbor_search(query_features, dataset_features, k=5):
    index = faiss.IndexFlatL2(query_features.shape[1])
    dataset_features = torch.cat(dataset_features, dim=0).cpu().numpy()
    index.add(dataset_features)
    distances, indices = index.search(query_features.cpu().numpy(), k)
    return distances, indices

def instance_retrieval(query_dataset, query_labels, dataset_features, dataset_labels, k):

    similarity_mtx = np.zeros((len(query_dataset) + 1, k + 1), dtype=object)  # add one row and column for labels
    similarity_mtx[0, 1:] = [dataset_labels[idx] for idx in range(k)]  # Set placeholder labels in the first row (excluding the first column)
    similarity_mtx[1:, 0] = [f"{query_labels}" for i in range(len(query_dataset))]  # Set GTA image labels in the first column

    for i, query_feature in enumerate(query_dataset):
        query_features = extract_features(query_feature, device)
        distances, indices = nearest_neighbor_search(query_features, dataset_features, k)
        
        # Set the labels and distances in the correct order
        similarity_mtx[0, 1:] = [dataset_labels[idx] for idx in indices[0]]  # Set the actual labels corresponding to the indices
        similarity_mtx[i + 1, 1:] = distances[0]  # Fill in the similarity scores

    return similarity_mtx

# Extract features from query and base images
# loda gta5 data
gta5_dataset = GTA5DataSet(root=query_path, color='RGB', index=1)
gta_images = []; gta_labels = []
while True:
    gta_image, gta_label, gta_status = gta5_dataset.__getitem__()
    if gta_status: break
    gta_images.append(gta_images)
    gta_labels.append(gta_label)

# calculate similarity mtx with whole gta5 dataset bering query images respect to each city in cityscapes dataset 
city = os.listdir(base_path)
for i in range(0,len(city)):  
    cityscapes_dataset = cityscapesDataSet(root=base_path, color='RGB', city=city[i])
    city_features = []; city_labels=[]
    while True:
        city_image, city_label, city_status = cityscapes_dataset.__getitem__()
        if city_status: break
        base_features = extract_features(city_image)
        # base_features = base_features / np.linalg.norm(base_features)
        city_features.append(base_features)
        city_labels.append(city_label)

    result = instance_retrieval(gta_images, gta_labels, city_features, city_labels, len(city_features))

    # Save the result as CSV file using the city name
    df_result = pd.DataFrame(result)
    result_filename = os.path.join(result_path, f"{city[i]}.csv")
    df_result.to_csv(result_filename, index=False, header=False)
    print(f"Saved similarity matrix for city '{city[i]}' to '{result_filename}'")
