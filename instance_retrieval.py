import os
import faiss
import numpy as np
import torch
import pandas as pd
from torchvision import transforms
from PIL import Image
from data.gta5_dataset import GTA5DataSet
from data.cityscapes_dataset import cityscapesDataSet

# Paths to the query and base image directories
query_path = "/gpfs/home4/dshen/dataset/gta5/"
base_path = "/gpfs/home4/dshen/dataset/cityscape/"
result_path = "/gpfs/home4/dshen/video_augmentation/result/"

# Load the DINOV2 model
# Replace with the actual way to load the DINOV2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
model = model.to(device)
model.eval()

to_tensor = transforms.ToTensor()

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

def instance_retrieval(query_image, query_label, dataset_features, dataset_labels, k):

    similarity_mtx = np.zeros((2, k + 1), dtype=object)  # 2*(k+1) mtx 
    similarity_mtx[1,0] = query_label  # Set GTA image labels in the [0,0]
    print("initialize similarity mtx")

    query_features = extract_features(query_image, device)
    
    #query_features_cpu = query_features.cpu().numpy()
    #norm = np.linalg.norm(query_features_cpu, axis=1, keepdims=True)
    #query_features = query_features / torch.tensor(norm, device=query_features.device)
    
    distances, indices = nearest_neighbor_search(query_features, dataset_features, k)
    
    # Set the labels and distances in the correct order
    similarity_mtx[0, 1:] = [dataset_labels[idx] for idx in indices[0]]  # Set the indices
    similarity_mtx[1, 1:] = distances[0]  # Fill in the similarity scores

    return similarity_mtx 

# Extract features from query and base images
# loda gta5 data

# calculate similarity mtx with whole gta5 dataset bering query images respect to each city in cityscapes dataset 
city = os.listdir(base_path)
# for i in range(0,len(city)):
# print(f'load {city[i]}')  
cityscapes_dataset = cityscapesDataSet(root=base_path, color='RGB', city=city[2])
city_features = []; city_labels=[]
while True:
    city_image, city_label, city_status = cityscapes_dataset.__getitem__()
    base_features = extract_features(city_image, device)
    
    #base_features_cpu = base_features.cpu().numpy()
    #norm = np.linalg.norm(base_features_cpu, axis=1, keepdims=True)
    #base_features = base_features / torch.tensor(norm, device=base_features.device)
    
    city_features.append(base_features)
    city_labels.append(city_label)
    if city_status: break


gta_clips = sorted(os.listdir(query_path), key=lambda x: int(x.split('_')[1]))
gta5_dataset = GTA5DataSet(root=query_path, color='RGB', clip=gta_clips[27])
all_results = []

while True:
    
    gta_image, gta_label, gta_status = gta5_dataset.__getitem__()
    result = instance_retrieval(gta_image, gta_label, city_features, city_labels, len(city_features))
    all_results.append(result)
    if gta_status: break

# Convert all_results to a single DataFrame and save it to a CSV
# Flatten the list to prepare it for DataFrame conversion
flat_results = [item for sublist in all_results for item in sublist]  # Flattening list of lists of results

# Convert to DataFrame
df_result = pd.DataFrame(flat_results)

# Save to CSV
result_filename = os.path.join(result_path, f"{city[2]}.csv")
df_result.to_csv(result_filename, index=False, header=False)
print(f"Saved combined similarity matrix to '{result_filename}'")