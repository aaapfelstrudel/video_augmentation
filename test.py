import os
import faiss
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import warnings

warnings.filterwarnings("ignore")

# Paths to the query and base image directories
query_path = '/gpfs/home4/dshen/video_augmentation/dataset/extracted_frames'
base_path = '/gpfs/home4/dshen/video_augmentation/dataset/random'

# Paths to specific images in the query and base directories
query_image_paths = [
    os.path.join(query_path, "00001.png"),
    os.path.join(query_path, "00018.png"),
    os.path.join(query_path, "00034.png")
]

base_image_paths = [
    os.path.join(base_path, "berlin_000534_000019_leftImg8bit.png"),
    os.path.join(base_path, "bielefeld_000000_000321_leftImg8bit.png"),
    os.path.join(base_path, "leverkusen_000009_000019_leftImg8bit.png")
]

# Load the DINOV2 model
# Replace with the actual way to load the DINOV2 model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14")
model = model.to(device)

# Define functions to extract features using DINOV2
def extract_features(model, images, device):
    model.eval()
    features = []
    with torch.no_grad():
        images = images.to(device)
        feature = model(images.float())
        features.append(feature.cpu().numpy())
    return np.vstack(features)

data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Process query images
query_images = []
for img_path in query_image_paths:
    image = Image.open(img_path).convert("RGB")
    image = data_transform(image)
    query_images.append(image)
query_images = torch.stack(query_images)

# Process base images
base_images = []
for img_path in base_image_paths:
    image = Image.open(img_path).convert("RGB")
    image = data_transform(image)
    base_images.append(image)
base_images = torch.stack(base_images)

# Extract features from query and base images
query_features = extract_features(model, query_images, device)
base_features = extract_features(model, base_images, device)

# Normalize features to unit length (recommended for cosine similarity)
query_features = query_features / np.linalg.norm(query_features, axis=1, keepdims=True)
base_features = base_features / np.linalg.norm(base_features, axis=1, keepdims=True)

# Create Faiss index for the base features
d = base_features.shape[1]  # Dimension of the features
index = faiss.IndexFlatIP(d)  # Using Inner Product for similarity (cosine similarity)
index.add(base_features.astype(np.float32))

# Perform search for all query images
k = len(base_features)  # Number of matches to retrieve, can be changed to fewer
similarity_matrix = np.zeros((len(query_features), len(base_features)))

for i, query_feature in enumerate(query_features):
    query_feature = np.expand_dims(query_feature, axis=0).astype(np.float32)
    distances, indices = index.search(query_feature, k)
    similarity_matrix[i, indices[0]] = distances[0]

# Save the similarity matrix to file
np.save("similarity_matrix.npy", similarity_matrix)

# Example: Printing the similarity matrix
print("Similarity Matrix:")
print(similarity_matrix)

# The similarity matrix is now available, where each element (i, j) represents the similarity score between
# query image i and base image j.
