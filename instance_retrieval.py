import torch
import faiss
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dinov2.model import load_dinov2_model  # Hypothetical import, depending on actual DINOV2 implementation

# Define functions to extract features using DINOV2
def extract_features(model, dataloader, device):
    model.eval()
    features = []
    with torch.no_grad():
        for images, _ in dataloader:
            images = images.to(device)
            feature = model(images)
            features.append(feature.cpu().numpy())
    return np.vstack(features)

# Load the DINOV2 model
# Replace with the actual way to load the DINOV2 model
model = load_dinov2_model("dinov2_model_path")  # Example placeholder
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Load the datasets - query and base
data_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

query_dataset = datasets.ImageFolder("/path/to/query_dataset", transform=data_transform)
base_dataset = datasets.ImageFolder("/path/to/base_dataset", transform=data_transform)

query_loader = DataLoader(query_dataset, batch_size=16, shuffle=False)
base_loader = DataLoader(base_dataset, batch_size=16, shuffle=False)

# Extract features from query and base datasets
device = "cuda" if torch.cuda.is_available() else "cpu"
query_features = extract_features(model, query_loader, device)
base_features = extract_features(model, base_loader, device)

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