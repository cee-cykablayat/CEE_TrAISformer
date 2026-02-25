import nbformat as nbf
import os

notebook_path = '/home/crimsondeepdarshak/Desktop/Deep_Darshak/References/TRAIS_former_paper_Work_/CEE-Replication/Notebook/03_embedding_pipeline.ipynb'

nb = nbf.v4.new_notebook()

nb.cells = [
    nbf.v4.new_markdown_cell("# AIS Dataset Full Embedding and Hyperparameter Tuning\n\nThis notebook demonstrates how to calculate the optimal resolution parameters for AIS embeddings by minimizing reconstruction error. Finally, it provides a pipeline to process the **entire** dataset efficiently using batching to avoid Out-Of-Memory (OOM) errors."),
    
    nbf.v4.new_code_cell("""import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import gc
import os
from tqdm.auto import tqdm"""),
    
    nbf.v4.new_markdown_cell("## 1. Class Definitions\n\nWe first define our core classes: `AISConfig`, `AISDiscretizer`, `AISEmbeddingBuilder`, and `AISReconstructor`."),
    
    nbf.v4.new_code_cell("""class AISConfig:
    def __init__(self, lat_range, lon_range, lat_res=0.01, lon_res=0.01, sog_res=1.0, cog_res=5.0, max_sog=40.0, embed_dim=64, device="cuda"):
        self.lat_min, self.lat_max = lat_range
        self.lon_min, self.lon_max = lon_range
        self.lat_res, self.lon_res = lat_res, lon_res
        self.sog_res, self.cog_res = sog_res, cog_res
        self.max_sog = max_sog
        self.embed_dim = embed_dim
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Bin counts, conservatively rounded up
        self.N_lat = int(math.ceil((self.lat_max - self.lat_min) / lat_res)) + 1
        self.N_lon = int(math.ceil((self.lon_max - self.lon_min) / lon_res)) + 1
        self.N_sog = int(math.ceil(self.max_sog / sog_res)) + 1
        self.N_cog = int(math.ceil(360.0 / cog_res)) + 1

    def summary(self):
        return f"Lat/Lon Bins: {self.N_lat}/{self.N_lon}, SOG/COG: {self.N_sog}/{self.N_cog}"

class AISDiscretizer:
    def __init__(self, config):
        self.cfg = config

    def forward(self, tensor):
        lat, lon, sog, cog = tensor[:, 0], tensor[:, 1], tensor[:, 2], tensor[:, 3]
        lat_bin = torch.clamp(torch.floor((lat - self.cfg.lat_min) / self.cfg.lat_res).long(), 0, self.cfg.N_lat - 1)
        lon_bin = torch.clamp(torch.floor((lon - self.cfg.lon_min) / self.cfg.lon_res).long(), 0, self.cfg.N_lon - 1)
        sog_bin = torch.clamp(torch.floor(sog / self.cfg.sog_res).long(), 0, self.cfg.N_sog - 1)
        cog_bin = torch.clamp(torch.floor(cog / self.cfg.cog_res).long(), 0, self.cfg.N_cog - 1)
        return lat_bin, lon_bin, sog_bin, cog_bin

class AISEmbeddingBuilder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.lat_embed = nn.Embedding(config.N_lat, config.embed_dim)
        self.lon_embed = nn.Embedding(config.N_lon, config.embed_dim)
        self.sog_embed = nn.Embedding(config.N_sog, config.embed_dim)
        self.cog_embed = nn.Embedding(config.N_cog, config.embed_dim)

    def forward(self, lat_bin, lon_bin, sog_bin, cog_bin):
        e_lat = self.lat_embed(lat_bin)
        e_lon = self.lon_embed(lon_bin)
        e_sog = self.sog_embed(sog_bin)
        e_cog = self.cog_embed(cog_bin)
        return torch.cat([e_lat, e_lon, e_sog, e_cog], dim=-1)

class AISReconstructor:
    def __init__(self, config):
        self.cfg = config

    def reconstruct(self, lat_bin, lon_bin, sog_bin, cog_bin):
        lat = self.cfg.lat_min + (lat_bin + 0.5) * self.cfg.lat_res
        lon = self.cfg.lon_min + (lon_bin + 0.5) * self.cfg.lon_res
        sog = (sog_bin + 0.5) * self.cfg.sog_res
        cog = (cog_bin + 0.5) * self.cfg.cog_res
        return torch.stack([lat, lon, sog, cog], dim=1)"""),

    nbf.v4.new_markdown_cell("## 2. Global Data Statistics\n\nTo construct our embedding configuration with global bounds, we need the overall dataset min/max values. We extract these directly without loading everything into memory (if using PyArrow/Parquet), but since we need a large sample for hyperparameter tuning, we will load a substantial dataframe."),
    
    nbf.v4.new_code_cell("""dataset_path = "/home/crimsondeepdarshak/Desktop/Deep_Darshak/References/TRAIS_former_paper_Work_/CEE-Replication/Notebook/region_1_interpolated.parquet"
# Read a large sample (e.g., 2 million rows) to tune hyperparameters robustly
df_sample = pd.read_parquet(dataset_path).sample(2_000_000, random_state=42)

# Global bounds
lat_range = (df_sample.Latitude.min(), df_sample.Latitude.max())
lon_range = (df_sample.Longitude.min(), df_sample.Longitude.max())
max_sog = min(df_sample.SOG.max(), 40.0) # Cap at realistic max speed if there's noise

print(f"Lat Range: {lat_range}")
print(f"Lon Range: {lon_range}")
print(f"Max SOG: {max_sog}")"""),

    nbf.v4.new_markdown_cell("## 3. Hyperparameter Tuning for Optimal Resolution\n\nWe iteratively evaluate different resolutions (`lat_res`, `lon_res`, `sog_res`, `cog_res`) to track the **Mean Reconstruction Error** and the **Total Number of Parameters** (Memory/Complexity metric). \nThis determines the convergence criteria."),

    nbf.v4.new_code_cell("""device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert dataframe sample to tensor once
tensor_sample = torch.tensor(
    df_sample[["Latitude", "Longitude", "SOG", "COG"]].values,
    dtype=torch.float32,
    device=device
)

def evaluate_config(config):
    discretizer = AISDiscretizer(config)
    reconstructor = AISReconstructor(config)
    
    lat_bin, lon_bin, sog_bin, cog_bin = discretizer.forward(tensor_sample)
    reconstructed = reconstructor.reconstruct(lat_bin, lon_bin, sog_bin, cog_bin)
    
    # Calculate Mean L2 Reconstruction Error
    error = torch.norm(tensor_sample - reconstructed, dim=1).mean().item()
    
    num_params = (config.N_lat + config.N_lon + config.N_sog + config.N_cog) * config.embed_dim
    return error, num_params

# Define search space
spatial_resolutions = [0.1, 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001, 0.00001]
kinematic_resolutions = [(5.0, 15.0), (2.0, 10.0), (1.0, 5.0), (0.5, 2.0), (0.1, 0.5)]

results = []
for s_res in tqdm(spatial_resolutions, desc="Spatial Res"):
    for k_res_idx, (sog_res, cog_res) in enumerate(kinematic_resolutions):
        cfg = AISConfig(
            lat_range=lat_range, 
            lon_range=lon_range, 
            lat_res=s_res, 
            lon_res=s_res, 
            sog_res=sog_res, 
            cog_res=cog_res,
            max_sog=max_sog,
            embed_dim=64,
            device=device
        )
        try:
            error, params = evaluate_config(cfg)
            results.append({
                "lat_res": s_res,
                "lon_res": s_res,
                "sog_res": sog_res,
                "cog_res": cog_res,
                "mean_reconstruction_error": error,
                "total_params": params
            })
        except RuntimeError as e: # Catch OutOfMemory or index errors
            print(f"Failed for conf {s_res}, {sog_res}, {cog_res}: {e}")
            continue

df_results = pd.DataFrame(results)"""),

    nbf.v4.new_markdown_cell("### Plotting the Convergence Criteria\n\nWe plot Reconstruction Error against Model Complexity (Number of Parameters). The optimal point is the 'elbow' of the curve where error is minimized before parameter explosion."),

    nbf.v4.new_code_cell("""# Plotting
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df_results, 
    x="total_params", 
    y="mean_reconstruction_error", 
    marker="o"
)
plt.xscale("log")
plt.yscale("log")
plt.title("Reconstruction Error vs Parameter Complexity")
plt.xlabel("Total Embedding Parameters (Log Scale)")
plt.ylabel("Mean L2 Reconstruction Error (Log Scale)")
plt.grid(True, which="both", ls="--")
plt.show()

# Display sorted results
df_results.sort_values("mean_reconstruction_error").head(10)"""),

    nbf.v4.new_markdown_cell("From the plot, you can identify an acceptable threshold. For the full dataset pipeline below, let's take a computationally efficient yet precise resolution (e.g. `lat_res=0.00001, sog_res=0.5` as previously tested, or slightly larger if RAM requires)."),

    nbf.v4.new_markdown_cell("## 4. Full Dataset Pipeline Processing\n\nTo process the entire 28 million rows without causing CUDA OutOfMemory, we will load the Parquet file iteratively using PyArrow/Pandas batches, apply the discretizer, and export the encoded features to a concise Parquet format. If you need continuous dense vector embeddings, we show how to compile those too."),

    nbf.v4.new_code_cell("""# 1. Define Optimal Configuration Here
optimal_config = AISConfig(
    lat_range=lat_range,
    lon_range=lon_range,
    lat_res=0.00001,
    lon_res=0.00001,
    sog_res=0.5,
    cog_res=0.5,
    embed_dim=64,
    device=device
)

print("Optimal Config:", optimal_config.summary())
discretizer = AISDiscretizer(optimal_config)
# embedding_builder = AISEmbeddingBuilder(optimal_config).to(device)

# Processing entirely by chunk
chunk_size = 2_000_000
output_file = "/home/crimsondeepdarshak/Desktop/Deep_Darshak/References/TRAIS_former_paper_Work_/CEE-Replication/Notebook/region_1_encoded_features.parquet"

# To avoid crashing, we can read chunks if pandas supports it natively via PyArrow, 
# or just process a loaded dataframe in minibatches. 
# We'll load full df into RAM (takes ~5GB), and then batch loop it into GPU.

print("Loading full dataset into RAM...")
df_full = pd.read_parquet(dataset_path)
print(f"Total Rows: {len(df_full)}")

encoded_bins = []

print("Starting batch embedding computation...")
for i in tqdm(range(0, len(df_full), chunk_size)):
    batch_df = df_full.iloc[i : i+chunk_size]
    
    # 1. Prepare tensor
    batch_tensor = torch.tensor(
        batch_df[["Latitude", "Longitude", "SOG", "COG"]].values,
        dtype=torch.float32,
        device=device
    )
    
    # 2. Get Discrete Indices (Bins)
    lat_bin, lon_bin, sog_bin, cog_bin = discretizer.forward(batch_tensor)
    
    # 3. Create DataFrame chunk and append
    chunk_encoded_df = pd.DataFrame({
        "MMSI": batch_df["MMSI"].values,
        "Time": batch_df["Time"].values,
        "lat_bin": lat_bin.cpu().numpy(),
        "lon_bin": lon_bin.cpu().numpy(),
        "sog_bin": sog_bin.cpu().numpy(),
        "cog_bin": cog_bin.cpu().numpy(),
    })
    encoded_bins.append(chunk_encoded_df)
    
    # Memory management
    del batch_tensor, lat_bin, lon_bin, sog_bin, cog_bin, chunk_encoded_df
    torch.cuda.empty_cache()

# Concatenate all encoded chunks
df_final_encoded = pd.concat(encoded_bins, ignore_index=True)

# Export the discrete bins. This is highly memory efficient.
# In Downstream PyTorch training (TRAIS-former or similar), you pass these bins to AISEmbeddingBuilder.
df_final_encoded.to_parquet(output_file, index=False)
print(f"Successfully saved fully encoded discrete dataset to {output_file}")"""),

    nbf.v4.new_markdown_cell("### Optional: Generating Dense Embedding Tensors\n\nIf your pipeline specifically requires pre-calculating the final float vectors (e.g. $[N, 256]$ dimensional arrays) to disk, run the code below. **Note: A 28M x 256 float32 matrix consumes ~28GB of disk space.**"),

    nbf.v4.new_code_cell("""'''
# Example code for dense embedding export

builder = AISEmbeddingBuilder(optimal_config).to(device)
embedding_file = "region_1_dense_embeddings.pt"

# You would do this iteratively as well:
with torch.no_grad():
    list_of_embeddings = []
    
    # Process the binned dataframe just created
    for i in tqdm(range(0, len(df_final_encoded), chunk_size)):
        chunk = df_final_encoded.iloc[i : i+chunk_size]
        l = torch.tensor(chunk["lat_bin"].values, device=device)
        o = torch.tensor(chunk["lon_bin"].values, device=device)
        s = torch.tensor(chunk["sog_bin"].values, device=device)
        c = torch.tensor(chunk["cog_bin"].values, device=device)
        
        embed_chunk = builder(l, o, s, c)
        
        # Save or append
        list_of_embeddings.append(embed_chunk.cpu())
        
        del l, o, s, c, embed_chunk
        torch.cuda.empty_cache()

    final_tensor = torch.cat(list_of_embeddings, dim=0)
    torch.save(final_tensor, embedding_file)
    print(f"Saved {final_tensor.shape} embeddings to {embedding_file}")
'''
print("Dense generation code provided but commented out to save disk space.")""")
]

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook {notebook_path} generated successfully.")
