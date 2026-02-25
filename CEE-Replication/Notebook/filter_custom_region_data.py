import polars as pl
import json
import os

def get_bounding_box(geojson_path):
    print(f"Loading GeoJSON from {geojson_path}...")
    with open(geojson_path, 'r') as f:
        data = json.load(f)
        
    if "features" in data and len(data["features"]) > 0:
        coords = data["features"][0]["geometry"]["coordinates"][0]
    else:
        coords = data["geometry"]["coordinates"][0]
        
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    return min(lons), max(lons), min(lats), max(lats)

def filter_data():
    base_dir = '/home/crimsondeepdarshak/Desktop/Deep_Darshak/References/TRAIS_former_paper_Work_/CEE-Replication/Notebook'
    geojson_path = os.path.join(base_dir, 'region_1.geojson')
    
    if not os.path.exists(geojson_path):
        print(f"Error: {geojson_path} not found.")
        return

    # Extract actual bounding box limits
    min_lon, max_lon, min_lat, max_lat = get_bounding_box(geojson_path)
    print(f"Bounding Box: Lon [{min_lon:.4f}, {max_lon:.4f}], Lat [{min_lat:.4f}, {max_lat:.4f}]")
    
    # Read newly created chunked interpolated parquet files
    input_pattern = '/home/crimsondeepdarshak/Desktop/Deep_Darshak/AIS_data_demo/Processed/Interpolated/interpolated_chunk_*.parquet'
    output_parquet = os.path.join(base_dir, 'region_1_interpolated.parquet')
    
    print(f"Scanning parquet files: {input_pattern}")
    
    # We use scan_parquet to lazily load and filter
    lazy_df = (
        pl.scan_parquet(input_pattern)
        .filter(
            (pl.col("Longitude") >= min_lon) & (pl.col("Longitude") <= max_lon) &
            (pl.col("Latitude") >= min_lat) & (pl.col("Latitude") <= max_lat)
        )
        .sort(["MMSI", "Time"])
    )
    
    print("Executing query and filtering data (this may take a moment)...")
    try:
        # Evaluate lazily with streaming to avoid OOM
        df = lazy_df.collect(streaming=True)
    except Exception as e:
        print(f"Streaming failed. Falling back to standard collect. Error: {e}")
        df = lazy_df.collect()
        
    print(f"Filtered DataFrame shape: {df.shape}")
    print(f"Writing concatenated and sorted parquet to: {output_parquet}")
    
    df.write_parquet(output_parquet, compression="snappy")
    print("Data processing complete!")

if __name__ == "__main__":
    filter_data()
