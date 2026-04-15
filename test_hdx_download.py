import logging
import zipfile
import os
import pandas as pd
from hdx.utilities.easy_logging import setup_logging
from hdx.api.configuration import Configuration
from hdx.data.dataset import Dataset

setup_logging()
Configuration.create(hdx_site="prod", user_agent="SOCA_spoptv2", hdx_read_only=True)

# Test searching for high resolution population density maps
country_norm = "peru"

query = f"title:{country_norm}-high-resolution-population-density-maps-demographic-estimates"

datasets = Dataset.search_in_hdx(query, rows=1)
if datasets:
    dataset = datasets[0]
    print(f"Dataset found: {dataset['name']}")
    
    resources = dataset.get_resources()
    
    # Same filtering logic
    csv_resources = [r for r in resources if r.get_format().lower() == "csv" or "csv" in r.get("name", "").lower()]
    target_resource = None
    for r in csv_resources:
        name = r.get("name", "").lower()
        if "general" in name or "overall" in name:
            target_resource = r
            break
            
    if not target_resource:
        print("No target resource found")
    else:
        print(f"Target found: {target_resource['name']}")
        url, path = target_resource.download()
        print(f"Downloaded to {path}")
        
        try:
            if path.endswith('.zip') or zipfile.is_zipfile(path):
                df = pd.read_csv(path, compression='zip')
                print(f"Read CSV from ZIP, columns: {list(df.columns)}")
                print(f"Length: {len(df)}")
            else:
                df = pd.read_csv(path)
                print(f"Read CSV, columns: {list(df.columns)}")
        except Exception as e:
            print(f"Error loading: {e}")
        finally:
            if path and os.path.exists(path):
                os.unlink(path)
                print("Cleaned up download.")
else:
    print("No datasets found.")
