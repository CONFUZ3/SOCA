import logging
from hdx.utilities.easy_logging import setup_logging
from hdx.api.configuration import Configuration
from hdx.data.dataset import Dataset

setup_logging()
Configuration.create(hdx_site="prod", user_agent="SOCA_spoptv2", hdx_read_only=True)

# Test searching for high resolution population density maps
country_norm = "peru"

query = f"title:{country_norm}-high-resolution-population-density-maps-demographic-estimates"
print(f"Querying: {query}")

datasets = Dataset.search_in_hdx(query, rows=1)
if datasets:
    dataset = datasets[0]
    print(f"Dataset found: {dataset['name']}")
    
    resources = dataset.get_resources()
    for r in resources:
        print(f" - Resource: {r['name']}, Format: {r['format']}, Size: {r.get('size')}")
else:
    print("No datasets found.")
