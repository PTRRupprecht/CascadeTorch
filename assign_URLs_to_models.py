# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 17:47:42 2026

@author: peter
"""

import glob
import os
from datetime import datetime
import ruamel.yaml as yaml

BASE_PATH = "C:\\Users\\peter\\Desktop\\CascadeTorch\\CascadeTorch\\"
OUTPUT_FILE = os.path.join(BASE_PATH, "available_models_CascadeTorch.yaml")

zip_files = [
    os.path.basename(path)
    for path in glob.glob(f"{BASE_PATH}/*.zip")
]

import requests
import xml.etree.ElementTree as ET

# Store results
results = {}

for zip_file in zip_files:
    
    model_name = zip_file.replace('.zip', '')

    # --- Configuration ---
    USER = "rupprecht@XXX.ch"
    PASS = "XXXXX"
    FILE = "/CascadeTorch/"+zip_file
    
    API_URL = "https://drive.switch.ch/ocs/v2.php/apps/files_sharing/api/v1/shares"

    # --- Request ---
    headers = {
        "OCS-APIRequest": "true"
    }

    data = {
        "path": FILE,
        "shareType": 3,     # public link
        "permissions": 1   # read-only
    }

    response = requests.post(
        API_URL,
        headers=headers,
        data=data,
        auth=(USER, PASS)
    )

    response.raise_for_status()

    # --- Parse XML ---
    root = ET.fromstring(response.text)

    # Extract the <url> element
    url_elem = root.find(".//url")
    if url_elem is None:
        raise RuntimeError("No <url> found in OCS response")

    share_url = url_elem.text
    download_url = share_url + "/download"

    results[model_name] = {
    'Link': download_url,
    'Info': ''
    }

    print("Public link:", share_url)
    print("Direct download:", download_url)
    
    
# Try to load model info from available_models.yaml
try:
    yaml_obj = yaml.YAML(typ='rt')
    with open(os.path.join(BASE_PATH, 'Pretrained_models', 'available_models.yaml'), 'r') as yf:
        model_info = yaml_obj.load(yf)
    
    # Add info from available_models.yaml
    for model_name in results:
        if model_name in model_info:
            results[model_name]['Info'] = model_info[model_name].get('Description', '')
except Exception as e:
    print(f"Warning: Could not load model info: {e}")

# Write results to YAML file
print(f"\n\nWriting results to {OUTPUT_FILE}")

yaml_obj = yaml.YAML()
yaml_obj.default_flow_style = False

with open(OUTPUT_FILE, 'w') as f:
    # Write header as comments
    f.write("###\n")
    f.write("### Download links for available models\n")
    f.write("###\n")
    f.write(f"### Last update: {datetime.now().strftime('%Y-%m-%d')}\n")
    f.write("###\n\n")
    
    # Write YAML data
    yaml_obj.dump(results, f)

print(f"✓ Successfully wrote {len(results)} entries to {OUTPUT_FILE}")    