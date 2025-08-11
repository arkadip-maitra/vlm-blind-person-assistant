import os
import json
import requests
from tqdm import tqdm

# Path to the JSON file
json_file_path = "all.json"  # Change to your actual file path

# Folder to store downloaded images
output_folder = "downloaded_images_lfvqa"
os.makedirs(output_folder, exist_ok=True)

# Load JSON data
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Download images
for key, item in tqdm(data.items()):
    image_url = item.get("image_url")
    if image_url:
        try:
            response = requests.get(image_url)
            response.raise_for_status()  # Ensure we catch HTTP errors

            # Save the image with the key as the filename
            filename = os.path.join(output_folder, f"{key}.jpg")
            with open(filename, 'wb') as img_file:
                img_file.write(response.content)
                
        except Exception as e:
            print(f"Failed to download image for {key}: {e}")
