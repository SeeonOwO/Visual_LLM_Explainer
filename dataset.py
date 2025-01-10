import os
import json
import pickle
import requests
from io import BytesIO
from pathlib import Path
from PIL import Image, ImageSequence
from torch.utils.data import Dataset
from transformers import AutoProcessor


class ImageCaptioningDataset(Dataset):
    def __init__(self, json_file, cache_dir='', caching=True):
        # Load raw json files
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        self.dataset = [item for item in data if item["Simplified_Caption"] != "N/A"]
        self.processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

        # Enable cache for fast data accessing
        self.caching = caching
        if self.caching:
            assert cache_dir
            self.version = json_file.split('/')[-1][:-5]
            self.cache_dir = Path(cache_dir) / self.version
            os.makedirs(self.cache_dir, exist_ok = True) 
        
    def __len__(self):
        return len(self.dataset)

    
    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Load the image information if already cached
        if self.caching and f'{idx}.pk' in os.listdir(self.cache_dir):
            with open(self.cache_dir / f'{idx}.pk', 'rb') as f:
                encoding = pickle.load(f)
        else:
            image_url = item["s3_fileUrl"]
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content))
            encoding = self.processor(images=image, padding="max_length", return_tensors="pt")
            
            encoding = {k: v.squeeze() for k, v in encoding.items()}
            encoding["image_id"] = str(idx)
            encoding["image"] = image
            encoding["s3_fileUrl"] = image_url
            encoding["text"] = item["Simplified_Caption"]
            encoding["agent_classifier"] = item["Agent-classifier"]
            encoding["geometry"] = item["geometry"]
            encoding["action"] = item["label"]

            # Cache the loaded image information if enabled
            if self.caching:
                with open(self.cache_dir / f'{idx}.pk', 'wb') as f:
                    pickle.dump(encoding, f)
            
        return encoding