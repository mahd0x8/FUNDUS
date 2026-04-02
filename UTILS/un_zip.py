import zipfile
import os

zip_path = "FUNDUS OUTPUT-20260316T211601Z-1-001.zip"
output_dir = "FUNDUS INFERENCE"

os.makedirs(output_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

print(f"Successfully unzipped {zip_path} to {output_dir}")