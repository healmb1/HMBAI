import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024
    progress_bar = tqdm(total=total_size, unit='iB', unit_scale=True)
    
    with open(filename, 'wb') as f:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            f.write(data)
    progress_bar.close()

def main():
    # Create checkpoints directory
    os.makedirs('checkpoints', exist_ok=True)
    
    # Download V1 checkpoints
    print("Downloading OpenVoice V1 checkpoints...")
    v1_url = "https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_1226.zip"
    v1_zip = "checkpoints_v1.zip"
    download_file(v1_url, v1_zip)
    
    # Extract V1 checkpoints
    print("Extracting V1 checkpoints...")
    with zipfile.ZipFile(v1_zip, 'r') as zip_ref:
        zip_ref.extractall('checkpoints')
    
    # Clean up
    os.remove(v1_zip)
    
    print("Download and extraction complete!")

if __name__ == "__main__":
    main() 