# download_models.py
import os
import urllib.request
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

# Create directories
os.makedirs('gfpgan/weights', exist_ok=True)
os.makedirs('realesrgan/weights', exist_ok=True)

print("Downloading GFPGAN model...")
download_url(
    'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
    'gfpgan/weights/GFPGANv1.4.pth'
)

print("Downloading RealESRGAN model...")
download_url(
    'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    'realesrgan/weights/RealESRGAN_x4plus.pth'
)

print("Models downloaded successfully!")