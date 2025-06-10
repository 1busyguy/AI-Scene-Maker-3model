import os
import gdown
from basicsr.utils.download_util import load_file_from_url

# Create directories
os.makedirs('gfpgan/weights', exist_ok=True)
os.makedirs('realesrgan/weights', exist_ok=True)

# Download GFPGAN model
load_file_from_url(
    url='https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth',
    model_dir='gfpgan/weights',
    progress=True,
    file_name='GFPGANv1.4.pth'
)

# Download RealESRGAN model
load_file_from_url(
    url='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
    model_dir='realesrgan/weights',
    progress=True,
    file_name='RealESRGAN_x4plus.pth'
)

print("Models downloaded successfully!")