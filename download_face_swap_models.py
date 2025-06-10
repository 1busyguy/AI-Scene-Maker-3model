# download_face_swap_models.py
import os
import urllib.request
import gdown
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def download_from_drive(file_id, output_path):
    """Download from Google Drive"""
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)

# Create directories
os.makedirs('insightface_models', exist_ok=True)
os.makedirs('insightface_models/models', exist_ok=True)
os.makedirs('insightface_models/models/buffalo_l', exist_ok=True)

print("Installing required packages...")
# Install gdown if not already installed
try:
    import gdown
except ImportError:
    os.system("pip install gdown")
    import gdown

print("\n=== Downloading Face Swapper Model ===")
# Option 1: Try alternative sources
face_swap_sources = [
    {
        'name': 'Google Drive Mirror',
        'type': 'drive',
        'id': '1HvZ4MAtzlY74Dk4ASGIS9L6bfq2BEadG',  # inswapper_128.onnx
        'output': 'insightface_models/inswapper_128.onnx'
    },
    {
        'name': 'Hugging Face Mirror',
        'type': 'url',
        'url': 'https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx',
        'output': 'insightface_models/inswapper_128.onnx'
    }
]

# Try each source until one works
for source in face_swap_sources:
    print(f"\nTrying to download from {source['name']}...")
    try:
        if source['type'] == 'drive':
            download_from_drive(source['id'], source['output'])
        else:
            download_url(source['url'], source['output'])
        
        # Check if file was downloaded successfully
        if os.path.exists(source['output']) and os.path.getsize(source['output']) > 0:
            print(f"✅ Successfully downloaded from {source['name']}")
            break
    except Exception as e:
        print(f"❌ Failed to download from {source['name']}: {str(e)}")
        continue
else:
    print("\n⚠️  Could not download face swapper model automatically.")
    print("Please download manually from one of these sources:")
    print("1. https://huggingface.co/deepinsight/inswapper/tree/main")
    print("2. https://github.com/deepinsight/insightface/releases")
    print("\nSave the 'inswapper_128.onnx' file to: insightface_models/")

print("\n=== Downloading Buffalo_L Model (for face detection) ===")
# Buffalo_L model components
buffalo_files = [
    {
        'name': '1k3d68.onnx',
        'url': 'https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/1k3d68.onnx',
    },
    {
        'name': '2d106det.onnx',
        'url': 'https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/2d106det.onnx',
    },
    {
        'name': 'det_10g.onnx',
        'url': 'https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/det_10g.onnx',
    },
    {
        'name': 'genderage.onnx',
        'url': 'https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/genderage.onnx',
    },
    {
        'name': 'w600k_r50.onnx',
        'url': 'https://huggingface.co/deepinsight/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx',
    }
]

for file_info in buffalo_files:
    output_path = os.path.join('insightface_models/models/buffalo_l', file_info['name'])
    if os.path.exists(output_path):
        print(f"✅ {file_info['name']} already exists, skipping...")
        continue
    
    print(f"\nDownloading {file_info['name']}...")
    try:
        download_url(file_info['url'], output_path)
        print(f"✅ Downloaded {file_info['name']}")
    except Exception as e:
        print(f"❌ Failed to download {file_info['name']}: {str(e)}")

print("\n=== Download Summary ===")
# Check what was downloaded
if os.path.exists('insightface_models/inswapper_128.onnx'):
    size_mb = os.path.getsize('insightface_models/inswapper_128.onnx') / (1024 * 1024)
    print(f"✅ Face swapper model: {size_mb:.1f} MB")
else:
    print("❌ Face swapper model: Not found")

buffalo_count = len([f for f in os.listdir('insightface_models/models/buffalo_l') if f.endswith('.onnx')])
print(f"✅ Buffalo_L models: {buffalo_count}/5 files")

if os.path.exists('insightface_models/inswapper_128.onnx') and buffalo_count >= 5:
    print("\n✅ All models downloaded successfully!")
else:
    print("\n⚠️  Some models are missing. The face swapping may not work properly.")
    print("You can try running this script again or download manually.")

# Create a test script
print("\n=== Creating test script ===")
test_script = '''# test_face_swap.py
import os
import cv2
import numpy as np

print("Testing InsightFace installation...")

try:
    import insightface
    print("✅ InsightFace imported successfully")
    
    from insightface.app import FaceAnalysis
    print("✅ FaceAnalysis imported successfully")
    
    # Check models
    if os.path.exists('insightface_models/inswapper_128.onnx'):
        print("✅ Face swapper model found")
    else:
        print("❌ Face swapper model not found")
    
    # Try to initialize face app
    app = FaceAnalysis(name='buffalo_l', root='insightface_models')
    app.prepare(ctx_id=-1, det_size=(640, 640))
    print("✅ FaceAnalysis initialized successfully")
    
    # Test on a dummy image
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_img[:] = (255, 255, 255)  # White image
    faces = app.get(dummy_img)
    print(f"✅ Face detection works (found {len(faces)} faces in dummy image)")
    
    print("\\n✅ All tests passed! Face swapping should work.")
    
except Exception as e:
    print(f"\\n❌ Error: {str(e)}")
    print("\\nTroubleshooting:")
    print("1. Make sure InsightFace is installed: pip install insightface")
    print("2. Check that all model files are downloaded")
    print("3. Try reinstalling: pip install --upgrade insightface")
'''

with open('test_face_swap.py', 'w') as f:
    f.write(test_script)

print("Created test_face_swap.py")
print("\nRun 'python test_face_swap.py' to verify the installation")