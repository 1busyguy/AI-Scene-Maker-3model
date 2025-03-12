import os
import logging
from datetime import datetime
import requests
from config import OUTPUT_DIR

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s")

def ensure_directory_exists(directory: str):
    """Ensure the specified directory exists, create if not."""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
        logging.info(f"Created directory: {directory}")

def generate_timestamped_filename(prefix: str, extension: str = "mp4") -> str:
    """Generate a timestamped filename for uniqueness."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

def download_file(url: str, save_path: str):
    """Download a file from a URL to a local path."""
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    logging.info(f"Downloaded file to: {save_path}")

def log_info(message: str):
    """Log an informational message."""
    logging.info(message)

def log_error(message: str):
    """Log an error message."""
    logging.error(message)

def sanitize_filename(filename: str) -> str:
    """Sanitize filenames to remove problematic characters."""
    return "".join(c for c in filename if c.isalnum() or c in (' ', '_', '-')).rstrip()

def get_output_filepath(filename: str) -> str:
    """Generate a full file path within the output directory."""
    ensure_directory_exists(OUTPUT_DIR)
    sanitized_filename = sanitize_filename(filename)
    return os.path.join(OUTPUT_DIR, sanitized_filename)