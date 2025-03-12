import os
import logging
import re
from ui.gradio_ui import create_ui

# Create a custom filter to completely suppress specific logs
class HttpxFilter(logging.Filter):
    """Filter to completely suppress httpx connection logs"""
    def filter(self, record):
        # Return False for logs about HTTP requests to queue.fal.run
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            if 'HTTP Request:' in record.msg and 'queue.fal.run' in record.msg:
                return False
        return True

# Configure root logger first
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add our custom filter to the root logger
root_logger.addFilter(HttpxFilter())

# Create a console handler with the filter
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
root_logger.handlers = [console_handler]  # Replace any existing handlers

# Suppress verbose logs from external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("gradio").setLevel(logging.WARNING)

# Set debug level of specific modules that generate polling logs to WARNING to hide them
for module in ["utils.fal_client", "fal_client", "httpx"]:
    logging.getLogger(module).setLevel(logging.INFO)  # Only show INFO and above (not DEBUG)

if __name__ == "__main__":
    # Create and launch the UI
    iface = create_ui()
    iface.launch(share=False, debug=True)