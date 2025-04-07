import logging
import time

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def timing_decorator(func):
    """A simple decorator to measure the execution time of a function."""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"Function '{func.__name__}' executed in {duration:.4f} seconds.")
        return result
    return wrapper

# Example of another utility function (optional)
def clean_text(text):
    """
    Simple text cleaning function.
    (Expand this based on actual needs, e.g., removing extra whitespace, special chars)
    """
    if isinstance(text, str):
        # Example: Remove leading/trailing whitespace and excessive spaces
        cleaned = ' '.join(text.strip().split())
        return cleaned
    return text

# You can add more utility functions here as your project grows.
# For example:
# - Functions for handling specific file types other than PDF
# - More advanced text cleaning or normalization functions
# - Helper functions for Streamlit UI elements
# - Functions for managing temporary files or directories safely 