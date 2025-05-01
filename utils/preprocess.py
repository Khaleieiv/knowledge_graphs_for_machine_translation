import re

def preprocess_text(text):
    """Cleans and normalizes text."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

