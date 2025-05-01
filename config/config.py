import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    WIKIDATA_SPARQL_URL = os.getenv("WIKIDATA_SPARQL_URL")
    DATASET_NAME = os.getenv("DATASET_NAME")
    MODEL_NAME = os.getenv("MODEL_NAME")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR")
    LOG_DIR = os.getenv("LOG_DIR")
