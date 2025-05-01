import re
from datasets import load_dataset, DatasetDict
from utils.preprocess import preprocess_text
from config.config import Config

def load_and_preprocess_dataset():
    """Loads and preprocesses the dataset."""
    print("Loading dataset...")
    ds = load_dataset(Config.DATASET_NAME)

    print("Preprocessing dataset...")
    ds = ds.map(lambda x: {'EN': preprocess_text(x['EN']), 'UK': preprocess_text(x['UK'])})

    print("Splitting dataset...")
    train_testvalid = ds['test'].train_test_split(test_size=0.2)
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5)

    return DatasetDict({
        'train': train_testvalid['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    })
