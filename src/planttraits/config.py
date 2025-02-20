import os

from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.
root_dir = Path(os.getenv('ROOT_DIR'))

TEST_IMAGES_FOLDER = root_dir / os.getenv('TEST_IMAGES_FOLDER')
TRAIN_IMAGES_FOLDER = root_dir / os.getenv('TRAIN_IMAGES_FOLDER')
TEST_CSV_FILE = root_dir / os.getenv('TEST_CSV_FILE')
TRAIN_CSV_FILE = root_dir / os.getenv('TRAIN_CSV_FILE')
META_CSV_FILE = root_dir / os.getenv('META_CSV_FILE')
