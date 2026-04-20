import kagglehub
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

print("Downloading BDD100K...")
path = kagglehub.dataset_download("solesensei/solesensei_bdd100k")
print(f"Downloaded to: {path}")

dest = DATA_DIR / "bdd100k"
if not dest.exists():
    shutil.copytree(path, dest)
    print(f"Copied to: {dest}")
else:
    print(f"Already exists at: {dest}")
