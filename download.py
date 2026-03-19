# Download files from GitLab
import requests
from tqdm import tqdm

url = "https://git.cs.uni-paderborn.de/cn/moragbench-artifacts/-/raw/main/artifacts.zip"

with requests.get(url, stream=True) as r:
    r.raise_for_status()
    
    total_size = int(r.headers.get("content-length", 0))
    chunk_size = 8192

    with open("artifacts.zip", "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc="Downloading artifacts.zip"
    ) as pbar:
        for chunk in r.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

# Extract files
import zipfile

print("Extracting artifacts...")
zip_path = "artifacts.zip"
extract_to = "."

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to)
    

# Move files to the right place
import shutil
from pathlib import Path

# Tuple format: (source, destination, name_in_destination)
files_to_move = [
    ("artifacts/faiss_arm64-v8a", "android/faiss/src/main/jniLibs/", "arm64-v8a"),
    ("artifacts/main_assets", "android/app/src/main/", "assets"),
    ("artifacts/sentenced_embeddings_build", "android/embedding/sentence_embeddings/", "build"),
    ("artifacts/libqwen_tokenizer.so", "android/llm/src/main/jniLibs/arm64-v8a", "libqwen_tokenizer.so")
]

def merge_dirs(src: Path, dst: Path):
    """Merge src directory into dst directory."""
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                merge_dirs(item, target)
            else:
                shutil.move(str(item), str(target))
        else:
            shutil.move(str(item), str(target))

print("Moving files...")
for src_path, dst_dir, new_name in files_to_move:
    src = Path(src_path)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)

    final_dst = dst_dir / new_name

    if not final_dst.exists():
        shutil.move(str(src), str(final_dst))
    else:
        if src.is_dir() and final_dst.is_dir():
            merge_dirs(src, final_dst)
            shutil.rmtree(src)
        else:
            # overwrite file
            shutil.move(str(src), str(final_dst))

print("Done")