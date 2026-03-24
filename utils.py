import gdown
import os

def download_video(url: str, output_path: str) -> str:
    result = gdown.download(url, str(output_path), quiet=False, fuzzy=True)
    if result is None:
        raise RuntimeError(f"Failed to download video: {url}")
    return result
