import os
import zipfile
import requests

def download_and_extract_zip():
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        raise ValueError("KAGGLE_API_TOKEN environment variable not set")

    url = "https://www.kaggle.com/api/v1/kernels/output/download/anamariabarrosnonato/sensecv-six-model-dataset-audit"
    headers = {"Authorization": f"Bearer {token}"}
    zip_path = "output_audit.zip"
    output_dir = "output_audit"

    print(f"Downloading ZIP archive from Kaggle...")
    r = requests.get(url, headers=headers, stream=True)
    r.raise_for_status()

    content_length = r.headers.get("Content-Length")
    if content_length:
        total_size = int(content_length)
        print(f"Total zip size: {total_size / (1024*1024):.2f} MB")
    else:
        print("Total zip size: chunked stream")

    downloaded = 0
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024): # 1MB chunks
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                print(f"Downloaded {downloaded / (1024*1024):.2f} MB...", end="\r")

    print(f"\nDownload complete! Zip saved to {zip_path} ({os.path.getsize(zip_path)} bytes)")

    print(f"Extracting {zip_path} into {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    print("Extraction complete! All files ready in output_audit directory.")

if __name__ == "__main__":
    download_and_extract_zip()
